//! Multi-head attention with RoPE and causal masking.
//!
//! Supports both MHA (encoder) and GQA (LLM) configurations.

use burn::config::Config;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Float, Tensor};

/// CPU-based stable softmax — avoids burn's max_dim/sum_dim reduction kernels
/// which are buggy on DZN (Vulkan-over-DX12).
/// For single-token decode, scores are [B,H,1,KV] ≈ 3000 elements — fast on CPU.
/// GPU-only softmax using matmul-based sum (avoids broken max_dim/sum_dim on DZN).
///
/// Approach: clamp scores to [-100, 80], then compute exp/sum without max-shift.
/// - Masked positions (-inf → clamped to -100): exp(-100) ≈ 3.7e-44 ≈ 0 ✓
/// - Valid scores (typically -10 to +10 after 1/√d scaling): exp fine in f32 range
/// - Upper clamp 80: exp(80) = 5.5e34, safely within f32 max (3.4e38)
fn gpu_softmax<B: Backend>(scores: Tensor<B, 4>) -> Tensor<B, 4> {
    let [b, h, s, kv] = scores.dims();
    let device = scores.device();

    // Clamp: -inf → -100 (becomes ≈0 after exp), cap at 80 (exp(80) < f32::MAX)
    let flat = scores.clamp(-100.0, 80.0).reshape([b * h * s, kv]);

    // exp(clamped_scores) — all values safely in f32 range
    let exp_vals = flat.exp();

    // sum per row via matmul: [N, KV] @ [KV, 1] → [N, 1]
    let ones = Tensor::<B, 2>::ones([kv, 1], &device);
    let row_sum = exp_vals.clone().matmul(ones);

    // normalize
    let result = exp_vals / (row_sum + 1e-10);

    result.reshape([b, h, s, kv])
}

use super::kv_cache::KVCache;
use super::rope::RoPE;

/// Attention configuration.
#[derive(Config, Debug)]
pub struct AttentionConfig {
    /// Model dimension.
    pub d_model: usize,
    /// Number of query heads.
    pub n_heads: usize,
    /// Number of KV heads (for GQA). If None, uses n_heads (MHA).
    pub n_kv_heads: Option<usize>,
    /// Head dimension (usually d_model / n_heads).
    pub head_dim: usize,
    /// Whether to use bias on Q projection.
    #[config(default = false)]
    pub q_bias: bool,
    /// Whether to use bias on K projection.
    #[config(default = false)]
    pub k_bias: bool,
    /// Whether to use bias on V projection.
    #[config(default = false)]
    pub v_bias: bool,
    /// Whether to use bias on O projection.
    #[config(default = false)]
    pub o_bias: bool,
    /// Sliding window size (None for full attention).
    pub sliding_window: Option<usize>,
}

/// Multi-head attention layer.
#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    pub(crate) wq: Linear<B>,
    pub(crate) wk: Linear<B>,
    pub(crate) wv: Linear<B>,
    pub(crate) wo: Linear<B>,
    pub(crate) n_heads: usize,
    pub(crate) n_kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) scale: f32,
    pub(crate) sliding_window: Option<usize>,
}

impl AttentionConfig {
    /// Initialize the attention layer.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Attention<B> {
        let n_kv_heads = self.n_kv_heads.unwrap_or(self.n_heads);

        let wq = LinearConfig::new(self.d_model, self.n_heads * self.head_dim)
            .with_bias(self.q_bias)
            .init(device);
        let wk = LinearConfig::new(self.d_model, n_kv_heads * self.head_dim)
            .with_bias(self.k_bias)
            .init(device);
        let wv = LinearConfig::new(self.d_model, n_kv_heads * self.head_dim)
            .with_bias(self.v_bias)
            .init(device);
        let wo = LinearConfig::new(self.n_heads * self.head_dim, self.d_model)
            .with_bias(self.o_bias)
            .init(device);

        Attention {
            wq,
            wk,
            wv,
            wo,
            n_heads: self.n_heads,
            n_kv_heads,
            head_dim: self.head_dim,
            scale: (self.head_dim as f32).powf(-0.5),
            sliding_window: self.sliding_window,
        }
    }
}

impl<B: Backend> Attention<B> {
    /// Create attention from linear layers (for weight loading).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        wq: Linear<B>,
        wk: Linear<B>,
        wv: Linear<B>,
        wo: Linear<B>,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        sliding_window: Option<usize>,
    ) -> Self {
        Self {
            wq,
            wk,
            wv,
            wo,
            n_heads,
            n_kv_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            sliding_window,
        }
    }

    /// Forward pass with RoPE.
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq, d_model]
    /// * `rope` - Rotary position embeddings
    /// * `offset` - Position offset for KV cache
    /// * `causal` - Whether to apply causal masking
    ///
    /// # Returns
    /// Output tensor [batch, seq, d_model]
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &RoPE<B>,
        offset: usize,
        causal: bool,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _d_model] = x.dims();

        // Project Q, K, V
        let q = self.wq.forward(x.clone());
        let k = self.wk.forward(x.clone());
        let v = self.wv.forward(x);

        // Reshape to [batch, seq, heads, head_dim]
        let q = q.reshape([batch, seq_len, self.n_heads, self.head_dim]);
        let k = k.reshape([batch, seq_len, self.n_kv_heads, self.head_dim]);
        let v = v.reshape([batch, seq_len, self.n_kv_heads, self.head_dim]);

        // Apply RoPE
        let (q, k) = rope.apply(q, k, offset);

        // Transpose to [batch, heads, seq, head_dim]
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        // Expand K, V for GQA if needed
        let (k, v) = self.expand_kv(k, v);

        // Compute attention scores: Q @ K^T * scale
        let k_t = k.swap_dims(2, 3);
        let scores = q.matmul(k_t) * self.scale;

        // Apply causal mask
        let scores = if causal {
            self.apply_causal_mask(scores, seq_len, offset)
        } else {
            scores
        };

        // Apply sliding window mask if configured
        let scores = if let Some(window) = self.sliding_window {
            self.apply_sliding_window_mask(scores, seq_len, window)
        } else {
            scores
        };

        // Softmax
        let attn = gpu_softmax(scores);

        // Apply attention: attn @ V
        let out = attn.matmul(v);

        // Transpose back and reshape: [batch, heads, seq, head_dim] -> [batch, seq, heads * head_dim]
        let out = out.swap_dims(1, 2);
        let out = out.reshape([batch, seq_len, self.n_heads * self.head_dim]);

        // Output projection
        self.wo.forward(out)
    }

    /// Forward pass with KV cache.
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq, d_model]
    /// * `rope` - Rotary position embeddings
    /// * `cache` - Mutable KV cache (updated in place)
    /// * `causal` - Whether to apply causal masking
    ///
    /// # Returns
    /// Output tensor [batch, seq, d_model]
    pub fn forward_with_cache(
        &self,
        x: Tensor<B, 3>,
        rope: &RoPE<B>,
        cache: &mut KVCache<B>,
        causal: bool,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _d_model] = x.dims();

        // Position offset is the current cache length
        let offset = cache.seq_len();

        // Project Q, K, V
        let q = self.wq.forward(x.clone());
        let k = self.wk.forward(x.clone());
        let v = self.wv.forward(x);

        // Reshape to [batch, seq, heads, head_dim]
        let q = q.reshape([batch, seq_len, self.n_heads, self.head_dim]);
        let k = k.reshape([batch, seq_len, self.n_kv_heads, self.head_dim]);
        let v = v.reshape([batch, seq_len, self.n_kv_heads, self.head_dim]);

        // Apply RoPE to new Q, K (with correct positional offset)
        let (q, k) = rope.apply(q, k, offset);

        // Transpose to [batch, heads, seq, head_dim]
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        // Update cache and get full K, V sequences
        let (k, v) = cache.update(k, v);

        // Total sequence length after cache update
        let total_seq_len = cache.seq_len();

        // Expand K, V for GQA if needed
        let (k, v) = self.expand_kv(k, v);

        // Compute attention scores: Q @ K^T * scale
        // Q: [batch, heads, seq_len, head_dim]
        // K: [batch, heads, total_seq_len, head_dim]
        // scores: [batch, heads, seq_len, total_seq_len]
        let k_t = k.swap_dims(2, 3);
        let scores = q.matmul(k_t) * self.scale;

        // Apply causal mask (accounts for different query/key lengths)
        let scores = if causal {
            self.apply_causal_mask_with_offset(scores, seq_len, total_seq_len, offset)
        } else {
            scores
        };

        // Apply sliding window mask if configured
        let scores = if let Some(window) = self.sliding_window {
            self.apply_sliding_window_mask_with_offset(
                scores,
                seq_len,
                total_seq_len,
                window,
                offset,
            )
        } else {
            scores
        };

        // Softmax
        let attn = gpu_softmax(scores);

        // Apply attention: attn @ V
        let out = attn.matmul(v);

        // Transpose back and reshape
        let out = out.swap_dims(1, 2);
        let out = out.reshape([batch, seq_len, self.n_heads * self.head_dim]);

        // Output projection
        self.wo.forward(out)
    }

    /// Expand K, V heads for GQA (grouped-query attention).
    fn expand_kv(&self, k: Tensor<B, 4>, v: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        if self.n_heads == self.n_kv_heads {
            return (k, v);
        }

        let repeat_factor = self.n_heads / self.n_kv_heads;
        let [batch, n_kv_heads, seq, head_dim] = k.dims();

        // Repeat: [batch, n_kv_heads, seq, head_dim] -> [batch, n_heads, seq, head_dim]
        let k = k
            .unsqueeze_dim::<5>(2) // [batch, n_kv_heads, 1, seq, head_dim]
            .repeat_dim(2, repeat_factor) // [batch, n_kv_heads, repeat, seq, head_dim]
            .reshape([batch, n_kv_heads * repeat_factor, seq, head_dim]);
        let v = v
            .unsqueeze_dim::<5>(2)
            .repeat_dim(2, repeat_factor)
            .reshape([batch, n_kv_heads * repeat_factor, seq, head_dim]);

        (k, v)
    }

    /// Apply causal mask to attention scores.
    fn apply_causal_mask(
        &self,
        scores: Tensor<B, 4>,
        seq_len: usize,
        _offset: usize,
    ) -> Tensor<B, 4> {
        super::masking::apply_causal_mask(scores, seq_len)
    }

    /// Apply sliding window mask to attention scores.
    fn apply_sliding_window_mask(
        &self,
        scores: Tensor<B, 4>,
        seq_len: usize,
        window: usize,
    ) -> Tensor<B, 4> {
        super::masking::apply_sliding_window_mask(scores, seq_len, window)
    }

    /// Apply causal mask with different query/key lengths (for KV cache).
    fn apply_causal_mask_with_offset(
        &self,
        scores: Tensor<B, 4>,
        q_len: usize,
        kv_len: usize,
        offset: usize,
    ) -> Tensor<B, 4> {
        super::masking::apply_causal_mask_with_offset(scores, q_len, kv_len, offset)
    }

    /// Apply sliding window mask with different query/key lengths (for KV cache).
    fn apply_sliding_window_mask_with_offset(
        &self,
        scores: Tensor<B, 4>,
        q_len: usize,
        kv_len: usize,
        window: usize,
        offset: usize,
    ) -> Tensor<B, 4> {
        super::masking::apply_sliding_window_mask_with_offset(scores, q_len, kv_len, window, offset)
    }
}

/// Create causal attention mask.
pub fn create_causal_mask<B: Backend>(seq_len: usize, device: &B::Device) -> Tensor<B, 4, Float> {
    let mut mask_data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                mask_data[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
    }

    let mask: Tensor<B, 1> = Tensor::from_floats(mask_data.as_slice(), device);
    let mask: Tensor<B, 2> = mask.reshape([seq_len, seq_len]);
    mask.unsqueeze_dim::<3>(0).unsqueeze_dim(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::kv_cache::KVCache;
    use super::super::rope::RoPEConfig;
    use burn::backend::Wgpu;

    type TestBackend = Wgpu;

    #[test]
    fn test_attention_shape() {
        let device = Default::default();

        // MHA config (encoder-style)
        let config = AttentionConfig::new(64, 4, 16);
        let attn = config.init::<TestBackend>(&device);

        let rope = RoPEConfig::new(16, 512).init::<TestBackend>(&device);

        let x = Tensor::<TestBackend, 3>::zeros([2, 10, 64], &device);
        let out = attn.forward(x, &rope, 0, true);

        assert_eq!(out.dims(), [2, 10, 64]);
    }

    #[test]
    fn test_attention_gqa_shape() {
        let device = Default::default();

        // GQA config (LLM-style: 32Q/8KV)
        let config = AttentionConfig::new(256, 8, 32).with_n_kv_heads(Some(2));
        let attn = config.init::<TestBackend>(&device);

        let rope = RoPEConfig::new(32, 512).init::<TestBackend>(&device);

        let x = Tensor::<TestBackend, 3>::zeros([1, 20, 256], &device);
        let out = attn.forward(x, &rope, 0, true);

        assert_eq!(out.dims(), [1, 20, 256]);
    }

    #[test]
    fn test_attention_with_cache() {
        let device = Default::default();

        // MHA config
        let config = AttentionConfig::new(64, 4, 16);
        let attn = config.init::<TestBackend>(&device);

        let rope = RoPEConfig::new(16, 512).init::<TestBackend>(&device);
        let mut cache: KVCache<TestBackend> = KVCache::new();

        // First forward: 5 tokens
        let x1 = Tensor::<TestBackend, 3>::zeros([1, 5, 64], &device);
        let out1 = attn.forward_with_cache(x1, &rope, &mut cache, true);
        assert_eq!(out1.dims(), [1, 5, 64]);
        assert_eq!(cache.seq_len(), 5);

        // Second forward: 3 tokens (incremental)
        let x2 = Tensor::<TestBackend, 3>::zeros([1, 3, 64], &device);
        let out2 = attn.forward_with_cache(x2, &rope, &mut cache, true);
        assert_eq!(out2.dims(), [1, 3, 64]);
        assert_eq!(cache.seq_len(), 8);

        // Third forward: 1 token (typical autoregressive step)
        let x3 = Tensor::<TestBackend, 3>::zeros([1, 1, 64], &device);
        let out3 = attn.forward_with_cache(x3, &rope, &mut cache, true);
        assert_eq!(out3.dims(), [1, 1, 64]);
        assert_eq!(cache.seq_len(), 9);
    }

    #[test]
    fn test_attention_cache_vs_full() {
        let device = Default::default();

        // MHA config
        let config = AttentionConfig::new(64, 4, 16);
        let attn = config.init::<TestBackend>(&device);

        let rope = RoPEConfig::new(16, 512).init::<TestBackend>(&device);

        // Create input tensors (use constant values for deterministic test)
        let x1 = Tensor::<TestBackend, 3>::ones([1, 3, 64], &device) * 0.5;
        let x2 = Tensor::<TestBackend, 3>::ones([1, 2, 64], &device) * 0.3;

        // Full forward (no cache) - concatenated input
        let x_full = Tensor::cat(vec![x1.clone(), x2.clone()], 1);
        let out_full = attn.forward(x_full, &rope, 0, true);

        // With cache: first chunk, then second
        let mut cache: KVCache<TestBackend> = KVCache::new();
        let _out1 = attn.forward_with_cache(x1, &rope, &mut cache, true);
        let out2 = attn.forward_with_cache(x2, &rope, &mut cache, true);

        // The output for the second chunk should match the corresponding
        // positions in the full forward
        let out_full_slice = out_full.slice([0..1, 3..5, 0..64]);

        let out2_data = out2.to_data();
        let out_full_slice_data = out_full_slice.to_data();

        let out2_slice = out2_data.as_slice::<f32>().unwrap();
        let out_full_slice_slice = out_full_slice_data.as_slice::<f32>().unwrap();

        let mut max_diff: f32 = 0.0;
        for (a, b) in out2_slice.iter().zip(out_full_slice_slice.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }

        println!("Cache vs full max diff: {:.2e}", max_diff);
        assert!(
            max_diff < 1e-5,
            "Cache output should match full forward. Max diff: {:.2e}",
            max_diff
        );
    }
}
