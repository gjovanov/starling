# Pending Decode Optimizations

Current: **14.6ms/step** (candle-native-flash, RTX 5090, BF16)
Theoretical floor: **~3.8ms** (memory bandwidth for weight reads)

## Bottleneck Analysis

74% of decode time is candle's per-op dispatch overhead, not GPU compute.

```
14.6ms/step breakdown:
├── GPU compute (bandwidth-limited)     3.8ms  (26%)
│   ├── FFN matmuls (gate+up 113MB, down 57MB per layer)  2.4ms
│   ├── Attn matmuls (QKV 38MB, output 25MB per layer)    0.9ms
│   └── lm_head (805MB single read)                       0.5ms
├── Candle dispatch overhead                              10.8ms  (74%)
│   └── ~500 ops × ~22μs/op (Tensor creation, layout check, Arc, alloc)
```

## Option A: Raw cudarc Decode Path

Bypass candle's Tensor abstraction entirely for the decode hot path. Call cuBLAS and CUDA kernels directly via cudarc.

**How:**
1. At model load, extract raw GPU pointers from all decoder weight tensors
2. Pre-allocate all intermediate buffers for seq=1 decode (fixed shapes)
3. Implement `raw_decoder_step()` using direct cuBLAS gemm + fused kernel launches
4. Reuse candle's existing CUDA kernels (RmsNorm, RoPE, SiLU) by calling them via cudarc

**Expected:** ~5-6ms/step (dispatch drops from ~22μs/op to ~3μs/op)
**Effort:** 3-5 days
**Risk:** High — lots of unsafe code, pointer arithmetic, no candle safety net
**Quality:** Preserved (same cuBLAS + flash_attn kernels)

### Key implementation details:
- Weight pointers: `tensor.storage_and_layout()` → `CudaStorage` → `CudaSlice` → raw ptr
- cuBLAS: use `device.blas` handle, call `gemm_strided_batched_bf16` directly
- Flash attention: call `candle_flash_attn` FFI, or use flash_attn_varlen with explicit seqlens
- KV cache: raw `CudaSlice` with manual offset tracking (no Tensor narrow/slice_set)
- Custom kernels: compile via cudarc nvrtc or link pre-compiled .ptx

## Option B: CUDA Graph Capture

Record the entire decode forward as a CUDA graph, replay it each step with zero dispatch.

**How:**
1. Pre-allocate fixed-capacity KV caches (no batch dim: `[capacity, heads, hd]`)
2. Switch to `flash_attn_varlen` (already done) with GPU-side seqlen management
3. Move KV cache offset tracking to GPU (atomic counter or pre-computed offsets)
4. `stream.begin_capture()` → run one decode step → `stream.end_capture()` → `graph.launch()`

**Expected:** ~4-5ms/step
**Effort:** 5-7 days
**Risk:** Very high — KV cache management must move entirely to GPU
**Quality:** Preserved

### Key constraints:
- `CudaDevice::cuda_stream()` returns `Arc<CudaStream>` with `begin_capture`/`end_capture`
- cudarc 0.19.4 has full graph API (`CudaGraph::launch()`)
- KV cache `slice_set` uses Rust-side offset → won't replay correctly in graph
- `flash_attn_varlen` seqlens must be at fixed GPU addresses, values updated before launch
- Re-capture needed when KV shape changes (or use graph node parameter updates)

## Option C: tch-rs Hybrid Decoder

Use PyTorch's C++ backend (via libtorch) for just the decoder forward.

**How:**
1. Add `tch` dependency, link libtorch
2. Convert decoder weights to `tch::Tensor` at load time
3. Implement decoder forward in tch (PyTorch dispatch: ~25μs/op vs candle's ~22μs)
4. Use `torch.compile()` or `torch.cuda.make_graphed_callables()` for CUDA graph capture

**Expected:** ~8-10ms without compile, ~5ms with CUDA graphs
**Effort:** 2-3 days for basic, +2 days for torch.compile
**Risk:** Medium — libtorch adds ~2GB to binary, memory sharing complexity
**Quality:** Preserved

### Evaluation (from plan):
- PyTorch's per-op dispatch is ~25μs vs candle's ~22μs — similar, so raw tch without compile gives NO benefit
- The real win is `torch.compile` / CUDA graph capture which handles KV cache natively
- libtorch packaging is complex on Linux (CUDA version must match exactly)

## Option D: INT8/FP8 Quantized Decoder

RTX 5090 has FP8 tensor cores. Quantizing decoder weights to FP8 halves memory reads.

**Expected:** ~2.5ms compute (vs 3.8ms) but dispatch overhead unchanged → ~13ms total
**Effort:** 3-5 days (calibration, custom quantization, validation)
**Risk:** Medium — quality may degrade without careful calibration
**Only worth it after** dispatch overhead is solved (Option A/B/C)

## Option E: Speculative Pad-Skip

~80% of decode positions output pad token 32. If we can predict "this will be pad" cheaply, skip the full decoder forward for those positions.

**How:**
1. Train a tiny classifier on encoder output to predict pad vs text
2. For predicted-pad positions, output 32 directly (skip 26-layer decoder)
3. For predicted-text positions, run full decoder

**Expected:** ~5-8ms effective (skip 80% of decoder forwards)
**Effort:** 3 days (research + implementation)
**Risk:** High — mis-prediction causes quality loss

## Attempted and Rejected

### Batched Anchor-Request Decode (chunk_size=6)
Broadcast same `prev_token` to N positions, single forward pass. 4.8ms/step decode but **quality collapsed** — only 14 text tokens instead of 692 for 300s audio. Voxtral's autoregressive decode requires position i+1 to see token generated at position i. Any chunk_size > 1 breaks this.

### cuBLAS FAST_BF16 Mode
`set_gemm_reduced_precision_bf16(true)` switches accumulation from F32 to BF16. No measurable speedup — GEMV is bandwidth-limited, not compute-limited.

## CPU Inference: candle-cpu-ggml (Current Best: 1.91× realtime)

Achieved via `candle-cpu-ggml` feature: ggml graph API for Q4 matmul + incremental mel/conv + aggressive KV compaction.

**Performance on AMD Ryzen 9 9955HX3D (16c/32t, 96MB L3, DDR5-5600):**

| Per 0.5s commit | Time | Notes |
|---|---|---|
| Mel (incremental) | ~5ms | Cached, only new samples |
| Conv (incremental) | ~10ms | Cached, only new frames |
| Encoder (32 layers) | 141-195ms | 24-28 new frames, KV compacted at 200 |
| Decoder (6-7 steps) | 577-753ms | 85ms/step via ggml AVX-512 |
| **Total** | **724-955ms** | **1.91× realtime** |

### Key optimizations applied:
1. **ggml graph API** — llama.cpp's ggml_mul_mat with AVX-512 + VNNI (2.6× faster than candle AVX2)
2. **Incremental mel** — cache mel output, only recompute on new audio samples with overlap
3. **Incremental conv** — cache conv output, only run conv on new mel frames
4. **Aggressive KV compaction** — compact at 200 positions (not 750) to keep 32-layer KV cache in 96MB L3
5. **Zero-copy input** — access CpuStorage f32 slice directly, no to_vec1() copy
6. **Persistent work buffer** — reuse ggml work buffer across matmul calls

### Remaining gap to real-time (1.91× → 1.0×):
The decoder at 85ms/step is the bottleneck (75-80% of commit time). The overhead is from:
- 182 ggml context create/free per step (~5ms)
- 182 Vec<f32> output allocations per step (~1ms)
- Candle non-matmul ops: RoPE, softmax, attention scores, norms (~20ms)
- Rust↔C boundary crossing overhead (~5ms)
- Q4 dequant+vec_dot compute (4× above DDR5 bandwidth floor)

### Path to 1.0× (not implemented):
- **Option F: Full ggml decoder graph** — build entire decoder step as single ggml computation graph (eliminates per-matmul context overhead). Requires implementing attention, softmax, RoPE, norms as ggml ops. Effort: 5+ days. Essentially rewriting the decoder in C.
- **Option G: Hybrid GPU encoder + CPU decoder** — run encoder on GPU (trivially fast), stream adapter tokens to CPU. Decoder at 600ms/commit is close to 500ms budget.
- **Option H: llama.cpp Voxtral support** — contribute Voxtral decoder to llama.cpp's ggml. Their runtime runs the full model without Rust↔C boundary. Expected ~30ms/step based on 7B benchmarks.

## Environment Variables

| Variable | Effect |
|----------|--------|
| `CANDLE_PROFILE=1` | Per-section GPU timing with stream sync (adds ~4ms overhead) |
| `CANDLE_FAST_BF16=1` | cuBLAS reduced precision (COMPUTE_32F_FAST_16BF) |
| `CANDLE_STREAMING=1` | Use streaming transcribe (chunked encoder + context rotation) |
| `CANDLE_CHUNK_SIZE=N` | Batched decode chunk size (default 1, >1 breaks quality) |
| `CANDLE_NATIVE_LAYERS=N` | Limit decoder to first N layers (for debugging) |
| `GGML_THREADS=16` | Thread count for ggml matmul (default: 16) |
| `CANDLE_CPU_F32_DECODER=1` | Dequantize decoder to F32 (candle-cpu only, slower) |
