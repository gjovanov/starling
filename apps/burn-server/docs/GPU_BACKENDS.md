# GPU Backend Options for burn-server

## Current State

burn-server uses **Burn 0.20** with the **wgpu** backend (WebGPU API). wgpu abstracts over multiple GPU APIs:
- **Vulkan** on Linux
- **Metal** on macOS
- **DX12** on Windows
- **WebGPU** in browsers (via WASM)

### WSL2 Problem

WSL2 with NVIDIA driver 591.86 does NOT expose Vulkan to Linux. wgpu falls back to `llvmpipe` (CPU software renderer) which has a 128 MB buffer limit — too small for the Voxtral model. The reference implementation (voxtral-mini-realtime-rs) has the exact same failure on WSL2.

CUDA works perfectly on WSL2 (vllm-server uses it), but Burn's wgpu backend cannot use CUDA directly.

## Architecture

```
                    ┌─────────────────────────────────┐
                    │         Burn Model Code          │
                    │   (generic over B: Backend)      │
                    │   encoder, decoder, adapter,     │
                    │   attention, RoPE, RmsNorm       │
                    └──────────┬──────────┬────────────┘
                               │          │
                    ┌──────────▼──┐  ┌────▼──────────┐
                    │  BF16 path  │  │   Q4 path     │
                    │ (generic)   │  │ (GPU-specific) │
                    │ std Tensor  │  │ custom shaders │
                    └──────┬──────┘  └───┬───────┬───┘
                           │             │       │
          ┌────────────────▼─────────────▼───┐   │
          │            Server (native)       │   │
          │  Option A: burn-wgpu (Vulkan)    │   │
          │  Option B: burn-cuda (CUDA/PTX)  │   │
          │  Option C: burn-cuda (BF16 only) │   │
          └──────────────────────────────────┘   │
                                                 │
                    ┌────────────────────────────▼───┐
                    │       Browser (WASM)           │
                    │   burn-wgpu (WebGPU/WGSL)     │
                    └────────────────────────────────┘
```

### What works across backends (no changes needed)

The **BF16 path** uses standard `Tensor<B, N>` operations that compile to any Burn backend:
- All 10 layer modules: Attention, RoPE, KVCache, SwiGLU, Conv, etc.
- Switching backend is a one-line type alias change

### What is backend-specific

The **Q4 path** uses raw WGSL compute shaders (`shader.wgsl`, `shader_naive.wgsl`) loaded via `SourceKernel`. These are tied to the wgpu runtime and cannot run on CUDA.

## Options

### Option A: Migrate WSL to Ubuntu 24.04 (Recommended first step)

**Effort:** ~1 hour

Ubuntu 24.04 ships Mesa 24.0+ which includes the **DZN** (Dozen) driver — a Vulkan-on-DX12 translation layer. This exposes the NVIDIA GPU as a Vulkan device to wgpu, bypassing the llvmpipe CPU fallback.

**What to do:**
1. Back up current WSL2 distro (script: `~/wsl-migrate-backup.sh`)
2. Install Ubuntu 24.04 WSL2 distro
3. Restore home, /pcon, packages
4. Verify `vulkaninfo` shows NVIDIA GPU (not llvmpipe)

**Pros:**
- Zero code changes to burn-server
- Same code path for server and WASM browser
- Q4 WGSL shaders work as-is
- Proven approach (voxtral-mini-realtime-rs benchmarks on Vulkan)

**Cons:**
- DZN is relatively new (may have edge cases)
- Vulkan-on-DX12 adds a thin translation layer vs native CUDA
- Still depends on Mesa quality on WSL2

**Expected performance (from voxtral-mini-realtime-rs benchmarks):**

| Metric | Q4 Native | BF16 Native |
|--------|-----------|-------------|
| RTF | 0.416 | 1.543 |
| Tokens/s | 19.4 | 4.6 |
| VRAM | 703 MB | 9.2 GB |

### Option B: Dual backend — CubeCL Q4 kernels (Best long-term)

**Effort:** 2–3 days

Rewrite `shader.wgsl` and `shader_naive.wgsl` as CubeCL `#[cube]` kernels. CubeCL compiles the same Rust source to both **CUDA PTX** and **WGSL**:

```rust
#[cube(launch)]
fn q4_matmul_kernel(
    input: &Tensor<f32>,
    weights: &Array<u32>,
    output: &mut Tensor<f32>,
    // ...
) {
    // Same kernel, compiles to CUDA PTX AND WGSL
}
```

**What to do:**
1. Add `burn-cuda` feature flag to Cargo.toml
2. Rewrite ~230 lines of WGSL shaders as CubeCL `#[cube]` kernels
3. Update `q4_matmul` dispatch to use CubeCL API instead of `SourceKernel`
4. Add `Q4Tensor` variant that uploads to CUDA device memory
5. Feature-gate: `#[cfg(feature = "cuda")]` vs `#[cfg(feature = "wgpu")]`
6. Validate output matches across both backends

**Cargo.toml:**
```toml
[features]
default = ["cuda"]
cuda = ["burn/cuda", "cubecl/cuda"]
wgpu = ["burn/wgpu", "cubecl/wgpu"]
wasm = ["wgpu"]
```

**Pros:**
- Best performance on NVIDIA (native CUDA, tensor cores via WMMA)
- Single kernel source compiles to CUDA, Vulkan, Metal, WebGPU
- Future-proof: AMD ROCm/HIP comes for free via CubeCL
- No Vulkan/DX12 translation overhead on server
- Works on current WSL2 without migration

**Cons:**
- 2–3 days of kernel rewrite and validation
- CubeCL's `#[cube]` DSL has different constraints than raw WGSL
  (e.g., no raw pointer arithmetic, different shared memory syntax)
- Two feature flags to maintain and test
- CubeCL CUDA backend maturity (production-ready but newer than wgpu)

### Option C: CUDA for BF16 server, wgpu for Q4 WASM only

**Effort:** ~1 day

Use `burn-cuda` for BF16 inference on the server. Keep Q4+WGSL for the WASM browser path only.

**What to do:**
1. Add `burn-cuda` dependency with `cuda` feature
2. Change server `Backend` type alias from `Wgpu` to `CudaBackend`
3. BF16 model loads SafeTensors, runs on CUDA — no Q4 needed on server
4. Q4+WGSL stays for `--features wasm` (browser) builds only

```toml
[features]
default = ["server-cuda"]
server-cuda = ["burn/cuda"]
wasm = ["burn/wgpu"]
```

**Pros:**
- Works immediately on current WSL2 (CUDA already available)
- BF16 gives best accuracy (WER ~4.9% vs Q4's ~8.5%)
- No shader rewriting needed
- 24 GB VRAM is plenty for BF16 (~9 GB)
- Clear separation: server = accuracy, browser = portability

**Cons:**
- BF16 always needs ≥16 GB VRAM (no Q4 option on server)
- BF16 is slower than Q4 (RTF ~1.5 vs ~0.4)
- Two different model paths to maintain (BF16 server vs Q4 browser)
- Cannot deploy Q4 server on small GPUs (only Q4 runs in 700 MB)

## Decision Matrix

| Factor | Option A (WSL migrate) | Option B (CubeCL) | Option C (BF16 CUDA) |
|--------|----------------------|-------------------|---------------------|
| **Effort** | ~1 hour | 2–3 days | ~1 day |
| **Code changes** | None | Q4 kernel rewrite | Backend type change |
| **Server Q4** | Yes (Vulkan) | Yes (CUDA) | No (BF16 only) |
| **Server BF16** | Yes | Yes | Yes |
| **WASM browser** | Yes | Yes | Yes |
| **Current WSL2** | No (need 24.04) | Yes | Yes |
| **Performance** | Good (DX12→Vulkan) | Best (native CUDA) | OK (BF16 slower) |
| **VRAM (Q4)** | 700 MB | 700 MB | N/A |
| **VRAM (BF16)** | 9 GB | 9 GB | 9 GB |
| **Future-proof** | Medium | High (multi-GPU) | Medium |

## Recommended Path

1. **Now:** Option A — migrate WSL to Ubuntu 24.04 (~1 hour)
2. **If DZN performance is insufficient:** Option B — rewrite Q4 kernels in CubeCL
3. **Fallback:** Option C — BF16-only CUDA server if CUDA is needed immediately

## References

- [Burn backends](https://burn.dev/docs/burn/) — official Burn docs
- [CubeCL](https://github.com/tracel-ai/cubecl) — GPU compute abstraction (CUDA + wgpu + ROCm)
- [Mesa DZN](https://docs.mesa3d.org/drivers/dozen.html) — Vulkan-on-DX12 driver
- [voxtral-mini-realtime-rs](https://github.com/TrevorS/voxtral-mini-realtime-rs) — reference implementation
