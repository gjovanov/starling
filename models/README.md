# Shared Model Storage

All Starling apps share this directory for model weights to avoid duplicate downloads.

## Layout

```
models/
  download.sh                  # Download script (BF16 + Q4 + chunking)
  cache/
    bf16/                      # Voxtral-Mini-4B SafeTensors (~9 GB)
      consolidated.safetensors
      params.json
      tekken.json
    q4/                        # Voxtral-Mini-4B Q4_0 GGUF (~2.5 GB)
      voxtral-q4.gguf
      tekken.json
      chunks/                  # 64 MB shards for browser WASM loading
        chunk_aa, chunk_ab, ...
    tokenizer/                 # Shared Tekken tokenizer
      tekken.json
```

## Usage

```bash
./download.sh              # Download all (BF16 + Q4)
./download.sh --bf16-only  # BF16 SafeTensors only (~9 GB)
./download.sh --q4-only    # Q4 GGUF only (~2.5 GB)
./download.sh --chunk      # Also split GGUF into 64 MB shards for WASM
```

## Which App Uses What

| App | Model Variant | Size | Notes |
|-----|---------------|------|-------|
| **vllm-server** | BF16 SafeTensors | ~9 GB | Loaded by vLLM into GPU VRAM |
| **burn-server** (BF16 mode) | BF16 SafeTensors | ~9 GB | Loaded by Burn into GPU VRAM |
| **burn-server** (Q4 mode) | Q4_0 GGUF | ~2.5 GB | Fused dequant on GPU, ~700 MB VRAM |
| **burn-server** (WASM) | Q4_0 chunks | ~2.5 GB | Fetched by browser in 64 MB shards |

## Environment

Set `STARLING_MODELS_DIR` to override the default cache location:

```bash
export STARLING_MODELS_DIR=/path/to/custom/cache
```

Default: `./models/cache/` (relative to repo root).

## Model Sources

- **BF16**: [mistralai/Voxtral-Mini-4B-Realtime-2602](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)
- **Q4 GGUF**: [TrevorJS/voxtral-mini-realtime-gguf](https://huggingface.co/TrevorJS/voxtral-mini-realtime-gguf) (Burn-compatible format)
