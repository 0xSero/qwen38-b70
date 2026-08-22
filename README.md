# qwen38-b70

Qwen3.8-27B inference on **two Intel Arc Pro B70 GPUs**.

## vLLM (recommended — highest throughput)

Using the Intel lab vLLM image (`intel/llm-scaler-vllm:0.21.0-b3`) with
PIECEWISE cudagraph mode, AutoRound INT4 quantization, and MTP speculative decoding.

**See [`DEFAULT_CONFIG.md`](DEFAULT_CONFIG.md) for the verified default configuration.**

| Config | Single-stream | Concurrent ×8 | Use case |
|--------|-------------:|--------------:|----------|
| MTP2, seqs=8 | **85.3 tok/s** | 306.0 tok/s | Interactive / single-user |
| MTP3, seqs=8 | 76.4 tok/s | **383.3 tok/s** | Multi-user / serving |

Both configs are verified coherent. Entrypoint scripts in `entrypoints/`.
Full benchmark history in [`HILLCLIMB.md`](HILLCLIMB.md).

```bash
# Start the best concurrent config
docker run -d --name vllm-tp2-pw --privileged \
    --device /dev/dri:/dev/dri --device /dev/dri/by-path:/dev/dri/by-path \
    -v /path/to/model:/model:ro \
    -v /path/to/entrypoints/ep_mtp3_seqs8.sh:/entrypoint.sh:ro \
    -v /path/to/vllm_gloo_kernels.py:/tmp/vllm_gloo_kernels.py:ro \
    -v /path/to/compile_cache:/root/.cache/vllm/torch_compile_cache \
    -e VLLM_XPU_ENABLE_XPU_GRAPH=1 -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200 \
    -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 -e UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e ZES_ENABLE_SYSMAN=1 \
    -e CCL_ZE_IPC_EXCHANGE=sockets -e VLLM_LOGGING_LEVEL=INFO \
    --entrypoint /bin/bash intel/llm-scaler-vllm:0.21.0-b3 /entrypoint.sh
```

## llama.cpp SYCL (alternative — long context)

One-command Qwen3.8-27B inference server on **two Intel Arc Pro B70 GPUs**,
tensor-parallel llama.cpp (SYCL), 262k context, OpenAI-compatible API.

```bash
docker compose up -d
```

That is the whole setup. On first start it downloads the pinned,
SHA-256-verified model (~19 GB) into `./models`, then serves on port **8010**.

```bash
curl http://localhost:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b","max_tokens":64,
       "messages":[{"role":"user","content":"Hello"}]}'
```

## Measured performance (Q4_K_M, f16 KV)

### 2x B70, tensor parallel (`GPU_COUNT=2`)

| Prompt context | Prefill tok/s | TTFT    | Decode tok/s |
|----------------|--------------:|---------|-------------:|
| ~2.5k          | 955           | 2.1 s   | 51.1         |
| ~10k           | 881           | 9.1 s   | 50.9         |
| ~20k           | 801           | 20.0 s  | 51.1         |
| ~40k           | 694           | 46.2 s  | 49.7         |
| ~60k           | 563           | 106.6 s | 46.9         |
| ~160k          | 365           | 355.8 s | 42.0         |
| ~245k          | 275           | 727.3 s | 30.8         |

### 1x B70 (`GPU_COUNT=1`, max context 131,072)

| Prompt context | Prefill tok/s | TTFT    | Decode tok/s |
|----------------|--------------:|---------|-------------:|
| ~2.5k          | 1164          | 1.8 s   | 33.3         |
| ~10k           | 1018          | 7.9 s   | 33.4         |
| ~40k           | 785           | 40.8 s  | 33.4         |
| ~65k           | 625           | 96.1 s  | 31.9         |
| ~128k          | 426           | 282.1 s | 27.5         |

Single-GPU prefill **beats** TP2 at every context length (the all-reduce costs
more than the second card recovers); TP2 wins on decode and KV capacity.

### MTP speculative decoding (`ENABLE_MTP=1`)

| Config            | Task              | tok/s | Baseline | Draft acceptance |
|-------------------|-------------------|------:|---------:|-----------------:|
| 2x B70 (TP2)      | Easy (counting)   | 84.3  | 51.1     | 97.2%           |
| 2x B70 (TP2)      | Hard (random gen) | 49.0  | 51.1     | 37.5%           |
| 1x B70 (64k ctx)  | Easy (counting)   | 61.7  | 33.3     | 97.2%           |
| 1x B70 (64k ctx)  | Hard (random gen) | 28.2  | 33.3     | 29.6%           |

Note: on 1x B70 the draft model plus 131k context exceeds VRAM
(UR_RESULT_ERROR_DEVICE_LOST); 65536 is the validated MTP context on a single card.

Decode is nearly flat with context: only 16 of 64 layers are full attention
(Qwen3.8 hybrid GDN); the linear-attention layers are O(1) per token.

## What is inside

- **Source**: [mndodd/llama.cpp](https://github.com/mndodd/llama.cpp) @
  `4302fb5` + the
  [b70-optimization-lab](https://github.com/steveseguin/b70-optimization-lab)
  full TP2 stack (`dp4a2` patch, decoded SHA-256
  `f21e9b55…`) + Q4K increment (`0a278585…`). Both digests verified against
  the lab's published values.
- **Model**: `ggml-org/Qwen3.8-27B-GGUF` @ `0669b986`, `Qwen3.8-27B-Q4_K_M.gguf`,
  SHA-256 checked at every container start.
- **Build**: SYCL JIT via Intel oneAPI DPC++ 2025.3.3. AOT `bmg_g31` builds
  work under oneAPI 2026.1.1, but that runtime's UR level-zero adapter does not
  enumerate devices on any publicly available GPU userspace today; JIT is the
  verified-working path and reaches decode parity with the lab's promoted
  record (51 tok/s vs 49.7).

## Options (docker-compose `environment`)

| Variable      | Default | Meaning                                          |
|---------------|---------|--------------------------------------------------|
| `GPU_COUNT`   | 2       | `1` or `2` Arc Pro B70s (1 GPU: higher prefill)  |
| `CTX_SIZE_OVERRIDE` | — | Blank: 262144 (2 GPU) / 131072 (1 GPU)        |
| `PARALLEL`    | 1       | Concurrent request slots                         |
| `ENABLE_MTP`  | 0       | `1` = speculative decoding (mtp-Q4_0 draft)      |
| `ENABLE_VISION` | 0     | `1` = image input (mmproj Q8_0, CPU encoder)     |
| `BATCH`/`UBATCH` | 8192 | Prefill batch; 8192 maximizes prefill throughput |

## Notes

- `/dev/dri/by-path` must be mounted (done in compose): the Intel level-zero
  driver enumerates GPUs through it and `--device /dev/dri` alone leaves the
  GPUs invisible.
- The Q4K reorder-family fused kernels are disabled by default: they assume the
  lab AOT build's reordered tensor layout and corrupt output otherwise.
- Tested on: Arch kernel 7.1.8 (xe driver), 2x Arc Pro B70 (Battlemage G31),
  oneAPI 2025.3.3 container image.

## sglang XPU — not viable (2026-08)

sglang 0.5.13 XPU (`llm-scaler-sgl:bmg`) was tested as an alternative path
for Qwen3.8-27B on 2× B70. It loads and prefills, but the GDN decode kernel
produces NaN after ~4 decode tokens → all-`!!!` output at 0.44 tok/s.

Root cause: **SSM state drift** in
`fused_sigmoid_gating_delta_rule_update_kernel`. The fp32 state overflows to
NaN as aggressive decay values (`g` reaching -33 → `exp(g) ≈ 5e-15`) interact
with growing `h` across decode steps, producing `inf - inf = NaN` in the delta
rule update `v -= sum(h * k)`. Related to sglang issue
[#35150](https://github.com/sgl-project/sglang/issues/35150). The ratio-3
(48 V : 16 K heads) split is handled by the fallback path, but the numerical
instability is independent of the ratio.

This repo (llama.cpp SYCL) is unaffected: `GGML_OP_GATED_DELTA_NET` enforces
fp32 at cache, op, and kernel levels, making the `!!!` state drift
structurally impossible.

Full investigation in [`docs/sglang-xpu-qwen38-analysis.md`](docs/sglang-xpu-qwen38-analysis.md).
