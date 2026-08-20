# Qwen3.8-27B on sglang XPU (Intel Arc B70) — Analysis

## Verdict

**sglang 0.5.13 XPU cannot serve Qwen3.8-27B.** The model loads and prefills
correctly, but the GDN (Gated Delta Network) decode kernel produces NaN after
~4 decode tokens, causing `!!!` garbage output. The llama.cpp SYCL path
remains the only working B70 backend for this model (51 tok/s decode, 84 tok/s
with MTP).

## What was tested

- **Host**: omarchy, 2× Intel Arc Pro B70 (Battlemage G31, 16 GB each)
- **Image**: `llm-scaler-sgl:bmg` (sglang 0.5.13, oneAPI 2025.x, XPU backend)
- **Model**: `/home/sero/models/Qwen3.8-27B` (HF safetensors, bf16)
- **Config**: TP=2, `--dtype float16`, `--quantization fp8 --load-format layered_fp8`,
  `--attention-backend intel_xpu`, `--mem-fraction-static 0.82`
- **Env**: `SGLANG_MAMBA_CONV_DTYPE=float16`, all ESIMD/fast paths disabled
  (`SGLANG_XPU_GDN_ESIMD=0`, `SGLANG_XPU_GDN_FAST_PATH=0`,
  `SGLANG_XPU_GDN_EXTEND_ESIMD=0`)

## Model architecture (Qwen3.8-27B)

| Parameter | Value |
|-----------|-------|
| `architectures` | `Qwen3_5ForConditionalGeneration` (VLM) |
| `model_type` | `qwen3_5` |
| `text_config.model_type` | `qwen3_5_text` |
| `hidden_size` | 5120 |
| `num_hidden_layers` | 64 (48 GDN + 16 GQA, every 4th) |
| `full_attention_interval` | 4 |
| **GDN (linear attn) heads** | K=16, V=48 → **3:1 V:K ratio** |
| GDN head dims | `linear_key_head_dim=128`, `linear_value_head_dim=128` |
| **Full attn (GQA) heads** | Q=24, KV=4 → 6:1 GQA ratio |
| Full attn `head_dim` | 256 |
| `partial_rotary_factor` | 0.25 (64 rope dims, 192 nope dims) |
| `mamba_ssm_dtype` | float32 |
| `mtp_num_hidden_layers` | 1 |
| Context | 262,144 |
| VLM | Yes (vision encoder present) |

The **3:1 V:K GDN head ratio** is the key difference from Qwen3.6 (which has
2:1). sglang 0.5.13's fused QKV split kernel (`fused_qkvzba_split_reshape_cat_contiguous`)
only supports ratios [1, 2, 4] on non-CUDA backends; PR #34859 adds ratio 3
for CUDA only (`(1, 2, 3, 4) if _is_cuda else (1, 2, 4)`).

## Debugging trail

### 1. Initial attempt: layered_fp8, TP=2

Server loads successfully (14.19 GB/GPU FP8, `Qwen3_5ForConditionalGeneration`).
Output is all `!!!` characters. Prefill throughput: 0.04 tok/s (vs 95 tok/s on
Qwen3.6 — 2000× slower).

### 2. Patch: skip packed_decode for non-power-of-2 ratios

The `fused_qkvzba_split_reshape_cat_contiguous` kernel crashes with
`"arange's range must be a power of 2"` for ratio 3 (Triton's `tl.arange`
requires power-of-2). The fallback path (`fix_query_key_value_ordering`)
produces a sequential `[Q|K|V]` layout.

Patch applied to `gdn_backend.py`: skip `packed_decode` when
`num_v_heads // num_k_heads` is not a power of 2, forcing the non-packed
decode path (split → reshape → `fused_sigmoid_gating_delta_rule_update`).

**Result**: No crashes, but still `!!!`.

### 3. Kernel isolation tests (all PASS)

Tested each kernel in isolation with ratio-3 inputs on XPU:

- `fused_sigmoid_gating_delta_rule_update` (Triton GDN decode kernel):
  ✅ Matches PyTorch reference, max abs diff 1.4e-5
- `causal_conv1d_update` (Triton conv1d): ✅ Correct output
- `chunk_gated_delta_rule_torch` (XPU prefill fallback): ✅ Correct output
- `fused_gdn_gating` (Triton gating kernel): ✅ Correct, grid `(batch, 1, 3)`
  handles 24 heads (3 blocks × 8 = 24)
- Full conv1d → decode pipeline: ✅ Valid output, no NaN

**The kernels are individually correct for ratio 3.**

### 4. Dimension analysis (all correct)

- `RadixLinearAttention.q_dim/k_dim/v_dim` use per-rank head counts
  (8 K heads, 24 V heads per rank at TP=2) → dimensions match
- `MergedColumnParallelLinear` for `in_proj_qkvz` with
  `output_sizes=[2048, 2048, 6144, 6144]` splits correctly per rank
- Checkpoint has separate `in_proj_qkv` [10240] + `in_proj_z` [6144],
  merged via `stacked_params_mapping` with shard IDs (0,1,2) and 3
- `conv_dim = key_dim*2 + value_dim = 10240` (full), splits to 5120 per rank
- `mixed_qkv` per rank = 5120 = q_dim(1024) + k_dim(1024) + v_dim(3072) ✓

### 5. Debug hooks on running server (found the bug)

Patched `gdn_backend.py` and `xpu_backend.py` directly in site-packages
to add debug logging in TP worker processes.

**Prefill (extend)**: All GDN layers produce valid output (no NaN).
Full attention layers also fine.

**Decode**: Layers 0–4 fine for first 3–4 decode steps. On the **4th decode
step**, GDN layer L4's SSM state goes from `[-0.42, 0.53]` (valid) to
`[NaN, NaN]`. The NaN propagates to all subsequent layers.

| Decode step | L4 state-in | L4 state-out | L4 output |
|-------------|-------------|--------------|-----------|
| 1 | [-0.40, 0.40] | [-0.38, 0.34] | valid |
| 2 | [-0.33, 0.54] | [-0.33, 0.53] | valid |
| 3 | [-0.33, 0.53] | [-0.32, 0.53] | valid |
| 4 | [-0.42, 0.34] | **[NaN, NaN]** | **NaN** |

SSM state is `torch.float32` (confirmed via debug hooks). The NaN is produced
inside `fused_sigmoid_gating_delta_rule_update_kernel` — the Triton kernel
computes `exp(g)` where `g` can reach -33 (from prefill logs), then the delta
rule update `v -= sum(h * k, dim=0)` produces inf - inf = NaN when the state
`h` grows large and `exp(g)` underflows to 0 in certain heads.

### 6. TP=1 test

27B model at FP8 = ~27 GB, exceeds single B70's 16 GB. OOM at
`UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` during weight loading. TP=1 is not
viable for this model on B70.

## Root cause

**SSM state drift in the Triton GDN decode kernel** (`fused_sigmoid_gating_delta_rule_update_kernel`)
after ~4 decode steps. This is NOT a ratio-3-specific bug — the kernel
handles ratio 3 correctly in isolation. The issue is numerical instability
that accumulates over decode steps, likely triggered by the combination of:

1. Very aggressive decay values (`g` reaching -33 in prefill, meaning
   `exp(-33) ≈ 4.7e-15`)
2. The delta rule's `v -= sum(h * k)` operation, which can produce
   inf - inf = NaN when `h` has grown large from prior steps and the decay
   underflows
3. fp16 activations (q, k, v are fp16) feeding into fp32 state accumulation

This is the same class of bug as Qwen3.6's `!!!` (which was caused by fp16
SSM state drift in the ESIMD GDN kernel). On Qwen3.8, it occurs even with
fp32 SSM state and the Triton (non-ESIMD) path, because the instability is
in the activation dtypes and the decay dynamics, not the state dtype.

Related sglang issues:
- [#35150](https://github.com/sgl-project/sglang/issues/35150): "Qwen3.8
  DSpark forced-reject is not lossless: accumulated GDN state drift vs Base
  decode" — describes the same state drift class of bug
- [#34859](https://github.com/sgl-project/sglang/pull/34859): Adds ratio 3
  support for CUDA only; XPU still uses `(1, 2, 4)`

## Comparison: sglang XPU vs llama.cpp SYCL

| Metric | sglang 0.5.13 XPU | llama.cpp SYCL (qwen38-b70) |
|--------|---------------------|------------------------------|
| **Status** | ❌ Broken (NaN after ~4 decode tokens) | ✅ Working |
| Decode tok/s (TP2) | N/A (garbage output) | 51.1 |
| Decode tok/s (MTP) | N/A | 84.3 |
| Prefill tok/s | 0.04 (2000× slower than Qwen3.6) | 955 (2.5k context) |
| GDN implementation | Triton kernel (fp16 activations, fp32 state) | GGML_OP_GATED_DELTA_NET (fp32 at 3 enforcement levels) |
| !!! bug | Present (SSM state drift) | Structurally impossible (fp32 throughout) |
| Quantization | FP8 (layered_fp8 on-device) | Q4_K_M GGUF |
| 3:1 ratio support | Fallback path works, but kernel numerically unstable | First-class (op-level fp32) |

## Why llama.cpp works

llama.cpp's GDN implementation enforces fp32 at three levels:
1. **Cache allocation**: GDN state cache is always allocated as fp32
2. **Op assertion**: `GGML_OP_GATED_DELTA_NET` asserts fp32 on all inputs
3. **Kernel**: The SYCL GDN kernel computes entirely in fp32

This makes the `!!!` bug structurally impossible — there is no fp16
activation path that can produce the inf/NaN cascade.

## Conclusion

sglang 0.5.13 XPU is not viable for Qwen3.8-27B on Intel Arc B70 due to
SSM state drift in the GDN decode kernel. The llama.cpp SYCL path
(`qwen38-b70` repo) remains the working solution at 51 tok/s decode
(84 tok/s with MTP speculative decoding).

To make sglang XPU work for Qwen3.8-27B, the following would be needed:
1. Fix the GDN decode kernel's numerical instability (likely needs fp32
   activations or a clamped decay)
2. Add ratio 3 to the XPU fused split kernel path (PR #34859 only adds it
   for CUDA)
3. Port the ESIMD GDN decode kernel to handle ratio 3 (currently only
   validated for ratios 1 and 2)

Until then, use `docker compose up -d` from the `qwen38-b70` repo.
