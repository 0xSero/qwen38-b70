# Default Configuration — Qwen3.8-27B on 2× Intel Arc Pro B70

This is the recommended default config, verified coherent and benchmarked.
Two profiles depending on workload:

## Best Single-Stream (MTP2/seqs=8)

Use this for interactive workloads where one request at a time matters most.

```
--dtype float16 --max-model-len 4096 --gpu-memory-utilization 0.80
--tensor-parallel-size 2 --max-num-seqs 8 --max-num-batched-tokens 4096
--trust-remote-code --skip-mm-profiling --distributed-timeout-seconds 600
--speculative-config model:/model
VLLM_XPU_ENABLE_XPU_GRAPH=1
```

**Results**: 85.3 tok/s hard, 81.1 tok/s easy, 306.0 tok/s concurrent×8 aggregate.

## Best Concurrent (MTP3/seqs=8)

Use this for serving multiple users simultaneously.

```
--dtype float16 --max-model-len 4096 --gpu-memory-utilization 0.80
--tensor-parallel-size 2 --max-num-seqs 8 --max-num-batched-tokens 4096
--trust-remote-code --skip-mm-profiling --distributed-timeout-seconds 600
--speculative-config model:/model
VLLM_XPU_ENABLE_XPU_GRAPH=1
```

**Results**: 76.4 tok/s hard, 74.0 tok/s easy, 383.3 tok/s concurrent×8 aggregate.

## Entrypoint scripts

- `entrypoints/ep_mtp2_seqs8.sh` — best single-stream
- `entrypoints/ep_mtp3_seqs8.sh` — best concurrent

## Required patches (applied by entrypoint scripts)

1. **mamba_utils.py** — ctypes int64 pointer fix for MTP state addresses
2. **gdn_linear_attn.py** — try/except for ESIMD eligibility on quantized RowParallelLinear
3. **utils.py** — non_blocking=False for CPU→GPU copy (level_zero staging issue)
4. **xpu_communicator.py** — CPU gloo fallback for XCCL all_reduce OUT_OF_RESOURCES
5. **vllm_gloo_kernels.py** — inductor codegen patches for TP2 PIECEWISE graph mode
6. **comm_lowering.py** — split coalesced all_gather/all_reduce into individual calls (XCCL over PCIe)
7. **distributed_c10d.py** — disable device coalescing during profile_run
8. **gpu_worker.py** — set VLLM_XPU_PROFILE_CPU_GLOO around profile_run

## Docker image

```
intel/llm-scaler-vllm:0.21.0-b3
```

## Hardware

- 2× Intel Arc Pro B70 (BMG-G31), 32 GB GDDR6 each, 608 GB/s, 256 EUs
- Connected via PCIe (not XeLinks)
- Host: omarchy

## Why MTP2-3 instead of MTP5

MTP5 acceptance metrics showed positions 4-5 had near-zero acceptance (~0.03 and ~0.00),
meaning 40% of draft compute was wasted. Reducing to MTP2-3 keeps mean acceptance
length at 2.6-2.8 (vs 2.9 for MTP5) while cutting draft forward passes by 40-60%.

## Why seqs=8

The B70 has 256 EUs. With seqs=2, decode batches are too small to saturate them.
seqs=8 lets the scheduler batch up to 8 concurrent decode sequences into one
forward pass, dramatically improving GPU utilization. seqs=12+ OOMs at mem=0.80.
