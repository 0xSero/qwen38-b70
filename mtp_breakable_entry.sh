#!/bin/bash
set -e
MAMBA_FILE=/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/mamba_utils.py
VLLM_FILE=/opt/venv/lib/python3.12/site-packages/vllm/config/vllm.py

# Patch mamba_utils.py pointer overflow
grep -q "import ctypes" "$MAMBA_FILE" || sed -i '/^import torch$/a import ctypes' "$MAMBA_FILE"
grep -q "ctypes.c_int64" "$MAMBA_FILE" || {
  sed -i 's|self.state_base_addrs\[idx\] = state.data_ptr()|self.state_base_addrs[idx] = ctypes.c_int64(state.data_ptr()).value|' "$MAMBA_FILE"
  sed -i 's|base_addr = state.data_ptr()|base_addr = ctypes.c_int64(state.data_ptr()).value|' "$MAMBA_FILE"
  sed -i 's|self.block_table_ptrs\[i\] = bt.data_ptr()|self.block_table_ptrs[i] = ctypes.c_int64(bt.data_ptr()).value|' "$MAMBA_FILE"
}

# Patch vllm.py: don't disable CompilationMode when breakable is enabled
if ! grep -q "PATCHED_BREAKABLE_COMPILE" "$VLLM_FILE"; then
  sed -i 's|        if breakable_cudagraph_enabled:$|        if breakable_cudagraph_enabled:  # PATCHED_BREAKABLE_COMPILE|' "$VLLM_FILE"
  sed -i 's|            self.compilation_config.mode = CompilationMode.NONE|            pass  # PATCHED: keep VLLM_COMPILE with breakable cg|' "$VLLM_FILE"
fi

grep -n "PATCHED" "$VLLM_FILE"

exec vllm serve /model \
  --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 1 --max-num-seqs 1 --max-num-batched-tokens 8192 \
  --trust-remote-code --port 8000 --host 0.0.0.0 \
  --spec-method qwen3_5_mtp --spec-tokens 5
