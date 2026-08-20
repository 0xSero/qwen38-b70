#!/usr/bin/env bash
# launch-mtp5.sh — Launch vLLM/XPU with AutoRound INT4 + MTP5 + patched mamba_utils
#
# This script:
# 1. Stops any existing vLLM MTP container
# 2. Applies the ctypes pointer-overflow patch to mamba_utils.py
# 3. Launches vLLM serve with MTP5 speculative decoding
#
# The patch fixes XPU pointer overflow (0xFFFF... range > int64 max) by
# converting data_ptr() through ctypes.c_int64() — preserving the bit pattern
# as two's complement signed int64, which Triton correctly reinterprets.
set -euo pipefail

IMAGE="vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f"
CONTAINER="vllm-mtp5-1gpu"
PORT=8012
MODEL="/model"
MAMBA_FILE="/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/mamba_utils.py"

echo "=== Stopping any existing MTP container ==="
docker rm -f "$CONTAINER" 2>/dev/null || true

echo "=== Launching container (will patch inside) ==="
docker run -d \
  --name "$CONTAINER" \
  --privileged \
  --device /dev/dri:/dev/dri \
  -e VLLM_TARGET_DEVICE=xpu \
  -e UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e ZES_ENABLE_SYSMAN=1 \
  -v /home/sero/models/Qwen3.8-27B-int4-AutoRound:/model:ro \
  -p "$PORT":8000 \
  --entrypoint /bin/bash \
  "$IMAGE" \
  -c '
    set -e
    echo "=== Applying pointer-overflow patch to mamba_utils.py ==="

    # Add ctypes import if not present
    if ! grep -q "import ctypes" "'"$MAMBA_FILE"'"; then
      sed -i "/^import torch$/a import ctypes" "'"$MAMBA_FILE"'"
      echo "Added ctypes import"
    fi

    # Patch the 3 overflow sites (only if not already patched)
    if ! grep -q "ctypes.c_int64" "'"$MAMBA_FILE"'"; then
      sed -i "s|self.state_base_addrs\[idx\] = state.data_ptr()|self.state_base_addrs[idx] = ctypes.c_int64(state.data_ptr()).value|" "'"$MAMBA_FILE"'"
      sed -i "s|base_addr = state.data_ptr()|base_addr = ctypes.c_int64(state.data_ptr()).value|" "'"$MAMBA_FILE"'"
      sed -i "s|self.block_table_ptrs\[i\] = bt.data_ptr()|self.block_table_ptrs[i] = ctypes.c_int64(bt.data_ptr()).value|" "'"$MAMBA_FILE"'"
      echo "Patched 3 data_ptr() assignments"
    else
      echo "Already patched"
    fi

    # Verify
    grep -n "ctypes.c_int64" "'"$MAMBA_FILE"'"

    echo "=== Launching vLLM with MTP5 ==="
    exec vllm serve /model \
      --dtype bfloat16 \
      --max-model-len 32768 \
      --gpu-memory-utilization 0.90 \
      --tensor-parallel-size 1 \
      --max-num-seqs 1 \
      --max-num-batched-tokens 8192 \
      --trust-remote-code \
      --port 8000 \
      --host 0.0.0.0 \
      --spec-method qwen3_5_mtp \
      --spec-tokens 5
  '

echo "=== Container launched on port $PORT ==="
echo "Watching logs for startup..."
# Wait and show initial logs
sleep 10
docker logs "$CONTAINER" 2>&1 | tail -50
