#!/usr/bin/env bash
# launch-mtp1-graph.sh — MTP1 + XPU Graph
# With spec-tokens=1, decode processes 2 tokens (1 confirmed + 1 draft),
# which should fit within cudagraph capture size 2.
set -euo pipefail

IMAGE="vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f"
CONTAINER="vllm-mtp1-graph"
PORT=8015
SPEC="${2:-1}"  # default 1, can override
MAMBA_FILE="/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/mamba_utils.py"

docker rm -f "$CONTAINER" 2>/dev/null || true
sleep 2

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
    MAMBA_FILE="/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/mamba_utils.py"
    if ! grep -q "import ctypes" "$MAMBA_FILE"; then
      sed -i "/^import torch$/a import ctypes" "$MAMBA_FILE"
    fi
    if ! grep -q "ctypes.c_int64" "$MAMBA_FILE"; then
      sed -i "s|self.state_base_addrs\[idx\] = state.data_ptr()|self.state_base_addrs[idx] = ctypes.c_int64(state.data_ptr()).value|" "$MAMBA_FILE"
      sed -i "s|base_addr = state.data_ptr()|base_addr = ctypes.c_int64(state.data_ptr()).value|" "$MAMBA_FILE"
      sed -i "s|self.block_table_ptrs\[i\] = bt.data_ptr()|self.block_table_ptrs[i] = ctypes.c_int64(bt.data_ptr()).value|" "$MAMBA_FILE"
    fi
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
      --spec-tokens SPEC_TOKENS_PLACEHOLDER
  '

# Replace placeholder with actual spec tokens
docker stop "$CONTAINER" 2>/dev/null || true
docker commit "$CONTAINER" "${CONTAINER}-img" 2>/dev/null || true

# Actually, just re-run with correct spec tokens
docker rm -f "$CONTAINER" 2>/dev/null || true
sleep 1

SPEC_TOKENS="${1:-1}"

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
  -c "
    set -e
    MAMBA_FILE=/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/mamba_utils.py
    grep -q 'import ctypes' \"\$MAMBA_FILE\" || sed -i '/^import torch\$/a import ctypes' \"\$MAMBA_FILE\"
    grep -q 'ctypes.c_int64' \"\$MAMBA_FILE\" || {
      sed -i 's|self.state_base_addrs\[idx\] = state.data_ptr()|self.state_base_addrs[idx] = ctypes.c_int64(state.data_ptr()).value|' \"\$MAMBA_FILE\"
      sed -i 's|base_addr = state.data_ptr()|base_addr = ctypes.c_int64(state.data_ptr()).value|' \"\$MAMBA_FILE\"
      sed -i 's|self.block_table_ptrs\[i\] = bt.data_ptr()|self.block_table_ptrs[i] = ctypes.c_int64(bt.data_ptr()).value|' \"\$MAMBA_FILE\"
    }
    exec vllm serve /model \
      --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.90 \
      --tensor-parallel-size 1 --max-num-seqs 1 --max-num-batched-tokens 8192 \
      --trust-remote-code --port 8000 --host 0.0.0.0 \
      --spec-method qwen3_5_mtp --spec-tokens $SPEC_TOKENS
  "

echo "Launched MTP$SPEC_TOKENS + XPU Graph on port $PORT"
