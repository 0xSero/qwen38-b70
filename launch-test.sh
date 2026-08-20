#!/usr/bin/env bash
# launch-test.sh — Launch vLLM/XPU with configurable MTP and graph settings
# Usage: launch-test.sh [mtp|nomtp] [graph|eager] [port]
set -euo pipefail

MODE="${1:-nomtp}"
GRAPH="${2:-graph}"
PORT="${3:-8013}"

IMAGE="vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f"
CONTAINER="vllm-test-${MODE}-${GRAPH}"
MAMBA_FILE="/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/mamba_utils.py"

docker rm -f "$CONTAINER" 2>/dev/null || true
sleep 2

# Build args
ARGS="--dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.90 --tensor-parallel-size 1 --max-num-seqs 1 --max-num-batched-tokens 8192 --trust-remote-code --port 8000 --host 0.0.0.0"

if [ "$MODE" = "mtp" ]; then
  ARGS="$ARGS --spec-method qwen3_5_mtp --spec-tokens 5"
fi

if [ "$GRAPH" = "eager" ]; then
  ARGS="$ARGS --enforce-eager"
fi

# Env vars
ENV_FLAGS="-e VLLM_TARGET_DEVICE=xpu -e UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e ZES_ENABLE_SYSMAN=1"
if [ "$GRAPH" = "graph" ]; then
  ENV_FLAGS="$ENV_FLAGS -e VLLM_XPU_ENABLE_XPU_GRAPH=1"
fi

if [ "$MODE" = "mtp" ]; then
  # MTP needs the pointer patch + bash entrypoint
  docker run -d \
    --name "$CONTAINER" \
    --privileged \
    --device /dev/dri:/dev/dri \
    $ENV_FLAGS \
    -v /home/sero/models/Qwen3.8-27B-int4-AutoRound:/model:ro \
    -p "$PORT":8000 \
    --entrypoint /bin/bash \
    "$IMAGE" \
    -c "
      set -e
      if ! grep -q 'import ctypes' '$MAMBA_FILE'; then
        sed -i '/^import torch\$/a import ctypes' '$MAMBA_FILE'
      fi
      if ! grep -q 'ctypes.c_int64' '$MAMBA_FILE'; then
        sed -i 's|self.state_base_addrs\[idx\] = state.data_ptr()|self.state_base_addrs[idx] = ctypes.c_int64(state.data_ptr()).value|' '$MAMBA_FILE'
        sed -i 's|base_addr = state.data_ptr()|base_addr = ctypes.c_int64(state.data_ptr()).value|' '$MAMBA_FILE'
        sed -i 's|self.block_table_ptrs\[i\] = bt.data_ptr()|self.block_table_ptrs[i] = ctypes.c_int64(bt.data_ptr()).value|' '$MAMBA_FILE'
      fi
      exec vllm serve /model $ARGS
    "
else
  # Non-MTP: use image entrypoint directly
  docker run -d \
    --name "$CONTAINER" \
    --privileged \
    --device /dev/dri:/dev/dri \
    $ENV_FLAGS \
    -v /home/sero/models/Qwen3.8-27B-int4-AutoRound:/model:ro \
    -p "$PORT":8000 \
    "$IMAGE" \
    $ARGS
fi

echo "Launched $CONTAINER on port $PORT (mode=$MODE graph=$GRAPH)"
