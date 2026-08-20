#!/bin/bash
# launch-vllm-dual.sh — Launch 2 independent vLLM MTP5 instances on 2× B70
#
# Each GPU runs its own vLLM instance with PIECEWISE cudagraph + MTP5.
# ZE_AFFINITY_MASK pins each container to one GPU.
# Concurrent requests to both ports give ~180 tok/s aggregate.
#
# Usage: ./launch-vllm-dual.sh
# Benchmark: ./bench-vllm-dual.sh
set -euo pipefail

IMAGE="intel/llm-scaler-vllm:0.21.0-b3"
MODEL_PATH="/home/sero/models/Qwen3.8-27B-int4-AutoRound"

# Inner script: patches mamba_utils.py for XPU pointer overflow, then launches vllm
INNER_SCRIPT='#!/bin/bash
set -e
MAMBA_FILE="/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/mamba_utils.py"
if ! grep -q "import ctypes" "$MAMBA_FILE"; then
  sed -i "/^import torch$/a import ctypes" "$MAMBA_FILE"
fi
if ! grep -q "ctypes.c_int64" "$MAMBA_FILE"; then
  sed -i "s|self.state_base_addrs\[idx\] = state.data_ptr()|self.state_base_addrs[idx] = ctypes.c_int64(state.data_ptr()).value|" "$MAMBA_FILE"
  sed -i "s|base_addr = state.data_ptr()|base_addr = ctypes.c_int64(base_addr).value|" "$MAMBA_FILE"
  sed -i "s|self.block_table_ptrs\[i\] = bt.data_ptr()|self.block_table_ptrs[i] = ctypes.c_int64(bt.data_ptr()).value|" "$MAMBA_FILE"
fi
exec vllm serve /model \
  --dtype float16 --max-model-len 32768 --gpu-memory-utilization 0.88 \
  --tensor-parallel-size 1 --max-num-seqs 1 --max-num-batched-tokens 8192 \
  --trust-remote-code --port 8000 --host 0.0.0.0 \
  --speculative-config "{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":5,\"model\":\"/model\"}"
'

# Write inner script to temp file on host
echo "$INNER_SCRIPT" > /tmp/vllm_mtp_inner.sh
chmod +x /tmp/vllm_mtp_inner.sh

# Launch GPU0 instance
docker rm -f vllm-dual-gpu0 2>/dev/null || true
docker run -d \
  --name vllm-dual-gpu0 \
  --privileged \
  --device /dev/dri:/dev/dri \
  -e VLLM_TARGET_DEVICE=xpu \
  -e UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e ZES_ENABLE_SYSMAN=1 \
  -e ZE_AFFINITY_MASK=0 \
  -v "${MODEL_PATH}:/model:ro" \
  -v /tmp/vllm_mtp_inner.sh:/entrypoint.sh:ro \
  -p 8020:8000 \
  --entrypoint /bin/bash \
  "$IMAGE" \
  /entrypoint.sh

echo "Launched GPU0 instance on port 8020 (ZE_AFFINITY_MASK=0)"

sleep 2

# Launch GPU1 instance
docker rm -f vllm-dual-gpu1 2>/dev/null || true
docker run -d \
  --name vllm-dual-gpu1 \
  --privileged \
  --device /dev/dri:/dev/dri \
  -e VLLM_TARGET_DEVICE=xpu \
  -e UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e ZES_ENABLE_SYSMAN=1 \
  -e ZE_AFFINITY_MASK=1 \
  -v "${MODEL_PATH}:/model:ro" \
  -v /tmp/vllm_mtp_inner.sh:/entrypoint.sh:ro \
  -p 8021:8000 \
  --entrypoint /bin/bash \
  "$IMAGE" \
  /entrypoint.sh

echo "Launched GPU1 instance on port 8021 (ZE_AFFINITY_MASK=1)"
echo ""
echo "Wait ~4 minutes for both instances to load."
echo "Check: docker logs --tail 5 vllm-dual-gpu0 && docker logs --tail 5 vllm-dual-gpu1"
echo "Benchmark: ./bench-vllm-dual.sh"
