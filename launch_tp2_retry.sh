#!/usr/bin/env bash
# Retry launching TP2 until profile_run succeeds (XCCL is intermittent)
set -u
MAX_RETRIES=5
for attempt in $(seq 1 $MAX_RETRIES); do
  echo "=== Attempt $attempt/$MAX_RETRIES ==="
  docker rm -f vllm-tp2-bmtp 2>/dev/null
  sleep 3
  docker run -d --name vllm-tp2-bmtp --privileged --device /dev/dri:/dev/dri \
    -e VLLM_TARGET_DEVICE=xpu -e UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e ZES_ENABLE_SYSMAN=1 \
    -e CCL_ZE_IPC_EXCHANGE=sockets -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 \
    -v /home/sero/models/Qwen3.8-27B-int4-AutoRound:/model:ro \
    -v /tmp/lab_tp2_batch_mtp.sh:/entrypoint.sh:ro \
    -p 8020:8000 --entrypoint /bin/bash \
    intel/llm-scaler-vllm:0.21.0-b3 /entrypoint.sh

  # Wait up to 120s for health
  healthy=0
  for i in $(seq 1 40); do
    sleep 3
    if curl -sf http://localhost:8020/health >/dev/null 2>&1; then
      healthy=1
      break
    fi
    # Check for fatal errors
    if docker logs vllm-tp2-bmtp 2>&1 | grep -q "DEVICE_LOST\|Worker failed\|Worker proc.*died"; then
      echo "Fatal error detected, retrying..."
      break
    fi
  done

  if [ "$healthy" = "1" ]; then
    echo "Server healthy on attempt $attempt!"
    exit 0
  fi
  echo "Attempt $attempt failed, cleaning up..."
done

echo "All $MAX_RETRIES attempts failed."
exit 1
