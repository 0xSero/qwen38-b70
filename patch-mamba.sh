#!/bin/bash
# patch-mamba.sh — Fix ALL XPU pointer overflow in mamba_utils.py, then launch vLLM
set -e
cd /opt/venv/lib/python3.12/site-packages/vllm/v1/worker
cp -n mamba_utils.py mamba_utils.py.bak

python3 << 'PYEOF'
import re
path = "mamba_utils.py"
src = open(path).read()
lines = src.split("\n")
patched = 0
new_lines = []
for i, line in enumerate(lines):
    # Patch any line that assigns data_ptr() to a tensor or numpy array
    if "data_ptr()" in line and "= state.data_ptr()" in line:
        line = line.replace("state.data_ptr()", "state.data_ptr() & ((1 << 63) - 1)")
        patched += 1
        print(f"Patched line {i+1}: {line.strip()}")
    elif "data_ptr()" in line and "= state[dest_block_id].data_ptr()" in line:
        line = line.replace("state[dest_block_id].data_ptr()", "state[dest_block_id].data_ptr() & ((1 << 63) - 1)")
        patched += 1
        print(f"Patched line {i+1}: {line.strip()}")
    elif "data_ptr()" in line and "bt.data_ptr()" in line and "= bt.data_ptr()" in line:
        line = line.replace("bt.data_ptr()", "bt.data_ptr() & ((1 << 63) - 1)")
        patched += 1
        print(f"Patched line {i+1}: {line.strip()}")
    elif "data_ptr()" in line and "= state.data_ptr()" not in line:
        # Check if it's an assignment
        if "base_addr = state.data_ptr()" in line:
            line = line.replace("state.data_ptr()", "state.data_ptr() & ((1 << 63) - 1)")
            patched += 1
            print(f"Patched line {i+1}: {line.strip()}")
        else:
            print(f"UNPATCHED line {i+1}: {line.strip()}")
    new_lines.append(line)

open(path, "w").write("\n".join(new_lines))
print(f"\nTotal patched: {patched}")
PYEOF

echo "=== Launching vLLM ==="
exec vllm serve /model \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 1 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8192 \
  --trust-remote-code \
  --spec-method qwen3_5_mtp \
  --spec-tokens 5 \
  --port 8000 \
  --host 0.0.0.0
