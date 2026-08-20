#!/bin/bash
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

# Patch gdn_linear_attn.py
python3 << 'PYEOF'
with open("/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/mamba/gdn_linear_attn.py", "r") as f:
    code = f.read()
old = """        if ESIMD_AVAILABLE and not disable_esimd_gdn_outproj():
            w = self.out_proj.weight"""
new = """        if ESIMD_AVAILABLE and not disable_esimd_gdn_outproj():
            try:
                w = self.out_proj.weight
            except AttributeError:
                self._gdn_outproj_is_block = False
                self._gdn_outproj_esimd_ok = False
                return False"""
if old in code:
    code = code.replace(old, new)
    with open("/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/mamba/gdn_linear_attn.py", "w") as f:
        f.write(code)
    print("Patched gdn_linear_attn.py")
else:
    print("gdn_linear_attn.py already patched")
PYEOF

# Patch utils.py: non_blocking=False
python3 << 'PYEOF2'
with open("/opt/venv/lib/python3.12/site-packages/vllm/v1/utils.py", "r") as f:
    code = f.read()
code2 = code.replace("self.gpu.copy_(self.cpu, non_blocking=True)", "self.gpu.copy_(self.cpu, non_blocking=False)")
code2 = code2.replace("self.gpu[:n].copy_(self.cpu[:n], non_blocking=True)", "self.gpu[:n].copy_(self.cpu[:n], non_blocking=False)")
if code2 != code:
    with open("/opt/venv/lib/python3.12/site-packages/vllm/v1/utils.py", "w") as f:
        f.write(code2)
    print("Patched utils.py")
else:
    print("utils.py already patched")
PYEOF2

# Patch xpu_communicator.py: catch OUT_OF_RESOURCES and fall back to CPU gloo all_reduce
python3 << 'PYEOF3'
with open("/opt/venv/lib/python3.12/site-packages/vllm/distributed/device_communicators/xpu_communicator.py", "r") as f:
    code = f.read()

# Replace the main dist.all_reduce call with a try/except that falls back to CPU gloo
old = "        dist.all_reduce(output, group=self.device_group)\n        post_nan ="
new = """        try:
            dist.all_reduce(output, group=self.device_group)
        except RuntimeError as e:
            if "OUT_OF_RESOURCES" in str(e) or "OUT_OF_DEVICE_MEMORY" in str(e):
                logger.warning("XPU all_reduce failed (%s), falling back to CPU gloo", str(e))
                cpu_tensor = output.cpu()
                dist.all_reduce(cpu_tensor, group=self.cpu_group)
                output.copy_(cpu_tensor.to(output.device))
            else:
                raise
        post_nan ="""

if old in code:
    code = code.replace(old, new)
    with open("/opt/venv/lib/python3.12/site-packages/vllm/distributed/device_communicators/xpu_communicator.py", "w") as f:
        f.write(code)
    print("Patched xpu_communicator.py: CPU gloo fallback on OUT_OF_RESOURCES")
else:
    print("xpu_communicator.py pattern not found, trying alternative")
    old2 = "        dist.all_reduce(output, group=self.device_group)"
    new2 = """        try:
            dist.all_reduce(output, group=self.device_group)
        except RuntimeError as e:
            if "OUT_OF_RESOURCES" in str(e) or "OUT_OF_DEVICE_MEMORY" in str(e):
                logger.warning("XPU all_reduce failed (%s), falling back to CPU gloo", str(e))
                cpu_tensor = output.cpu()
                dist.all_reduce(cpu_tensor, group=self.cpu_group)
                output.copy_(cpu_tensor.to(output.device))
            else:
                raise"""
    if old2 in code:
        code = code.replace(old2, new2, 1)
        with open("/opt/venv/lib/python3.12/site-packages/vllm/distributed/device_communicators/xpu_communicator.py", "w") as f:
            f.write(code)
        print("Patched xpu_communicator.py (alternative)")
    else:
        print("FAILED to patch xpu_communicator.py")
PYEOF3

# MTP5 speculative config
MTP_CONFIG='{"method":"qwen3_5_mtp","num_speculative_tokens":5,"model":"/model"}'

exec vllm serve /model \
  --dtype float16 --max-model-len 4096 --gpu-memory-utilization 0.45 \
  --tensor-parallel-size 2 --max-num-seqs 1 --max-num-batched-tokens 2048 \
  --trust-remote-code --port 8000 --host 0.0.0.0 \
  --skip-mm-profiling \
  --enforce-eager \
  --distributed-timeout-seconds 600 \
  --speculative-config "$MTP_CONFIG"
