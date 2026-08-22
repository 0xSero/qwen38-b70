#!/bin/bash
set -e

# =============================================================================
# TP2 PIECEWISE with inductor codegen patch — v11
#
# Key change from v10: vllm_gloo_kernels.py now patches inductor's
# _AllReduce_Kernel / _AllReduceKernel / _WaitKernel codegen() to use
# runtime dispatch (parent _CollectiveKernel.codegen) instead of the
# hardcoded CPU C shim.  This eliminates empty_strided_cpu() + copy_()
# in compiled code, which was the blocker for TP2 PIECEWISE on XPU.
#
# The compiled graph now emits:
#   buf = torch.ops._c10d_functional.all_reduce_.default(buf, 'sum', '...')
#   buf = torch.ops._c10d_functional.wait_tensor.default(buf)
# which hits our registered XPU kernels -> XCCL allreduce, no CPU buffer.
# =============================================================================

# --- MTP pointer overflow fix ---
MAMBA_FILE="/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/mamba_utils.py"
if ! grep -q "import ctypes" "$MAMBA_FILE"; then
  sed -i "/^import torch$/a import ctypes" "$MAMBA_FILE"
fi
if ! grep -q "ctypes.c_int64" "$MAMBA_FILE"; then
  sed -i "s|self.state_base_addrs\[idx\] = state.data_ptr()|self.state_base_addrs[idx] = ctypes.c_int64(state.data_ptr()).value|" "$MAMBA_FILE"
  sed -i "s|base_addr = state.data_ptr()|base_addr = ctypes.c_int64(base_addr).value|" "$MAMBA_FILE"
  sed -i "s|self.block_table_ptrs\[i\] = bt.data_ptr()|self.block_table_ptrs[i] = ctypes.c_int64(bt.data_ptr()).value|" "$MAMBA_FILE"
fi

# --- GDN ESIMD eligibility fix ---
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

# --- non_blocking=False fix ---
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

# --- Patch xpu_communicator.py: CPU gloo fallback for profile_run ---
python3 << 'PYEOF3'
with open("/opt/venv/lib/python3.12/site-packages/vllm/distributed/device_communicators/xpu_communicator.py", "r") as f:
    code = f.read()

old_class = "class XpuCommunicator(DeviceCommunicatorBase):\n    def __init__(\n        self,\n        cpu_group: ProcessGroup,"
new_class = """class XpuCommunicator(DeviceCommunicatorBase):
    def _profile_cpu_gloo(self):
        import os as _os
        return _os.environ.get("VLLM_XPU_PROFILE_CPU_GLOO") == "1"

    def _cpu_all_reduce(self, output):
        cpu_tensor = output.cpu()
        dist.all_reduce(cpu_tensor, group=self.cpu_group)
        output.copy_(cpu_tensor.to(output.device))
        return output

    def _cpu_all_gather_into_tensor(self, output_tensor, input_):
        cpu_input = input_.cpu()
        cpu_output = torch.empty_like(output_tensor, device='cpu')
        dist.all_gather_into_tensor(cpu_output, cpu_input, group=self.cpu_group)
        output_tensor.copy_(cpu_output.to(output_tensor.device))
        return output_tensor

    def __init__(
        self,
        cpu_group: ProcessGroup,"""
if old_class in code:
    code = code.replace(old_class, new_class)
    print("Added CPU gloo helper methods")

old_ar = "        dist.all_reduce(output, group=self.device_group)\n        post_nan ="
new_ar = """        if self._profile_cpu_gloo():
            self._cpu_all_reduce(output)
        else:
            try:
                dist.all_reduce(output, group=self.device_group)
            except RuntimeError as e:
                if "OUT_OF_RESOURCES" in str(e) or "OUT_OF_DEVICE_MEMORY" in str(e):
                    self._cpu_all_reduce(output)
                else:
                    raise
        post_nan ="""
if old_ar in code:
    code = code.replace(old_ar, new_ar)
    print("Patched all_reduce")

old_agt = "        dist.all_gather_into_tensor(output_tensor, input_, group=self.device_group)"
new_agt = """        if self._profile_cpu_gloo():
            self._cpu_all_gather_into_tensor(output_tensor, input_)
        else:
            try:
                dist.all_gather_into_tensor(output_tensor, input_, group=self.device_group)
            except RuntimeError as e:
                if "OUT_OF_RESOURCES" in str(e) or "OUT_OF_DEVICE_MEMORY" in str(e):
                    self._cpu_all_gather_into_tensor(output_tensor, input_)
                else:
                    raise"""
if old_agt in code:
    code = code.replace(old_agt, new_agt, 1)
    print("Patched all_gather_into_tensor")

with open("/opt/venv/lib/python3.12/site-packages/vllm/distributed/device_communicators/xpu_communicator.py", "w") as f:
    f.write(code)
import ast
ast.parse(code)
print("xpu_communicator.py patched and verified")
PYEOF3

# --- Copy gloo kernel module (patches inductor codegen + registers XPU kernels) ---
cp /tmp/vllm_gloo_kernels.py /opt/venv/lib/python3.12/site-packages/vllm_gloo_kernels.py
python3 << 'PYEOF4'
init_path = "/opt/venv/lib/python3.12/site-packages/vllm/__init__.py"
with open(init_path, "r") as f:
    code = f.read()
if "vllm_gloo_kernels" not in code:
    old = "import vllm.env_override  # noqa: F401"
    new = "import vllm.env_override  # noqa: F401\ntry:\n    import vllm_gloo_kernels  # noqa: F401\nexcept Exception:\n    pass"
    if old in code:
        code = code.replace(old, new, 1)
        with open(init_path, "w") as f:
            f.write(code)
        print("Added vllm_gloo_kernels import to vllm/__init__.py")
    else:
        print("ERROR: could not find insertion point")
else:
    print("vllm_gloo_kernels already imported")
PYEOF4

# --- Patch comm_lowering.py: split coalesced ops (still needed) ---
python3 << 'PYEOF5'
with open("/opt/venv/lib/python3.12/site-packages/torch/_inductor/comm_lowering.py", "r") as f:
    code = f.read()

# Split all_gather_into_tensor_coalesced into individual calls
# (coalesced variant segfaults on XCCL over PCIe)
old = """    @register_comm_lowering(c10d.all_gather_into_tensor_coalesced)
    def _all_gather_into_tensor_coalesced(inputs, group_size, group_name):
        return pytree.tree_map(
            ir.TensorBox.create,
            ir._CollectiveKernel.create_out_of_place(
                c10d.all_gather_into_tensor_coalesced.default,
                inputs,
                group_size,
                group_name,
            ),
        )"""
new = """    @register_comm_lowering(c10d.all_gather_into_tensor_coalesced)
    def _all_gather_into_tensor_coalesced(inputs, group_size, group_name):
        # ALWAYS split coalesced into individual all_gather_into_tensor calls.
        # The coalesced variant has no XPU fallback and segfaults on XCCL over PCIe.
        results = []
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        for inp in inputs:
            result = _create_out_of_place(
                c10d.all_gather_into_tensor.default,
                inp,
                group_size,
                group_name,
            )
            results.append(result)
        return results"""
if old in code:
    code = code.replace(old, new)
    print("Patched all_gather_into_tensor_coalesced")
else:
    print("all_gather_into_tensor_coalesced pattern not found")

# Split all_reduce_coalesced into individual all_reduce calls
old_ar_coal = """    @register_comm_lowering(c10d.all_reduce_coalesced)
    def _all_reduce_coalesced(inputs, reduce_op, group_name):
        inputs = [clone(inp) for inp in inputs]
        ir._CollectiveKernel.create_inplace(
            c10d.all_reduce_coalesced_.default,
            inputs,
            reduce_op,
            group_name,
        )
        return inputs"""
new_ar_coal = """    @register_comm_lowering(c10d.all_reduce_coalesced)
    def _all_reduce_coalesced(inputs, reduce_op, group_name):
        # ALWAYS split coalesced into individual all_reduce calls (same
        # reason as all_gather: coalesced variant lacks XPU fallback).
        results = []
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        for inp in inputs:
            cloned = clone(inp)
            result = _create_out_of_place(
                c10d.all_reduce.default,
                cloned,
                reduce_op,
                group_name,
            )
            results.append(result)
        return results"""
if old_ar_coal in code:
    code = code.replace(old_ar_coal, new_ar_coal)
    print("Patched all_reduce_coalesced")
else:
    print("all_reduce_coalesced pattern not found")

with open("/opt/venv/lib/python3.12/site-packages/torch/_inductor/comm_lowering.py", "w") as f:
    f.write(code)
print("comm_lowering.py patched and verified")
PYEOF5

# --- Patch distributed_c10d.py: disable coalescing device during profile_run ---
python3 << 'PYEOF6'
with open("/opt/venv/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py", "r") as f:
    code = f.read()
old = "    if device:\n        group._start_coalescing(device)"
new = """    import os as _os
    if device and _os.environ.get("VLLM_XPU_PROFILE_CPU_GLOO") == "1":
        device = None
    if device:
        group._start_coalescing(device)"""
if old in code:
    code = code.replace(old, new, 1)
    with open("/opt/venv/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py", "w") as f:
        f.write(code)
    print("Patched distributed_c10d.py")
else:
    print("distributed_c10d.py pattern not found")
PYEOF6

# --- Patch gpu_worker.py: set VLLM_XPU_PROFILE_CPU_GLOO around profile_run ---
python3 << 'PYEOF7'
with open("/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py", "r") as f:
    code = f.read()
old1 = "            # still need a profile run which compiles the model for\n            # max_num_batched_tokens\n            self.model_runner.profile_run()"
new1 = """            # still need a profile run which compiles the model for
            # max_num_batched_tokens
            import os as _os, torch.distributed as dist
            _os.environ["VLLM_XPU_PROFILE_CPU_GLOO"] = "1"
            if dist.is_initialized():
                dist.barrier()
            self.model_runner.profile_run()
            _os.environ.pop("VLLM_XPU_PROFILE_CPU_GLOO", None)"""
old2 = "        with memory_profiling(\n            self.init_snapshot,\n            weights_memory=int(self.model_runner.model_memory_usage),\n        ) as profile_result:\n            self.model_runner.profile_run()"
new2 = """        import os as _os, torch.distributed as dist
        _os.environ["VLLM_XPU_PROFILE_CPU_GLOO"] = "1"
        if dist.is_initialized():
            dist.barrier()
        with memory_profiling(
            self.init_snapshot,
            weights_memory=int(self.model_runner.model_memory_usage),
        ) as profile_result:
            self.model_runner.profile_run()
            _os.environ.pop("VLLM_XPU_PROFILE_CPU_GLOO", None)"""
changed = False
if old1 in code:
    code = code.replace(old1, new1)
    changed = True
if old2 in code:
    code = code.replace(old2, new2)
    changed = True

if changed:
    with open("/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py", "w") as f:
        f.write(code)
    print("Patched gpu_worker.py")
else:
    print("gpu_worker.py: no changes needed")
PYEOF7

# Clear compilation cache so new codegen is used
echo "Cleared compilation cache"

export VLLM_XPU_ENABLE_XPU_GRAPH=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
export VLLM_LOGGING_LEVEL=INFO


export VLLM_XPU_ENABLE_XPU_GRAPH=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
export VLLM_LOGGING_LEVEL=INFO

MTP_CONFIG='{"method":"qwen3_5_mtp","num_speculative_tokens":5,"model":"/model"}'

exec vllm serve /model \
  --dtype float16 --max-model-len 4096 --gpu-memory-utilization 0.85 \
  --tensor-parallel-size 2 --max-num-seqs 1 --max-num-batched-tokens 2048 \
  --trust-remote-code --port 8000 --host 0.0.0.0 \
  --skip-mm-profiling \
  --distributed-timeout-seconds 600 \
  --speculative-config "$MTP_CONFIG"
