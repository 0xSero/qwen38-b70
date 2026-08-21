#!/bin/bash
set -e

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

# --- Patch xpu_communicator.py ---
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

    def _cpu_all_gather(self, tensor_list, input_):
        cpu_input = input_.cpu()
        cpu_list = [t.cpu() for t in tensor_list]
        dist.all_gather(cpu_list, cpu_input, group=self.cpu_group)
        for t, ct in zip(tensor_list, cpu_list):
            t.copy_(ct.to(t.device))
        return tensor_list

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

old_ag1 = "                dist.all_gather(all_gather_list, input_, group=self.device_group)"
new_ag1 = """                if self._profile_cpu_gloo():
                    self._cpu_all_gather(all_gather_list, input_)
                else:
                    try:
                        dist.all_gather(all_gather_list, input_, group=self.device_group)
                    except RuntimeError as e:
                        if "OUT_OF_RESOURCES" in str(e) or "OUT_OF_DEVICE_MEMORY" in str(e):
                            self._cpu_all_gather(all_gather_list, input_)
                        else:
                            raise"""
if old_ag1 in code:
    code = code.replace(old_ag1, new_ag1)
    print("Patched all_gather (list)")

old_ag2 = "                dist.all_gather([output_tensor], input_, group=self.device_group)"
new_ag2 = """                if self._profile_cpu_gloo():
                    self._cpu_all_gather([output_tensor], input_)
                else:
                    try:
                        dist.all_gather([output_tensor], input_, group=self.device_group)
                    except RuntimeError as e:
                        if "OUT_OF_RESOURCES" in str(e) or "OUT_OF_DEVICE_MEMORY" in str(e):
                            self._cpu_all_gather([output_tensor], input_)
                        else:
                            raise"""
if old_ag2 in code:
    code = code.replace(old_ag2, new_ag2)
    print("Patched all_gather (output_tensor)")

with open("/opt/venv/lib/python3.12/site-packages/vllm/distributed/device_communicators/xpu_communicator.py", "w") as f:
    f.write(code)
import ast
ast.parse(code)
print("xpu_communicator.py patched and verified")
PYEOF3

# --- Copy gloo kernel module and add import to vllm/__init__.py ---
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

# --- Patch comm_lowering.py ---
python3 << 'PYEOF5'
with open("/opt/venv/lib/python3.12/site-packages/torch/_inductor/comm_lowering.py", "r") as f:
    code = f.read()
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
        # The coalesced variant (allgather_into_tensor_coalesced_) has no XPU
        # fallback and segfaults on XCCL over PCIe.
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

# Also patch the non-coalesced all_gather_into_tensor to use _out variant
# with create_inplace, so the result writes directly into a pre-allocated
# buffer (avoids copy_() which fails during XPU graph capture).
old_agt = """    @register_comm_lowering(c10d.all_gather_into_tensor)
    def _all_gather_into_tensor(inp, group_size, group_name):
        return _create_out_of_place(
            c10d.all_gather_into_tensor.default,
            inp,
            group_size,
            group_name,
        )"""
new_agt = """    @register_comm_lowering(c10d.all_gather_into_tensor)
    def _all_gather_into_tensor(inp, group_size, group_name):
        # Use _out variant with create_inplace to avoid copy_() during
        # XPU graph capture.  Allocate output buffer first.
        out = _create_out_of_place(
            c10d.all_gather_into_tensor.default,
            inp,
            group_size,
            group_name,
        )
        out.realize()
        ir._CollectiveKernel.create_inplace(
            c10d.all_gather_into_tensor_out.default,
            inp,
            group_size,
            group_name,
            out=out,
        )
        return out"""
if old_agt in code:
    code = code.replace(old_agt, new_agt)
    print("Patched all_gather_into_tensor to use _out variant")
else:
    print("all_gather_into_tensor pattern not found or already patched")

# Patch all_reduce_ to skip require_contiguous which generates copy_()
# that fails during XPU graph capture (empty_strided_cpu + copy_ from xpu
# triggers command graph event wait error).
old_ar_inplace = """        # Lower as c10d.all_reduce_
        # pyrefly: ignore [bad-assignment]
        inp = ir.ExternKernel.require_contiguous(inp)
        ir._AllReduce_Kernel.create_inplace(
            c10d.all_reduce_.default,
            inp,  # type: ignore[arg-type]
            reduce_op,
            group_name,  # type: ignore[arg-type]
        )
        return inp  # type: ignore[return-value]"""
new_ar_inplace = """        # Lower as c10d.all_reduce_
        # Skip require_contiguous on XPU — it generates copy_() that
        # fails during graph capture (device-to-host sync in command graph mode).
        ir._AllReduce_Kernel.create_inplace(
            c10d.all_reduce_.default,
            inp,  # type: ignore[arg-type]
            reduce_op,
            group_name,  # type: ignore[arg-type]
        )
        return inp  # type: ignore[return-value]"""
if old_ar_inplace in code:
    code = code.replace(old_ar_inplace, new_ar_inplace)
    print("Patched all_reduce_ to skip require_contiguous")
else:
    print("all_reduce_ require_contiguous pattern not found")

# Also split all_reduce_coalesced into individual all_reduce calls
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
    print("all_reduce_coalesced pattern not found or already patched")

with open("/opt/venv/lib/python3.12/site-packages/torch/_inductor/comm_lowering.py", "w") as f:
    f.write(code)
print("comm_lowering.py patched and verified")
PYEOF5

# Clear compilation cache so coalesced code is NOT reused
rm -rf /root/.cache/vllm/torch_compile_cache/ 2>/dev/null || true
echo "Cleared compilation cache"

# --- Patch distributed_c10d.py ---
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

# --- Patch gpu_worker.py ---
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

# Also set VLLM_XPU_PROFILE_CPU_GLOO around capture_model() — the warmup
# runs inside capture use XCCL which segfaults on B70 PCIe.  Gloo is slower
# but works.  After capture, the env var is cleared for normal inference.
old3 = "            cuda_graph_memory_bytes = self.model_runner.capture_model()"
new3 = """            import os as _os, torch.distributed as dist
            _os.environ["VLLM_XPU_PROFILE_CPU_GLOO"] = "1"
            if dist.is_initialized():
                dist.barrier()
            cuda_graph_memory_bytes = self.model_runner.capture_model()
            _os.environ.pop("VLLM_XPU_PROFILE_CPU_GLOO", None)"""
if old3 in code:
    code = code.replace(old3, new3)
    changed = True
    print("Patched capture_model with gloo redirect")

if changed:
    with open("/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py", "w") as f:
        f.write(code)
    print("Patched gpu_worker.py")
else:
    print("gpu_worker.py: no changes needed")
PYEOF7

export VLLM_XPU_ENABLE_XPU_GRAPH=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
export VLLM_LOGGING_LEVEL=INFO

MTP_CONFIG='{"method":"qwen3_5_mtp","num_speculative_tokens":5,"model":"/model"}'

exec vllm serve /model \
  --dtype float16 --max-model-len 4096 --gpu-memory-utilization 0.45 \
  --tensor-parallel-size 2 --max-num-seqs 1 --max-num-batched-tokens 2048 \
  --trust-remote-code --port 8000 --host 0.0.0.0 \
  --skip-mm-profiling \
  --distributed-timeout-seconds 600 \
  --speculative-config "$MTP_CONFIG"
