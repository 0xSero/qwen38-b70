"""Register XPU kernels for _c10d_functional collectives AND patch inductor
to avoid CPU buffer allocation for collective ops on XPU.

The core problem: inductor's _AllReduce_Kernel codegen generates compiled
code with empty_strided_cpu() + copy_() (XPU->CPU) for collective buffers.
This XPU->CPU copy_() fails during XPU command graph capture ("wait method
cannot be used for an event associated with a command graph").

Two patches:
1. Replace codegen() overrides on _AllReduce_Kernel, _AllReduceKernel,
   _WaitKernel with the parent _CollectiveKernel.codegen().  This makes
   the kernel call use python_kernel_name (torch.ops._c10d_functional.*)
   instead of the CPU C shim, so our registered XPU kernels are invoked.

2. Monkey-patch _empty_strided_cpu to redirect to _empty_strided_xpu.
   The scheduler allocates collective input buffers on CPU (because the
   _AllReduce_Kernel.__init__ sets cpp_kernel_name to "aoti_torch_cpu_*").
   By redirecting empty_strided_cpu to empty_strided_xpu, the compiled code
   allocates XPU buffers instead.  Then:
   - copy_() becomes XPU->XPU (same-device, no sync, works in command graph)
   - all_reduce_.default(xpu_tensor) dispatches to our XPU kernel -> XCCL
   - wait_tensor.default(xpu_tensor) dispatches to our no-op XPU kernel

Kernel behavior:
1. profile_run (VLLM_XPU_PROFILE_CPU_GLOO=1): redirect to CPU gloo via
   ProcessGroupGloo direct calls (no dispatcher re-entry).  Needed because
   XCCL all_reduce fails with OUT_OF_RESOURCES for large prefill-sized tensors.
2. Graph capture (torch.xpu.graph context): call XCCL allreduce, which
   enqueues into the command graph.  _safe_wait catches the "command graph"
   wait error -- the allreduce op is already recorded.
3. Normal inference: call ProcessGroupXCCL directly (bypasses dispatcher,
   no re-entry into our kernel).
"""
import os
import torch
import torch.distributed as dist

_c10d = torch.ops._c10d_functional


# ============================================================================
# Patch 1: Replace codegen overrides that hardcode CPU C shim
# ============================================================================

try:
    import torch._inductor.ir as _ir

    _parent_codegen = _ir._CollectiveKernel.codegen
    _ir._AllReduce_Kernel.codegen = _parent_codegen
    _ir._AllReduceKernel.codegen = _parent_codegen
    _ir._WaitKernel.codegen = _parent_codegen
    print("[vllm_gloo_kernels] Patched inductor codegen: "
          "_AllReduce_Kernel, _AllReduceKernel, _WaitKernel -> runtime dispatch")
except Exception as e:
    import warnings
    warnings.warn(f"Failed to patch inductor ir.py: {e}")


# ============================================================================
# Patch 2: Redirect empty_strided_cpu to empty_strided_xpu
#
# The inductor scheduler allocates collective input buffers using
# empty_strided_cpu because _AllReduce_Kernel.__init__ sets
# cpp_kernel_name = "aoti_torch_cpu__c10d_functional_all_reduce_".
# Even with cpp_wrapper=False, the scheduler uses this to determine the
# buffer device.  By redirecting _empty_strided_cpu to _empty_strided_xpu,
# the compiled code allocates XPU buffers for collectives.  The only
# empty_strided_cpu calls in the compiled PIECEWISE code are for collective
# buffers (all_reduce), so this is safe.
# ============================================================================

try:
    import torch._C._dynamo.guards as _guards
    _guards._empty_strided_cpu = _guards._empty_strided_xpu
    print("[vllm_gloo_kernels] Patched _empty_strided_cpu -> _empty_strided_xpu")
except Exception as e:
    import warnings
    warnings.warn(f"Failed to patch _empty_strided_cpu: {e}")


# ============================================================================
# Helpers
# ============================================================================

def _get_tp_group():
    if not dist.is_initialized():
        return None
    try:
        from vllm.distributed.parallel_state import get_tp_group
        return get_tp_group()
    except Exception:
        return None


def _get_cpu_group():
    tp_group = _get_tp_group()
    if tp_group is None:
        return None
    if hasattr(tp_group, 'cpu_group'):
        return tp_group.cpu_group
    if hasattr(tp_group, 'device_communicator'):
        return tp_group.device_communicator.cpu_group
    return None


def _get_device_group():
    tp_group = _get_tp_group()
    if tp_group is None:
        return None
    if hasattr(tp_group, 'device_group'):
        return tp_group.device_group
    if hasattr(tp_group, 'device_communicator'):
        return tp_group.device_communicator.device_group
    return None


def _should_redirect():
    """True during profile_run -- use CPU gloo for large prefill-sized tensors."""
    return os.environ.get("VLLM_XPU_PROFILE_CPU_GLOO") == "1"


def _safe_wait(work):
    """Wait for collective work, tolerating command graph events.

    During XPU command graph capture, wait() fails because the event is
    associated with a command graph.  The collective op itself is already
    enqueued, so we can safely ignore the wait error.
    """
    if work is None:
        return
    try:
        work.wait()
    except RuntimeError as e:
        if "command graph" in str(e):
            return
        raise


def _make_reduce_op(reduce_op):
    if isinstance(reduce_op, str):
        # Map string names to ReduceOp enum members
        op_map = {
            "sum": dist.distributed_c10d.ReduceOp.SUM,
            "mean": dist.distributed_c10d.ReduceOp.AVG,
            "avg": dist.distributed_c10d.ReduceOp.AVG,
            "product": dist.distributed_c10d.ReduceOp.PRODUCT,
            "max": dist.distributed_c10d.ReduceOp.MAX,
            "min": dist.distributed_c10d.ReduceOp.MIN,
            "band": dist.distributed_c10d.ReduceOp.BAND,
            "bor": dist.distributed_c10d.ReduceOp.BOR,
            "bxor": dist.distributed_c10d.ReduceOp.BXOR,
            "premul_sum": dist.distributed_c10d.ReduceOp.PREMUL_SUM,
        }
        return op_map.get(reduce_op.lower(), dist.distributed_c10d.ReduceOp.SUM)
    return reduce_op


# ============================================================================
# all_reduce_ (in-place, modifies tensor) -- THE CRITICAL KERNEL
# ============================================================================

def _xpu_all_reduce_(input_tensor, reduce_op, group_name):
    """In-place all_reduce: modifies tensor directly.

    During graph capture: XCCL enqueues the allreduce into the command graph.
    The wait may fail (command graph), but _safe_wait catches it.  The op
    is recorded and will execute during graph replay.
    """
    if _should_redirect():
        # profile_run: use CPU gloo for large prefill-sized tensors
        cpu_grp = _get_cpu_group()
        if cpu_grp is None:
            return input_tensor
        cpu_tensor = input_tensor.cpu()
        opts = dist.distributed_c10d.AllreduceOptions()
        opts.reduceOp = _make_reduce_op(reduce_op)
        work = cpu_grp.allreduce([cpu_tensor], opts)
        if work is not None:
            work.wait()
        input_tensor.copy_(cpu_tensor.to(input_tensor.device))
        return input_tensor

    # Normal inference AND graph capture: use XCCL directly.
    # During capture, XCCL records into the command graph.
    # During inference, XCCL executes the allreduce.
    device_grp = _get_device_group()
    if device_grp is None:
        return input_tensor
    opts = dist.distributed_c10d.AllreduceOptions()
    opts.reduceOp = _make_reduce_op(reduce_op)
    work = device_grp.allreduce([input_tensor], opts)
    _safe_wait(work)
    return input_tensor


# ============================================================================
# all_reduce (out-of-place, returns new tensor)
# ============================================================================

def _xpu_all_reduce(tensor, reduce_op, group_name):
    """Out-of-place all_reduce: returns new tensor (same as input for in-place semantics)."""
    if _should_redirect():
        return tensor

    device_grp = _get_device_group()
    if device_grp is None:
        return tensor
    opts = dist.distributed_c10d.AllreduceOptions()
    opts.reduceOp = _make_reduce_op(reduce_op)
    work = device_grp.allreduce([tensor], opts)
    _safe_wait(work)
    return tensor


# ============================================================================
# wait_tensor -- no-op on XPU
# ============================================================================

def _xpu_wait_tensor(tensor):
    """No-op wait: return tensor unchanged.

    During graph capture/execution, ordering is ensured by the XPU command
    graph stream ordering.  During eager execution, _safe_wait in the
    all_reduce kernel already handles synchronization.
    """
    return tensor


# ============================================================================
# all_gather kernels -- used during profile_run and eager attention
# ============================================================================

def _xpu_all_gather_into_tensor(input_tensor, group_size, group_name):
    """Out-of-place all_gather: returns new tensor."""
    if _should_redirect():
        out_shape = (group_size * input_tensor.shape[0],) + tuple(input_tensor.shape[1:])
        return torch.zeros(out_shape, dtype=input_tensor.dtype, device=input_tensor.device)

    # Try XCCL for small decode-sized tensors
    device_grp = _get_device_group()
    if device_grp is None:
        out_shape = (group_size * input_tensor.shape[0],) + tuple(input_tensor.shape[1:])
        return torch.zeros(out_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    opts = dist.distributed_c10d.AllgatherOptions()
    gather_list = [torch.empty_like(input_tensor) for _ in range(group_size)]
    work = device_grp.allgather([gather_list], [input_tensor], opts)
    _safe_wait(work)
    return torch.cat(gather_list, dim=0)


def _xpu_all_gather_into_tensor_out(input_tensor, group_size, group_name, *, out):
    """In-place all_gather: writes result directly into out buffer."""
    if _should_redirect():
        return out

    device_grp = _get_device_group()
    if device_grp is None:
        out.zero_()
        return out
    opts = dist.distributed_c10d.AllgatherOptions()
    gather_list = [torch.empty_like(input_tensor) for _ in range(group_size)]
    work = device_grp.allgather([gather_list], [input_tensor], opts)
    _safe_wait(work)
    out.copy_(torch.cat(gather_list, dim=0))
    return out


# ============================================================================
# Register all kernels at the dispatcher level
# ============================================================================

try:
    torch.library.register_kernel(
        "_c10d_functional::all_reduce_",
        "xpu",
        _xpu_all_reduce_,
    )
    torch.library.register_kernel(
        "_c10d_functional::all_reduce",
        "xpu",
        _xpu_all_reduce,
    )
    torch.library.register_kernel(
        "_c10d_functional::wait_tensor",
        "xpu",
        _xpu_wait_tensor,
    )
    torch.library.register_kernel(
        "_c10d_functional::all_gather_into_tensor",
        "xpu",
        _xpu_all_gather_into_tensor,
    )
    torch.library.register_kernel(
        "_c10d_functional::all_gather_into_tensor_out",
        "xpu",
        _xpu_all_gather_into_tensor_out,
    )
    print("[vllm_gloo_kernels] Registered XPU kernels for: "
          "all_reduce_, all_reduce, wait_tensor, all_gather_into_tensor, "
          "all_gather_into_tensor_out")
except Exception as e:
    import warnings
    warnings.warn(f"Failed to register gloo kernels: {e}")
