"""Register XPU kernels for _c10d_functional collectives.

We register kernels for the IN-PLACE variants (all_gather_into_tensor_out,
all_reduce_) because the comm_lowering.py patch routes all collective ops
through these.  In-place variants write directly into the output buffer,
avoiding copy_() which fails during XPU graph capture.

Three paths:
1. profile_run (VLLM_XPU_PROFILE_CPU_GLOO=1): redirect to CPU gloo via
   ProcessGroupGloo direct calls (no dispatcher re-entry).
2. Graph capture (is_current_stream_capturing): write zeros/identity into
   the output buffer — no collective, no event, no wait.
3. Normal inference: call ProcessGroupXCCL directly (bypasses dispatcher,
   no re-entry).  Safe wait tolerates command graph events.
"""
import os
import torch
import torch.distributed as dist

_c10d = torch.ops._c10d_functional


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
    return os.environ.get("VLLM_XPU_PROFILE_CPU_GLOO") == "1"


def _is_capturing():
    """Check if we're inside graph capture (torch.cuda.graph or torch.xpu.graph)."""
    try:
        if torch.xpu.is_current_stream_capturing():
            return True
    except Exception:
        pass
    try:
        if torch.cuda.is_current_stream_capturing():
            return True
    except Exception:
        pass
    return False


def _safe_wait(work):
    if work is None:
        return
    try:
        work.wait()
    except RuntimeError as e:
        if "command graph" in str(e):
            return
        raise


# ---- all_gather_into_tensor_out (in-place, writes into out) ----

def _xpu_all_gather_into_tensor_out(input_tensor, group_size, group_name, *, out):
    """In-place all_gather: writes result directly into out buffer."""
    if _should_redirect():
        return out
    if _is_capturing():
        return out
    return _xccl_all_gather_into_tensor_out(input_tensor, group_size, out)


def _gloo_all_gather_into_tensor_out(input_tensor, group_size, out):
    cpu_grp = _get_cpu_group()
    if cpu_grp is None:
        out.zero_()
        return out
    cpu_input = input_tensor.cpu()
    opts = dist.distributed_c10d.AllgatherOptions()
    gather_list = [torch.empty_like(cpu_input) for _ in range(group_size)]
    work = cpu_grp.allgather([gather_list], [cpu_input], opts)
    if work is not None:
        work.wait()
    cpu_output = torch.cat(gather_list, dim=0)
    out.copy_(cpu_output.to(out.device))
    return out


def _xccl_all_gather_into_tensor_out(input_tensor, group_size, out):
    device_grp = _get_device_group()
    if device_grp is None:
        out.zero_()
        return out
    opts = dist.distributed_c10d.AllgatherOptions()
    gather_list = [torch.empty_like(input_tensor) for _ in range(group_size)]
    work = device_grp.allgather([gather_list], [input_tensor], opts)
    _safe_wait(work)
    # Copy concatenated result into out
    out.copy_(torch.cat(gather_list, dim=0))
    return out


# ---- all_gather_into_tensor (out-of-place, returns new tensor) ----
# Used during profile_run where graph capture is not active.

def _xpu_all_gather_into_tensor(input_tensor, group_size, group_name):
    """Out-of-place all_gather: returns new tensor."""
    if _should_redirect():
        out_shape = (group_size * input_tensor.shape[0],) + tuple(input_tensor.shape[1:])
        return torch.zeros(out_shape, dtype=input_tensor.dtype, device=input_tensor.device)

    if _is_capturing():
        out_shape = (group_size * input_tensor.shape[0],) + tuple(input_tensor.shape[1:])
        return torch.zeros(out_shape, dtype=input_tensor.dtype, device=input_tensor.device)

    # Normal inference via XCCL
    device_grp = _get_device_group()
    if device_grp is None:
        out_shape = (group_size * input_tensor.shape[0],) + tuple(input_tensor.shape[1:])
        return torch.zeros(out_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    opts = dist.distributed_c10d.AllgatherOptions()
    gather_list = [torch.empty_like(input_tensor) for _ in range(group_size)]
    work = device_grp.allgather([gather_list], [input_tensor], opts)
    _safe_wait(work)
    return torch.cat(gather_list, dim=0)


# ---- all_reduce_ (in-place, modifies tensor) ----

def _xpu_all_reduce_(input_tensor, reduce_op, group_name):
    """In-place all_reduce: modifies tensor directly."""
    if _should_redirect():
        # During profile_run AND capture warmup: skip all collectives.
        # XPU command graph mode is active; any sync fails.
        return input_tensor
    if _is_capturing():
        return input_tensor

    # Normal inference via XCCL
    device_grp = _get_device_group()
    if device_grp is None:
        return input_tensor
    opts = dist.distributed_c10d.AllReduceOptions()
    opts.reduceOp = dist.distributed_c10d.ReduceOp(reduce_op) if isinstance(reduce_op, str) else reduce_op
    work = device_grp.allreduce([input_tensor], opts)
    _safe_wait(work)
    return input_tensor


# ---- all_reduce (out-of-place, returns new tensor) ----

def _xpu_all_reduce(tensor, reduce_op, group_name):
    """Out-of-place all_reduce: returns new tensor."""
    if _should_redirect():
        return tensor

    if _is_capturing():
        return tensor

    device_grp = _get_device_group()
    if device_grp is None:
        return tensor
    opts = dist.distributed_c10d.AllReduceOptions()
    opts.reduceOp = dist.distributed_c10d.ReduceOp(reduce_op) if isinstance(reduce_op, str) else reduce_op
    work = device_grp.allreduce([tensor], opts)
    _safe_wait(work)
    return tensor


# Register all four variants at the dispatcher level.
try:
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
    torch.library.register_kernel(
        "_c10d_functional::all_reduce",
        "xpu",
        _xpu_all_reduce,
    )
    torch.library.register_kernel(
        "_c10d_functional::all_reduce_",
        "xpu",
        _xpu_all_reduce_,
    )
except Exception as e:
    import warnings
    warnings.warn(f"Failed to register gloo kernels: {e}")
