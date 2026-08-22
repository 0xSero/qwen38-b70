"""
esimd_int4_bridge.py — Route AutoRound/INC INT4 linear layers through ESIMD GEMV/GEMM kernels.

Problem: The Qwen3.8-27B model uses AutoRound INT4 quantization (quant_method="auto-round",
dispatched to INCConfig). INCXPULinearMethod.apply() always calls the generic oneDNN
int4_gemm_w4a16 — even for M==1 decode, which is really a GEMV. The hand-tuned ESIMD
kernels (esimd_gemv_int4 for M==1, esimd_gemm_int4_pgrp for M>=2) are purpose-built for
Intel Arc XMX but never invoked because they're only called from GDN-specific code paths
gated behind quant_is_sym_int4(), which returns False for INC/AutoRound.

Solution: Monkey-patch at import time to:
1. Register weight_esimd/scale_esimd attributes on INC INT4 layers (the ESIMD kernels need these).
2. Patch INCXPULinearMethod.apply() to dispatch to esimd_gemv_int4 (M==1) or
   esimd_gemm_int4_pgrp (2<=M<=64) when weight_esimd is available, falling back to
   oneDNN for large M (prefill) or when constraints aren't met.

This routes ALL INT4 linear layers — MLP gate_up/down, GQA qkv/o, GDN in_proj_qkvz/out_proj —
through the hand-tuned Intel ESIMD kernels at decode time, not just the GDN-specific paths.

Weight format compatibility:
  INC/GPTQ: qweight [K/8, N] int32 (after process_weights_after_loading, strides=(1, K/8))
             scales [K/group, N] fp16, qzeros scalar int8=8
  ESIMD:     weight_esimd [N, K/2] uint8 (2 int4 per byte, low nibble = even index)
             scale_esimd  [N, K/group] fp16

  Both use symmetric int4 with zero_point=8 and the same sequential nibble packing.
  Conversion = transpose + contiguous + uint8 view.
"""

import torch
from torch.nn.parameter import Parameter
import vllm.model_executor.layers.quantization.inc as inc_mod
import vllm.model_executor.layers.esimd_utils as esimd_utils

_orig_inc_process = inc_mod.INCXPULinearMethod.process_weights_after_loading
_orig_inc_apply = inc_mod.INCXPULinearMethod.apply

# Group size from the AutoRound config — must match QK4_GROUP_SIZE (128)
_BRIDGE_GROUP_SIZE = 128

# Maximum M for ESIMD GEMM dispatch (match GDN code's threshold)
_ESIMD_GEMM_MAX_M = 64

# One-time dispatch logging
_dispatch_logged = False


def _patched_inc_process_weights_after_loading(self, layer: torch.nn.Module) -> None:
    """Call original INC processing, then register ESIMD weight views."""
    _orig_inc_process(self, layer)

    # Only register ESIMD views for INT4 layers (weight_bits=4, sym=True)
    if getattr(self, "weight_bits", 0) != 4 or not getattr(self, "sym", False):
        return
    if not esimd_utils.ESIMD_AVAILABLE or esimd_utils.disable_esimd_int4():
        return
    if self.group_size != _BRIDGE_GROUP_SIZE:
        return

    # Check if this layer has the INC quantized weight params
    if not hasattr(layer, "qweight") or not hasattr(layer, "scales"):
        return

    # After INC's process_weights_after_loading:
    #   layer.qweight is [K_packed, N] with strides (1, K_packed) — NT layout
    #   layer.scales is [K/group, N] contiguous
    #
    # ESIMD expects:
    #   weight_esimd = [N, K_packed] as uint8, then view as [N, K/2] uint8
    #   scale_esimd  = [N, K/group] as fp16
    qweight = layer.qweight.data
    scales = layer.scales.data

    weight_esimd = qweight.t().contiguous().view(torch.uint8)
    scale_esimd = scales.t().contiguous()

    layer.weight_esimd = Parameter(weight_esimd, requires_grad=False)
    layer.scale_esimd = Parameter(scale_esimd, requires_grad=False)

    prefix = getattr(layer, "prefix", getattr(layer, "name", "?"))
    print(f"[esimd_int4_bridge] Registered weight_esimd on {prefix}: "
          f"qweight={qweight.shape} -> weight_esimd={weight_esimd.shape}, "
          f"scale_esimd={scale_esimd.shape}")


def _patched_inc_apply(self, layer, x, bias=None):
    """Dispatch to ESIMD INT4 GEMV (M==1 only) when available.

    The ESIMD GEMM kernel (esimd_gemm_int4_pgrp) is slower than oneDNN for
    M>=2 batched decode on BMG-G31, so we only dispatch to ESIMD for the
    M==1 GEMV case (single-token decode, which is bandwidth-bound and
    benefits from the hand-tuned GEMV kernel's lower launch overhead).

    Falls back to oneDNN int4_gemm_w4a16 for all other cases.
    """
    global _dispatch_logged

    # Fast path: ESIMD INT4 GEMV for single-token decode only
    if (
        bias is None
        and hasattr(layer, "weight_esimd")
        and hasattr(layer, "scale_esimd")
        and esimd_utils.ESIMD_AVAILABLE
        and not esimd_utils.disable_esimd_int4()
        and x.reshape(-1, x.shape[-1]).shape[0] == 1
        and x.dtype == torch.float16
    ):
        reshaped_x = x.reshape(-1, x.shape[-1])
        N = layer.weight_esimd.shape[0]
        out = torch.empty(1, N, dtype=torch.float16, device=x.device)
        esimd_utils.esimd_gemv_int4(
            reshaped_x, layer.weight_esimd, layer.scale_esimd, out
        )
        if not _dispatch_logged:
            print(f"[esimd_int4_bridge] ESIMD GEMV dispatched: "
                  f"M=1, N={N}, K={reshaped_x.shape[1]}, "
                  f"layer={getattr(layer, 'prefix', '?')}")
            _dispatch_logged = True
        out_shape = x.shape[:-1] + (N,)
        return out.reshape(out_shape)

    # Fall back to oneDNN int4_gemm_w4a16
    out_shape = x.shape[:-1] + (layer.qweight.shape[1],)
    reshaped_x = x.reshape(-1, x.shape[-1])
    out = torch.ops._xpu_C.int4_gemm_w4a16(
        reshaped_x,
        layer.qweight,
        bias,
        layer.scales,
        layer.qzeros,
        self.group_size,
        None,
    )
    return out.reshape(out_shape)


def apply_patches():
    """Apply all ESIMD INT4 bridge patches. Call once at import time."""
    inc_mod.INCXPULinearMethod.process_weights_after_loading = (
        _patched_inc_process_weights_after_loading
    )
    inc_mod.INCXPULinearMethod.apply = _patched_inc_apply

    print("[esimd_int4_bridge] Patched INC apply() -> ESIMD INT4 GEMV/GEMM dispatch")


# Apply at import time
apply_patches()
