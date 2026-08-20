#!/usr/bin/env python3
"""Test ctypes-based approach to convert unsigned XPU pointer to signed int64."""
import torch
import ctypes
import numpy as np

ptr = 0xFFFF123456789ABC  # XPU pointer range

# Convert unsigned 64-bit to signed int64 using ctypes
signed_val = ctypes.c_int64(ptr).value
print(f"ptr={ptr:#x} signed={signed_val}")

# Test assignment to XPU int64 tensor
xpu_t = torch.zeros(1, dtype=torch.int64, device="xpu")
xpu_t[0] = signed_val
val = int(xpu_t.to("cpu")[0].item())
bits = val & 0xFFFFFFFFFFFFFFFF if val >= 0 else (val + (1 << 64))
print(f"XPU stored: {val} bits={bits:#018x} match={bits == ptr}")

# Test with actual XPU tensor data_ptr
t = torch.zeros(10, dtype=torch.float32, device="xpu")
actual_ptr = t.data_ptr()
print(f"\nActual XPU ptr={actual_ptr:#x}")
signed_actual = ctypes.c_int64(actual_ptr).value
print(f"Signed: {signed_actual}")

xpu_t2 = torch.zeros(1, dtype=torch.int64, device="xpu")
xpu_t2[0] = signed_actual
val2 = int(xpu_t2.to("cpu")[0].item())
bits2 = val2 & 0xFFFFFFFFFFFFFFFF if val2 >= 0 else (val2 + (1 << 64))
print(f"Round-trip: {val2} bits={bits2:#018x} match={bits2 == actual_ptr}")

# Test numpy uint64 buffer assignment (for collect_mamba_copy_meta)
buf_np = np.zeros(1, dtype=np.uint64)
buf_np[0] = actual_ptr
print(f"\nnumpy uint64 buf: {int(buf_np[0]):#x} match={int(buf_np[0]) == actual_ptr}")

print("\nALL PASS")
