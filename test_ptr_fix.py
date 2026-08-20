#!/usr/bin/env python3
"""Test that XPU pointers (0xFFFF... range) can be stored in int64 tensors
via numpy uint64 reinterpretation, preserving the bit pattern."""
import torch
import numpy as np

ptr = 0xFFFF123456789ABC  # typical XPU pointer range

# Approach: numpy uint64 array -> view as int64 -> torch from_numpy -> to xpu
cpu_uint = np.array([ptr, ptr + 0x1000], dtype=np.uint64)
print("numpy uint64:", [hex(int(x)) for x in cpu_uint])

cpu_int64 = cpu_uint.view(np.int64)
print("numpy int64 view:", [int(x) for x in cpu_int64])

cpu_t = torch.from_numpy(cpu_int64.copy())
print("torch CPU int64:", [int(x) for x in cpu_t])

# Move to XPU
xpu_t = cpu_t.to("xpu")
vals = [int(x) for x in xpu_t.to("cpu")]
print("torch XPU int64:", vals)

# Verify bit patterns match
all_match = True
for i, v in enumerate(vals):
    bits = v & 0xFFFFFFFFFFFFFFFF if v >= 0 else (v + (1 << 64))
    match = bits == int(cpu_uint[i])
    all_match = all_match and match
    print(f"  slot {i}: bits={bits:#018x} match={match}")

# Test copy_ to pre-existing XPU tensor (the actual use case in mamba_utils)
xpu_existing = torch.zeros(2, dtype=torch.int64, device="xpu")
xpu_existing.copy_(cpu_t)
vals2 = [int(x) for x in xpu_existing.to("cpu")]
print("copy_ to existing XPU:", vals2)
for i, v in enumerate(vals2):
    bits = v & 0xFFFFFFFFFFFFFFFF if v >= 0 else (v + (1 << 64))
    match = bits == int(cpu_uint[i])
    all_match = all_match and match
    print(f"  slot {i}: bits={bits:#018x} match={match}")

# Test the numpy .np buffer pattern used by CpuGpuBuffer
buf_np = np.zeros(2, dtype=np.uint64)
buf_np[0] = ptr
buf_np[1] = ptr + 0x2000
print("numpy uint64 buffer:", [hex(int(x)) for x in buf_np])

# Convert uint64 numpy to int64 torch (what copy_to_gpu would do)
buf_int64 = buf_np.view(np.int64)
buf_torch = torch.from_numpy(buf_int64.copy())
print("buffer as int64 torch:", [int(x) for x in buf_torch])

print(f"\n{'ALL PASS' if all_match else 'FAIL'}")
