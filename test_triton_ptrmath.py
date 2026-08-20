#!/usr/bin/env python3
"""Test pointer ARITHMETIC with negative int64 (two's complement) base addresses
in Triton on XPU. This mirrors what postprocess_mamba_fused_kernel does:
  state_base_addr (negative int64 from XPU pointer) + offset * stride -> pointer"""
import torch
import triton
import triton.language as tl
import ctypes
import numpy as np

@triton.jit
def ptr_arith_kernel(base_addrs, offsets, strides, out_vals, N: tl.constexpr):
    """Load a base address (int64, may be negative), compute
    base + offset * stride, store the result as int64, AND
    dereference it as a float32 pointer to verify correctness."""
    pid = tl.program_id(0)
    if pid >= N:
        return
    base = tl.load(base_addrs + pid)
    offset = tl.load(offsets + pid)
    stride = tl.load(strides + pid)
    addr = base + offset * stride
    # Store the computed address for verification
    tl.store(out_vals + pid, addr)
    # Also try to use it as a pointer — write a known value
    fptr = addr.to(tl.pointer_type(tl.float32))
    tl.store(fptr, pid * 100.0 + 42.0)

N = 4
# Allocate target float32 tensors on XPU (1 element each)
targets = [torch.zeros(1, dtype=torch.float32, device="xpu") for _ in range(N)]

# These are the actual XPU base addresses (0xFFFF... range)
base_ptrs = [int(t.data_ptr()) for t in targets]
print("Base pointers:", [hex(p) for p in base_ptrs])

# Convert to signed int64 via ctypes (our patch approach)
signed_bases = [ctypes.c_int64(p).value for p in base_ptrs]
print("Signed int64:", signed_bases)

# The offset and stride: offset=0, stride=4 (float32 = 4 bytes)
# So addr = base + 0 * 4 = base (should point to the start of each target)
offsets = [0, 0, 0, 0]
strides = [4, 4, 4, 4]

# Create XPU tensors
base_t = torch.tensor(signed_bases, dtype=torch.int64, device="xpu")
offset_t = torch.tensor(offsets, dtype=torch.int64, device="xpu")
stride_t = torch.tensor(strides, dtype=torch.int64, device="xpu")
out_t = torch.zeros(N, dtype=torch.int64, device="xpu")

# Run kernel
ptr_arith_kernel[(N,)](base_t, offset_t, stride_t, out_t, N=N)

# Check computed addresses
out_vals = [int(x) for x in out_t.to("cpu")]
print("\nComputed addresses (int64):", out_vals)
for i, (v, expected) in enumerate(zip(out_vals, base_ptrs)):
    bits = v & 0xFFFFFFFFFFFFFFFF if v >= 0 else (v + (1 << 64))
    match = bits == expected
    print(f"  slot {i}: computed={bits:#018x} expected={expected:#018x} match={match}")

# Check if the pointers actually wrote to the right place
print("\nTarget values (should be pid*100+42):")
all_ok = True
for i, t in enumerate(targets):
    val = float(t.item())
    expected = i * 100.0 + 42.0
    ok = val == expected
    all_ok = all_ok and ok
    print(f"  target[{i}] = {val} (expected {expected}) {'OK' if ok else 'FAIL'}")

# Now test with NON-ZERO offset to verify arithmetic
targets2 = [torch.zeros(4, dtype=torch.float32, device="xpu") for _ in range(N)]
base_ptrs2 = [int(t.data_ptr()) for t in targets2]
signed_bases2 = [ctypes.c_int64(p).value for p in base_ptrs2]
# Write to element at offset 2 (byte offset = 2*4 = 8)
offsets2 = [2, 2, 2, 2]
strides2 = [4, 4, 4, 4]

base_t2 = torch.tensor(signed_bases2, dtype=torch.int64, device="xpu")
offset_t2 = torch.tensor(offsets2, dtype=torch.int64, device="xpu")
stride_t2 = torch.tensor(strides2, dtype=torch.int64, device="xpu")
out_t2 = torch.zeros(N, dtype=torch.int64, device="xpu")

ptr_arith_kernel[(N,)](base_t2, offset_t2, stride_t2, out_t2, N=N)

print("\nWith offset=2 (should write to element[2]):")
for i, t in enumerate(targets2):
    vals = [float(x) for x in t.to("cpu")]
    expected_val = i * 100.0 + 42.0
    ok = vals[2] == expected_val and vals[0] == 0.0 and vals[1] == 0.0
    all_ok = all_ok and ok
    print(f"  target[{i}] = {vals} (expected [{expected_val} at idx 2]) {'OK' if ok else 'FAIL'}")

print(f"\n{'ALL PASS' if all_ok else 'FAIL'}")
