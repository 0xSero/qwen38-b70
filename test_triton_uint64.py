#!/usr/bin/env python3
"""Test that Triton kernels can load int64 pointer values (with XPU pointer
bit patterns stored via numpy reinterpretation) and use them as pointers."""
import torch
import triton
import triton.language as tl
import numpy as np

@triton.jit
def test_kernel(ptrs_tensor, N: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= N:
        return
    ptr_val = tl.load(ptrs_tensor + pid)
    out_ptr = ptr_val.to(tl.pointer_type(tl.float32))
    tl.store(out_ptr, pid * 2.0)

N = 4
targets = [torch.zeros(1, dtype=torch.float32, device="xpu") for _ in range(N)]
ptrs = np.array([int(t.data_ptr()) for t in targets], dtype=np.uint64)
print("Pointers:", [hex(int(p)) for p in ptrs])

# Store as int64 via numpy reinterpretation (the fix approach)
np_int64 = ptrs.view(np.int64)
ptrs_tensor = torch.from_numpy(np_int64.copy()).to("xpu")
print("int64 tensor values:", [int(x) for x in ptrs_tensor.to("cpu")])

# Run kernel
test_kernel[(N,)](ptrs_tensor, N=N)

# Check results
all_ok = True
for i, t in enumerate(targets):
    val = float(t.item())
    ok = val == i * 2.0
    all_ok = all_ok and ok
    print(f"  target[{i}] = {val} (expected {i * 2.0}) {'OK' if ok else 'FAIL'}")

print(f"\n{'ALL PASS' if all_ok else 'FAIL'}")
