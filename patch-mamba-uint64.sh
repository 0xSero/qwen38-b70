#!/usr/bin/env bash
# patch-mamba-uint64.sh — Fix XPU pointer overflow in mamba_utils.py
#
# Root cause: XPU device pointers are in the 0xFFFF... range (> int64 max).
# Assigning data_ptr() to torch.int64 tensors causes:
#   ValueError: Overflow when unpacking long long
#
# Fix: Convert unsigned pointer to signed int64 via ctypes.c_int64().
# The bit pattern is preserved (two's complement), and Triton's tl.load()
# + .to(tl.pointer_type()) correctly reconstructs the pointer.
#
# This is NOT masking (which corrupted output). This is proper unsigned→signed
# reinterpretation that preserves the exact bit pattern.
set -euo pipefail

CONTAINER="${1:-vllm-autoround-1gpu}"
FILE="/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/mamba_utils.py"

echo "Patching $FILE in container $CONTAINER..."

# 1. Add ctypes import after the torch import (line ~8)
docker exec "$CONTAINER" python3 -c "
import re

with open('$FILE', 'r') as f:
    code = f.read()

# Check if already patched
if 'ctypes.c_int64' in code:
    print('Already patched, skipping')
    exit(0)

# Add ctypes import after 'import torch'
code = code.replace(
    'import torch\n',
    'import ctypes\nimport torch\n',
    1
)

# Fix line 767: self.state_base_addrs[idx] = state.data_ptr()
code = code.replace(
    'self.state_base_addrs[idx] = state.data_ptr()',
    'self.state_base_addrs[idx] = ctypes.c_int64(state.data_ptr()).value'
)

# Fix line 820: base_addr = state.data_ptr()
code = code.replace(
    'base_addr = state.data_ptr()',
    'base_addr = ctypes.c_int64(state.data_ptr()).value'
)

# Fix line 855: self.block_table_ptrs[i] = bt.data_ptr()
code = code.replace(
    'self.block_table_ptrs[i] = bt.data_ptr()',
    'self.block_table_ptrs[i] = ctypes.c_int64(bt.data_ptr()).value'
)

with open('$FILE', 'w') as f:
    f.write(code)

print('Patched 3 data_ptr() assignments with ctypes.c_int64() conversion')
print('Also added ctypes import')
"

echo "Verifying patch..."
docker exec "$CONTAINER" grep -n 'ctypes.c_int64' "$FILE"
docker exec "$CONTAINER" grep -n 'import ctypes' "$FILE"

echo "Done."
