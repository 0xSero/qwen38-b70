from vllm.config import CUDAGraphMode
m = CUDAGraphMode.FULL_AND_PIECEWISE
print("mixed_mode:", m.mixed_mode())
print("decode_mode:", m.decode_mode())
print("has_full:", m.has_mode(CUDAGraphMode.FULL))
print("has_piecewise:", m.has_mode(CUDAGraphMode.PIECEWISE))
print("separate_routine:", m.separate_routine())
