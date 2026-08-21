import torch
for i in range(torch.xpu.device_count()):
    props = torch.xpu.get_device_properties(i)
    used = torch.xpu.memory_allocated(i) / 1e9
    total = props.total_memory / 1e9
    free = total - used
    print(f"GPU{i}: {used:.1f} / {total:.1f} GB used, {free:.1f} GB free")
