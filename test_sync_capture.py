import torch
g = torch.xpu.XPUGraph()
try:
    with torch.xpu.graph(g):
        torch.xpu.synchronize()
    print("synchronize during capture: OK")
except Exception as e:
    print("synchronize during capture: FAIL:", e)

# Also test: can we run a simple xpu op during capture?
g2 = torch.xpu.XPUGraph()
try:
    with torch.xpu.graph(g2):
        x = torch.ones(10, device="xpu")
        y = x * 2
    print("simple op during capture: OK")
except Exception as e:
    print("simple op during capture: FAIL:", e)
