"""Run this on the remote NVIDIA box, not on the Mac."""

import torch

print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
try:
    import triton

    print("triton", triton.__version__)
except ImportError:
    print("triton NOT INSTALLED")
