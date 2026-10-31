#!/usr/bin/env bash
# Environment verification: interpreter, pinned stack, video decode, CUDA.
set -euo pipefail
python - <<'PY'
import sys, importlib
print("python", sys.version.split()[0])
for m in ["torch", "torchvision", "torchcodec", "deepspeed", "accelerate", "transformers",
          "datasets", "hydra", "h5py", "av", "fastwam", "fastwam.runtime", "fastwam.trainer"]:
    mod = importlib.import_module(m)
    print(f"  {m:18s} {getattr(mod, '__version__', 'ok')}")
import torch
print("cuda", torch.version.cuda, "available", torch.cuda.is_available(), "devices", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(" ", i, torch.cuda.get_device_name(i))
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
print("matmul ok", float((x @ x).float().abs().mean()) > 0)
from torchcodec.decoders import VideoDecoder  # links system libav*
print("torchcodec decoder import ok")
PY
ffmpeg -version | head -1
