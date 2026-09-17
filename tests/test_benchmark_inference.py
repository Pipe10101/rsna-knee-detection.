import os
import sys
import torch
import pytest

import importlib.util
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("benchmark_inference", os.path.join(ROOT, "scripts", "benchmark_inference.py"))
mod = importlib.util.module_from_spec(spec)
sys.modules["scripts.benchmark_inference"] = mod
spec.loader.exec_module(mod)
resolve_autocast_dtype = mod.resolve_autocast_dtype
pick_device = mod.pick_device

def test_resolve_autocast_dtype():
    cpu_dev = torch.device("cpu")
    assert resolve_autocast_dtype(cpu_dev) == torch.bfloat16
    
    cuda_dev = torch.device("cuda")
    dtype = resolve_autocast_dtype(cuda_dev)
    assert dtype in (torch.float16, torch.bfloat16)

def test_pick_device():
    dev = pick_device()
    assert isinstance(dev, torch.device)
