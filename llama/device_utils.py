"""
Device detection utilities for cross-platform compatibility.

Supports:
- CUDA (NVIDIA GPUs)
- MPS (Apple Silicon)
- CPU (fallback)
"""

import os
import torch
from typing import Optional, Literal

DeviceType = Literal["cuda", "mps", "cpu"]

# Global device cache
_device: Optional[torch.device] = None
_device_type: Optional[DeviceType] = None


def get_device_type() -> DeviceType:
    """
    Detect the best available device type.

    Returns:
        "cuda" if NVIDIA GPU available
        "mps" if Apple Silicon GPU available
        "cpu" otherwise
    """
    global _device_type

    if _device_type is not None:
        return _device_type

    if torch.cuda.is_available():
        _device_type = "cuda"
    elif torch.backends.mps.is_available():
        _device_type = "mps"
    else:
        _device_type = "cpu"

    return _device_type


def get_device() -> torch.device:
    """
    Get the torch.device object for the best available device.

    Returns:
        torch.device configured for cuda, mps, or cpu
    """
    global _device

    if _device is not None:
        return _device

    device_type = get_device_type()
    _device = torch.device(device_type)

    return _device


def get_device_name() -> str:
    """
    Get a human-readable name for the current device.

    Returns:
        Device name string (e.g., "NVIDIA RTX 3090", "Apple M2 Max", "CPU")
    """
    device_type = get_device_type()

    if device_type == "cuda":
        return torch.cuda.get_device_name(torch.cuda.current_device())
    elif device_type == "mps":
        return "Apple Silicon (MPS)"
    else:
        return "CPU"


def get_distributed_backend() -> str:
    """
    Get the appropriate distributed backend for the current device.

    Returns:
        "nccl" for CUDA, "gloo" for MPS/CPU
    """
    device_type = get_device_type()

    if device_type == "cuda":
        return "nccl"
    else:
        # gloo works for both MPS and CPU
        return "gloo"


def setup_device_environment():
    """
    Configure environment variables based on the detected device.

    For CUDA: Sets CUDA_LAUNCH_BLOCKING and TORCH_USE_CUDA_DSA
    For MPS: Sets PYTORCH_MPS_HIGH_WATERMARK_RATIO for memory management
    """
    device_type = get_device_type()

    if device_type == "cuda":
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
        os.environ['TORCH_USE_CUDA_DSA'] = "1"
    elif device_type == "mps":
        # Allow MPS to use more memory (default is 0.0 which is unlimited)
        # Set to 0.0 to allow full memory usage
        os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')


def to_device(tensor_or_model, device: Optional[torch.device] = None):
    """
    Move a tensor or model to the appropriate device.

    Args:
        tensor_or_model: A torch.Tensor or nn.Module to move
        device: Optional specific device, defaults to auto-detected device

    Returns:
        The tensor or model on the specified device
    """
    if device is None:
        device = get_device()

    return tensor_or_model.to(device)


def get_generator(device: Optional[torch.device] = None) -> torch.Generator:
    """
    Create a random number generator for the appropriate device.

    Args:
        device: Optional specific device, defaults to auto-detected device

    Returns:
        torch.Generator configured for the device
    """
    if device is None:
        device = get_device()

    return torch.Generator(device=device)


def is_distributed_available() -> bool:
    """
    Check if distributed training is available on the current device.

    Returns:
        True if distributed training can be used, False otherwise
    """
    device_type = get_device_type()

    if device_type == "cuda":
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    elif device_type == "mps":
        # MPS doesn't support multi-GPU distributed training
        # but single-device "distributed" with gloo backend can work
        return True
    else:
        # CPU can use gloo backend
        return True


def print_device_info():
    """Print information about the detected device."""
    device_type = get_device_type()
    device_name = get_device_name()

    print(f"Device type: {device_type}")
    print(f"Device name: {device_name}")

    if device_type == "cuda":
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    elif device_type == "mps":
        print("MPS (Metal Performance Shaders) is available")
        print("Note: Some operations may fall back to CPU")
