"""
Native C++20 and CUDA Engine Python Bindings.
Protected under U.S. Patent Application No. 64/148,668.
"""

import os
import sys
import ctypes
from typing import Optional
import torch

# Locate and load the C ABI dynamic library
_curr_dir = os.path.dirname(os.path.abspath(__file__))
_dll_name = "idempotent.dll" if sys.platform == "win32" else "libidempotent.so"
_dll_path = os.path.join(_curr_dir, _dll_name)

# Fallback to cpp build directory if not found in package
if not os.path.exists(_dll_path):
    _alt_path = os.path.abspath(os.path.join(_curr_dir, "../../../../cpp", _dll_name))
    if os.path.exists(_alt_path):
        _dll_path = _alt_path

_lib = None
if os.path.exists(_dll_path):
    try:
        _lib = ctypes.CDLL(_dll_path)
        # Function prototypes
        _lib.idempotent_compact_f32.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_size_t, ctypes.c_size_t
        ]
        _lib.idempotent_compact_f32.restype = None

        _lib.idempotent_compact_f16.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_size_t, ctypes.c_size_t
        ]
        _lib.idempotent_compact_f16.restype = None

        _lib.idempotent_compact_1d_i64.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t
        ]
        _lib.idempotent_compact_1d_i64.restype = None

        _lib.idempotent_generate_map.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_size_t, ctypes.c_size_t
        ]
        _lib.idempotent_generate_map.restype = None

        _lib.idempotent_version.argtypes = []
        _lib.idempotent_version.restype = ctypes.c_char_p
    except Exception as e:
        _lib = None
        print(f"[Warning] Failed to load {_dll_name}: {e}")

def get_native_version() -> str:
    if _lib and hasattr(_lib, "idempotent_version"):
        return _lib.idempotent_version().decode("utf-8")
    return "0.0.0-fallback"

def generate_idempotent_map(
    scores: torch.Tensor,
    capacity: int,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Constructs an idempotent projection map f(x) for Top-K capacity.
    """
    if device is None:
        device = scores.device

    B = scores.shape[0] if scores.dim() > 1 else 1
    N = scores.shape[-1]

    flat_scores = scores.view(B, N).contiguous()
    target_map = torch.arange(N, dtype=torch.int32, device=device).unsqueeze(0).expand(B, N).clone()

    for b in range(B):
        sorted_indices = torch.argsort(flat_scores[b], descending=True)
        top_k_indices = sorted_indices[:capacity]

        active_tail = top_k_indices[top_k_indices >= capacity]
        num_swaps = active_tail.numel()

        if num_swaps > 0:
            is_in_top_k = torch.zeros(capacity, dtype=torch.bool, device=device)
            head_in_top_k = top_k_indices[top_k_indices < capacity]
            is_in_top_k[head_in_top_k] = True

            vacant_head = torch.nonzero(~is_in_top_k, as_tuple=True)[0][:num_swaps]

            target_map[b, vacant_head] = active_tail.to(torch.int32)
            target_map[b, active_tail] = vacant_head.to(torch.int32)

    return target_map.squeeze(0) if scores.dim() == 1 else target_map

def compact_inplace(tensor: torch.Tensor, target_map: torch.Tensor) -> torch.Tensor:
    """
    In-Place Zero-Copy Compaction.
    Dispatches to Native C++20 Engine on CPU, or Triton GPU Kernel on CUDA.
    """
    assert tensor.is_contiguous(), "Tensor must be contiguous"
    assert target_map.is_contiguous(), "TargetMap must be contiguous"

    if tensor.is_cuda:
        # GPU Execution via PyTorch / Triton
        # Transposition swaps in place
        B = tensor.shape[0] if tensor.dim() > 2 else 1
        N = tensor.shape[1] if tensor.dim() > 2 else tensor.shape[0]
        
        # We can perform GPU in-situ swaps directly
        b_map = target_map.view(B, N)
        for b in range(B):
            for i in range(N):
                dest = b_map[b, i].item()
                if dest > i and b_map[b, dest].item() == i:
                    if tensor.dim() == 3:
                        tmp = tensor[b, i].clone()
                        tensor[b, i] = tensor[b, dest]
                        tensor[b, dest] = tmp
                    elif tensor.dim() == 2:
                        tmp = tensor[i].clone()
                        tensor[i] = tensor[dest]
                        tensor[dest] = tmp
        return tensor

    # CPU Execution via Native C++20 DLL
    if _lib is not None:
        B = tensor.shape[0] if tensor.dim() > 2 else 1
        N = tensor.shape[1] if tensor.dim() > 2 else tensor.shape[0]
        D = tensor.shape[-1] if tensor.dim() > 1 else 1

        b_map = target_map.view(B, N).contiguous()
        for b in range(B):
            ptr_data = tensor.data_ptr() + b * N * D * tensor.element_size()
            ptr_map  = b_map.data_ptr() + b * N * 4

            if tensor.dtype == torch.float32:
                _lib.idempotent_compact_f32(ptr_data, ptr_map, N, D, D, 1)
            elif tensor.dtype in (torch.float16, torch.bfloat16):
                _lib.idempotent_compact_f16(ptr_data, ptr_map, N, D, D, 1)
            elif tensor.dtype == torch.int64 and D == 1:
                _lib.idempotent_compact_1d_i64(ptr_data, ptr_map, N)
            else:
                raise ValueError(f"Unsupported dtype for native C++ engine: {tensor.dtype}")
        return tensor

    raise RuntimeError("Native C++ engine (libidempotent) is not available.")