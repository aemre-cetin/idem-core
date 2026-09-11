"""
Native C++20 and CUDA Engine Python Bindings.
Protected under U.S. Patent Application No. 64/148,668.
"""

import os
import sys
import ctypes
from typing import Optional
import torch

# Locate and load the C ABI dynamic library across Windows, Linux, and macOS
_curr_dir = os.path.dirname(os.path.abspath(__file__))
_candidate_names = []
if sys.platform == "win32":
    _candidate_names = ["idempotent.dll"]
elif sys.platform == "darwin":
    _candidate_names = ["libidempotent.dylib", "idempotent.dylib", "libidempotent.so"]
else:
    _candidate_names = ["libidempotent.so", "idempotent.so"]

# Search order: current package dir, then cpp build directories
_candidate_paths = []
for name in _candidate_names:
    _candidate_paths.append(os.path.join(_curr_dir, name))
    _candidate_paths.append(os.path.abspath(os.path.join(_curr_dir, "../../../../cpp", name)))
    _candidate_paths.append(os.path.abspath(os.path.join(_curr_dir, "../../../../cpp/build", name)))

_lib = None
_dll_name = _candidate_names[0]
for _p in _candidate_paths:
    if os.path.exists(_p):
        try:
            _lib = ctypes.CDLL(_p)
            _dll_name = os.path.basename(_p)
            break
        except Exception as e:
            continue

if _lib is not None:
    try:
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

        if hasattr(_lib, "idempotent_cuda_compact_f32"):
            _lib.idempotent_cuda_compact_f32.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_void_p
            ]
            _lib.idempotent_cuda_compact_f32.restype = None

        if hasattr(_lib, "idempotent_cuda_compact_f16"):
            _lib.idempotent_cuda_compact_f16.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_void_p
            ]
            _lib.idempotent_cuda_compact_f16.restype = None

        _lib.idempotent_generate_map.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_size_t, ctypes.c_size_t
        ]
        _lib.idempotent_generate_map.restype = None

        _lib.idempotent_version.argtypes = []
        _lib.idempotent_version.restype = ctypes.c_char_p
    except Exception as e:
        _lib = None
        print(f"[Warning] Failed to configure {_dll_name}: {e}")

def get_native_version() -> str:
    if _lib and hasattr(_lib, "idempotent_version"):
        return _lib.idempotent_version().decode("utf-8")
    return "1.1.1-fallback"

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
    Dispatches to Native CUDA Engine on GPU (NVIDIA Blackwell sm_120), Native C++20 Engine on CPU,
    or zero-allocation in-situ cycle transposition fallback across Windows, Linux, and macOS.
    """
    assert tensor.is_contiguous(), "Tensor must be contiguous"
    assert target_map.is_contiguous(), "TargetMap must be contiguous"

    # Shape inference supporting 1D, 2D, and 3D+ tensors
    if tensor.dim() == 1:
        B, N, D = 1, tensor.shape[0], 1
        b_map = target_map.view(1, N)
    elif tensor.dim() == 2:
        if target_map.dim() == 2 and target_map.shape == tensor.shape:
            B, N, D = tensor.shape[0], tensor.shape[1], 1
            b_map = target_map
        elif target_map.dim() == 1 and target_map.shape[0] == tensor.shape[0]:
            B, N, D = 1, tensor.shape[0], tensor.shape[1]
            b_map = target_map.view(1, N)
        elif target_map.dim() == 2 and target_map.shape[0] == 1 and target_map.shape[1] == tensor.shape[0]:
            B, N, D = 1, tensor.shape[0], tensor.shape[1]
            b_map = target_map
        else:
            N = target_map.shape[-1]
            B = target_map.numel() // N
            D = tensor.numel() // (B * N)
            b_map = target_map.view(B, N)
    else:
        B = tensor.shape[0]
        N = tensor.shape[1]
        D = tensor.shape[2] if tensor.dim() == 3 else tensor[0, 0].numel()
        b_map = target_map.view(B, N)

    b_map = b_map.to(device=tensor.device, dtype=torch.int32).contiguous()

    # GPU Execution Path
    if tensor.is_cuda:
        # Direct dispatch to Native CUDA Blackwell sm_120 Kernel
        if _lib is not None and hasattr(_lib, "idempotent_cuda_compact_f32"):
            stream = torch.cuda.current_stream().cuda_stream
            if tensor.dtype == torch.float32:
                _lib.idempotent_cuda_compact_f32(tensor.data_ptr(), b_map.data_ptr(), B, N, D, stream)
                return tensor
            elif tensor.dtype in (torch.float16, torch.bfloat16):
                _lib.idempotent_cuda_compact_f16(tensor.data_ptr(), b_map.data_ptr(), B, N, D, stream)
                return tensor

        # In-situ transposition fallback on GPU
        t_view = tensor.view(B, N, D)
        for b in range(B):
            for i in range(N):
                dest = b_map[b, i].item()
                if dest > i and b_map[b, dest].item() == i:
                    tmp = t_view[b, i].clone()
                    t_view[b, i] = t_view[b, dest]
                    t_view[b, dest] = tmp
        return tensor

    # CPU Execution Path: Native C++20 DLL/SO if available
    if _lib is not None and hasattr(_lib, "idempotent_compact_f32"):
        try:
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
        except Exception:
            pass  # Fall through to in-situ Python fallback

    # Pure In-Situ Zero-Allocation CPU / MPS Fallback (Linux / macOS / Non-native)
    t_view = tensor.view(B, N, D)
    for b in range(B):
        visited = set()
        for i in range(N):
            dest = int(b_map[b, i].item())
            if dest != i and i not in visited:
                # 2-cycle involution fast-path
                if int(b_map[b, dest].item()) == i:
                    if dest > i:
                        tmp = t_view[b, i].clone()
                        t_view[b, i] = t_view[b, dest]
                        t_view[b, dest] = tmp
                        visited.add(i)
                        visited.add(dest)
                else:
                    # General cyclic permutation traversal
                    orbit = [i]
                    curr = dest
                    while curr != i and curr not in visited:
                        orbit.append(curr)
                        curr = int(b_map[b, curr].item())
                    for idx in orbit:
                        visited.add(idx)
                    if len(orbit) > 1 and curr == i:
                        tmp = t_view[b, orbit[0]].clone()
                        for k in range(len(orbit) - 1):
                            t_view[b, orbit[k]] = t_view[b, orbit[k + 1]]
                        t_view[b, orbit[-1]] = tmp

    return tensor