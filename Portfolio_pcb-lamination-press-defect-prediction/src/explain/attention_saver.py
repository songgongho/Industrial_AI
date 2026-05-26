"""Utilities to save attention tensors for Streamlit Explain tab.

Saves:
 - outputs/sample_run/attention.npy (numpy array)
 - outputs/sample_run/attention_meta.json (metadata)

Function:
 - save_attention_batch(attention_tensor, output_dir, max_samples=16, average_heads=False)

Supports torch.Tensor or numpy.ndarray inputs with shapes:
 - (B, H, T, T)
 - (H, T, T)
 - (T, T) (will be promoted to (1, H=1, T, T))

"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

import numpy as np
import torch


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _to_numpy(t: Any) -> np.ndarray:
    """Convert torch.Tensor or numpy array to numpy ndarray on CPU."""
    if isinstance(t, np.ndarray):
        return t
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    raise TypeError("attention_tensor must be a torch.Tensor or numpy.ndarray")


def save_attention_batch(
    attention_tensor: Any,
    output_dir: str,
    max_samples: int = 16,
    average_heads: bool = False,
) -> Dict[str, Any]:
    """Save attention tensor(s) to output_dir.

    Parameters
    ----------
    attention_tensor : torch.Tensor | np.ndarray
        Attention tensor with shape (B, H, T, T) or (H, T, T) or (T, T).
    output_dir : str
        Directory to write outputs (created if missing).
    max_samples : int
        Maximum number of samples to keep (if B > max_samples will truncate to first samples).
    average_heads : bool
        If True, average across heads producing shape (B, 1, T, T) or (T, T) accordingly.

    Returns
    -------
    dict
        {'attention_path': ..., 'meta_path': ..., 'saved_shape': tuple}
    """
    _ensure_dir(output_dir)

    arr = _to_numpy(attention_tensor)

    # Normalize dims to (B, H, T, T)
    if arr.ndim == 4:
        B, H, T1, T2 = arr.shape
    elif arr.ndim == 3:
        # (H, T, T) -> treat as single sample
        arr = arr[np.newaxis, ...]
        B, H, T1, T2 = arr.shape
    elif arr.ndim == 2:
        # (T, T) -> treat as single sample, single head
        arr = arr[np.newaxis, np.newaxis, ...]
        B, H, T1, T2 = arr.shape
    else:
        raise ValueError(f"Unsupported attention tensor ndim={arr.ndim}; expected 2,3,or4")

    # Truncate samples if needed
    if B > max_samples:
        arr = arr[:max_samples]
        B = arr.shape[0]

    # Optionally average heads
    meta: Dict[str, Any] = {}
    if average_heads:
        # compute mean across head axis
        arr = np.mean(arr, axis=1, keepdims=True)  # shape -> (B,1,T,T)
        H_saved = 1
    else:
        H_saved = H

    saved_shape = arr.shape  # (B, H_saved, T, T)

    attention_path = os.path.join(output_dir, "attention.npy")
    np.save(attention_path, arr.astype(np.float32))

    meta_path = os.path.join(output_dir, "attention_meta.json")
    meta = {
        "attention_path": os.path.basename(attention_path),
        "shape": saved_shape,
        "dtype": str(arr.dtype),
        "max_samples_used": int(min(max_samples, B)),
        "average_heads": bool(average_heads),
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    return {"attention_path": attention_path, "meta_path": meta_path, "saved_shape": saved_shape}


# Acceptance criteria (human-readable):
# 1) Calling save_attention_batch with a torch.Tensor of shape (B,H,T,T) writes attention.npy and attention_meta.json
# 2) If B > max_samples, only the first max_samples are saved
# 3) If average_heads=True, saved array has head dimension =1 and meta records average_heads=True
# 4) Function accepts numpy arrays as well as torch tensors
# 5) The returned dict contains correct file paths and saved_shape

