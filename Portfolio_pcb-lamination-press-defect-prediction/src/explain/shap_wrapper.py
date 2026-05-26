"""SHAP wrapper utilities for PressFuse model.

compute_and_save_shap(model, dataloader, output_dir, method='shap', subsample=64, baseline='mean', device='cpu')

Saves:
 - outputs/sample_run/shap_values.npy  (numpy array: (N, T, D) or (T, D) or (D,))
 - outputs/sample_run/shap_summary.csv (CSV with columns feature,importance)

This module supports shap.GradientExplainer (preferred) and Captum (alternative).
"""
from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import pandas as pd


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _pip_install_hint(pkg: str) -> str:
    return f"pip install {pkg}"


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _sample_from_dataloader(dataloader: Iterable, subsample: int) -> Tuple[np.ndarray, List[Any]]:
    """Collect up to `subsample` samples from dataloader and return numpy array (N,T,D) and ids list.

    Assumes dataloader yields tuples like (X, y, ids) where X is torch.Tensor (B,T,D) or numpy.
    """
    X_list: List[np.ndarray] = []
    ids_list: List[Any] = []
    collected = 0
    for batch in dataloader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 1:
            Xb = batch[0]
            batch_ids = batch[2] if len(batch) > 2 else [None] * (len(Xb) if hasattr(Xb, '__len__') else 1)
        else:
            Xb = batch
            batch_ids = [None] * (len(Xb) if hasattr(Xb, '__len__') else 1)

        # convert to numpy
        if isinstance(Xb, torch.Tensor):
            Xb_np = Xb.detach().cpu().numpy()
        else:
            Xb_np = np.asarray(Xb)

        B = Xb_np.shape[0]
        for i in range(B):
            X_list.append(Xb_np[i])
            ids_list.append(batch_ids[i] if i < len(batch_ids) else None)
            collected += 1
            if collected >= subsample:
                break
        if collected >= subsample:
            break
    if len(X_list) == 0:
        raise ValueError("Dataloader yielded no samples for SHAP computation")
    arr = np.stack(X_list, axis=0)
    return arr, ids_list


def compute_and_save_shap(
    model: torch.nn.Module,
    dataloader: Iterable,
    output_dir: str,
    method: str = "shap",
    subsample: int = 64,
    baseline: str = "mean",
    device: str = "cpu",
) -> Dict[str, Any]:
    """Compute SHAP values for a PyTorch model and save numpy + summary CSV.

    Parameters
    ----------
    model : torch.nn.Module
        Model that accepts input X shape (B, T, D) and returns dict with key 'binary_logits' or single tensor.
    dataloader : Iterable
        DataLoader yielding (X, y, ids) batches.
    output_dir : str
        Directory to write outputs (will be created if missing).
    method : str
        'shap' to use shap.GradientExplainer (preferred) or 'captum' to use Captum's GradientShap/IntegratedGradients.
    subsample : int
        Max number of samples to compute SHAP for (to limit compute/memory).
    baseline : str
        'mean' or 'zeros' or 'sample' - used to construct baseline for SHAP/Captum.
    device : str
        'cpu' or 'cuda' device string

    Returns
    -------
    dict
        {'shap_path':..., 'summary_csv':..., 'shape': shap_values.shape}
    """
    _ensure_dir(output_dir)

    # prepare device and model
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model.to(dev)
    model.eval()

    # sample limited inputs from dataloader
    X_samples_np, ids = _sample_from_dataloader(dataloader, subsample=subsample)
    # X_samples_np shape: (N, T, D)
    if X_samples_np.ndim != 3:
        raise ValueError(f"Expected sampled data of ndim==3 (N,T,D), got shape {X_samples_np.shape}")
    N, T, D = X_samples_np.shape

    # feature names
    feature_names = [f"f{i}" for i in range(1, D + 1)]

    shap_values: Optional[np.ndarray] = None

    if method == "shap":
        try:
            import shap

            # prepare model prediction function for shap that accepts numpy array
            def model_predict_np(x: np.ndarray) -> np.ndarray:
                # x: (B, T, D) numpy
                with torch.no_grad():
                    xt = torch.from_numpy(x.astype(np.float32)).to(dev)
                    # model may accept events param; pass xt for both
                    out = model(xt, events=xt)
                    # obtain logits or single output
                    if isinstance(out, dict) and "binary_logits" in out:
                        logits = out["binary_logits"]
                    elif isinstance(out, torch.Tensor):
                        logits = out
                    else:
                        # try first tensor-like value
                        vals = None
                        if isinstance(out, dict):
                            for v in out.values():
                                if isinstance(v, torch.Tensor):
                                    vals = v
                                    break
                        if vals is None:
                            raise RuntimeError("Model output not recognized for SHAP wrapper")
                        logits = vals
                    # ensure shape (B,)
                    logits = logits.detach().cpu().numpy()
                    if logits.ndim > 1 and logits.shape[-1] == 1:
                        logits = logits.reshape(-1)
                    return logits

            # background/baseline: use small set from samples
            bg = X_samples_np[: min(10, N)].astype(np.float32)
            # Try GradientExplainer first (may require tensorflow). If unavailable/fails,
            # fall back to shap.Explainer which is model-agnostic and does not require TF.
            try:
                explainer = shap.GradientExplainer(model_predict_np, bg)
                shap_res = explainer.shap_values(X_samples_np.astype(np.float32))
                if isinstance(shap_res, list):
                    shap_values = np.array(shap_res[0])
                else:
                    shap_values = np.array(shap_res)
            except ModuleNotFoundError:
                # GradientExplainer attempted to import tensorflow but it's not present.
                # Use shap.Explainer as a fallback (may be slower but avoids TF dependency).
                # shap.Explainer expects a "masker" object rather than a raw ndarray in
                # newer versions. Try to construct an Independent masker from the bg.
                try:
                    masker = getattr(shap, 'maskers', None)
                    if masker is not None and hasattr(masker, 'Independent'):
                        masker_obj = masker.Independent(bg)
                    else:
                        # older shap versions accept a numpy background directly
                        masker_obj = bg
                except Exception:
                    masker_obj = bg

                expl = shap.Explainer(model_predict_np, masker_obj)
                out = expl(X_samples_np.astype(np.float32))
                # expl(...) returns an Explanation object; .values contains shap array
                shap_values = np.array(out.values)
        except Exception as e:
            # provide install hint if shap missing
            if isinstance(e, ModuleNotFoundError):
                raise ModuleNotFoundError(
                    f"shap is required for method='shap'. Install with: {_pip_install_hint('shap')}"
                )
            else:
                raise

    elif method == "captum":
        try:
            from captum.attr import GradientShap

            # prepare baselines per Captum: baseline tensor shape (B, T, D) or (samples, T, D)
            xt = torch.from_numpy(X_samples_np.astype(np.float32)).to(dev)
            if baseline == "mean":
                baseline_tensor = torch.mean(xt, dim=0, keepdim=True)  # (1, T, D)
            elif baseline == "zeros":
                baseline_tensor = torch.zeros_like(xt[:1])
            else:
                baseline_tensor = xt[: min(10, N)]

            gs = GradientShap(model)
            # Captum expects function returning scalar logits; define wrapper
            def _model_forward(inp: torch.Tensor) -> torch.Tensor:
                out = model(inp, events=inp)
                if isinstance(out, dict) and "binary_logits" in out:
                    logits = out["binary_logits"]
                elif isinstance(out, torch.Tensor):
                    logits = out
                else:
                    # pick first tensor-like
                    vals = None
                    if isinstance(out, dict):
                        for v in out.values():
                            if isinstance(v, torch.Tensor):
                                vals = v
                                break
                    if vals is None:
                        raise RuntimeError("Model output not recognized for Captum wrapper")
                    logits = vals
                # ensure shape (B,)
                if logits.dim() > 1 and logits.size(-1) == 1:
                    logits = logits.view(-1)
                return logits

            attributions = []
            # compute attributions in batches to save memory
            batch_size = max(1, min(8, N))
            for i in range(0, N, batch_size):
                batch_x = xt[i : i + batch_size]
                # baselines for GradientShap should be (n_baselines, T, D)
                attr = gs.attribute(batch_x, baselines=baseline_tensor, n_samples=50)
                attributions.append(attr.detach().cpu().numpy())
            shap_values = np.concatenate(attributions, axis=0)
        except Exception as e:
            if isinstance(e, ModuleNotFoundError):
                raise ModuleNotFoundError(
                    f"captum is required for method='captum'. Install with: {_pip_install_hint('captum')}"
                )
            else:
                raise
    else:
        raise ValueError("method must be either 'shap' or 'captum'")

    # post-process shap_values: ensure shape is (N, T, D) or reduce appropriately
    if shap_values is None:
        raise RuntimeError("SHAP computation failed; no values produced")

    # convert to float32
    shap_values = shap_values.astype(np.float32)

    # If shap_values has shape (T,D) or (D,), promote to (N, T, D) if needed
    if shap_values.ndim == 2:
        # (T, D) -> single sample
        shap_values = shap_values[np.newaxis, ...]
    elif shap_values.ndim == 1:
        # (D,) -> expand to (1,1,D)
        shap_values = shap_values.reshape(1, 1, -1)

    # Save full array
    shap_path = os.path.join(output_dir, "shap_values.npy")
    np.save(shap_path, shap_values)

    # compute summary importance: mean absolute over samples and time
    importance = np.mean(np.abs(shap_values), axis=(0, 1))  # (D,)
    # create dataframe
    df_summary = pd.DataFrame({"feature": feature_names, "importance": importance})
    summary_csv = os.path.join(output_dir, "shap_summary.csv")
    df_summary.sort_values("importance", ascending=False).to_csv(summary_csv, index=False)

    # optional metadata
    meta = {
        "method": method,
        "shape": shap_values.shape,
        "n_samples": int(N),
        "feature_count": int(D),
    }
    with open(os.path.join(output_dir, "shap_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    return {"shap_path": shap_path, "summary_csv": summary_csv, "shape": shap_values.shape, "meta": meta}


# Acceptance criteria (for human review):
# 1) When provided a working PyTorch model and a dataloader with at least one batch, calling compute_and_save_shap
#    produces outputs/sample_run/shap_values.npy and outputs/sample_run/shap_summary.csv.
# 2) shap_values.npy is a floating-point numpy array with shape (N, T, D) or (1, T, D) etc.
# 3) shap_summary.csv contains columns 'feature' and 'importance' with D rows and is readable by pandas.
# 4) If shap is not installed and method='shap' is requested, the function raises ModuleNotFoundError with install hint.
# 5) If captum is not installed and method='captum' is requested, the function raises ModuleNotFoundError with install hint.


