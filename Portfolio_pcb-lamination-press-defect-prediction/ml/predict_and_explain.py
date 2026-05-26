r"""Predict and generate explainability artifacts for PressFuse model.

CLI example (PowerShell):
python ml\predict_and_explain.py --checkpoint outputs\sample_run\model_mvp.ckpt --output-dir outputs\sample_run --n-cycles 512 --n-points 192 --batch-size 16 --shap --shap-method shap --shap-subsample 32 --save-attn --seed 42

This script:
 - loads dataset (SyntheticPressDataset)
 - loads model and checkpoint
 - runs predictions and collects attention tensors
 - computes evaluation metrics and saves metrics.json
 - saves predictions.csv and params.json
 - optionally calls SHAP wrapper and attention saver to create shap_values.npy/shap_summary.csv and attention.npy

Acceptance criteria (at end of file)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from src.data.dataset import SyntheticPressDataset, collate_fn
from src.models.pressfuse import CrossModalAttentionConfig, PressFuseModel

# shap wrapper and attention saver (optional)
try:
    from src.explain.shap_wrapper import compute_and_save_shap  # type: ignore
except Exception:  # pragma: no cover - optional import
    compute_and_save_shap = None  # type: ignore

try:
    from src.explain.attention_saver import save_attention_batch  # type: ignore
except Exception:  # pragma: no cover - optional import
    save_attention_batch = None  # type: ignore


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _git_sha_short() -> Optional[str]:
    try:
        sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('utf-8').strip()
        return sha
    except Exception:
        return None


def predict_and_explain(
    checkpoint: str,
    output_dir: str,
    n_cycles: int = 512,
    n_points: int = 192,
    batch_size: int = 16,
    shap: bool = False,
    shap_method: str = 'shap',
    shap_subsample: int = 32,
    save_attn: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    _ensure_dir(output_dir)

    # dataset
    dataset = SyntheticPressDataset(n_cycles=n_cycles, n_points=n_points, anomaly_prob=0.15, seed=seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # model
    # infer dims
    sample_x, _, _ = dataset[0]
    T, D = sample_x.shape

    # load checkpoint state first to infer model dims like d_model
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')

    # preliminary device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        state = torch.load(checkpoint, map_location=device)
    except Exception as e:
        print(f'Failed to load checkpoint file: {e}', file=sys.stderr)
        raise

    # infer d_model from state dict if possible
    inferred_d_model: Optional[int] = None
    for key in ['ts_encoder.0.weight', 'event_encoder.0.weight']:
        if key in state:
            inferred_d_model = int(state[key].shape[0])
            break
    if inferred_d_model is None:
        # fallback to default
        inferred_d_model = CrossModalAttentionConfig().d_model

    cfg = CrossModalAttentionConfig(ts_input_dim=D, d_model=inferred_d_model)
    model = PressFuseModel(config=cfg)
    model.to(device)

    # load into model
    try:
        model.load_state_dict(state)
    except Exception as e:
        # rethrow with context
        print(f'Failed to load checkpoint into model with inferred d_model={inferred_d_model}: {e}', file=sys.stderr)
        raise

    model.eval()

    ids_all: List[int] = []
    ys_true: List[int] = []
    ys_prob: List[float] = []
    attn_list: List[np.ndarray] = []

    try:
        with torch.no_grad():
            for X, y, ids in loader:
                X = X.to(device)
                out = model(X, events=X)
                logits = out.get('binary_logits') if isinstance(out, dict) else out
                probs = torch.sigmoid(logits).detach().cpu().numpy().ravel().tolist()
                ys_prob.extend(probs)
                ys_true.extend([int(v) for v in y.cpu().numpy().ravel().tolist()])
                ids_all.extend(ids)

                attn = None
                if isinstance(out, dict) and 'attention' in out:
                    attn = out['attention']
                if attn is not None:
                    # convert to numpy and append
                    if isinstance(attn, torch.Tensor):
                        attn_np = attn.detach().cpu().numpy()
                    else:
                        attn_np = np.asarray(attn)
                    attn_list.append(attn_np)
    except Exception as e:
        print(f'Error during prediction loop: {e}', file=sys.stderr)
        raise

    # threshold selection: fixed and ROC-based optimal (Youden's J)
    threshold_fixed = 0.5
    threshold_opt = threshold_fixed
    threshold_type = "fixed"
    try:
        if len(set(ys_true)) > 1:
            fpr, tpr, thresholds = roc_curve(ys_true, ys_prob)
            j_scores = tpr - fpr
            best_idx = int(np.argmax(j_scores))
            threshold_opt = float(thresholds[best_idx])
            threshold_type = "roc_youden"
    except Exception as e:
        print(f"Failed to compute optimal threshold from ROC: {e}", file=sys.stderr)
        threshold_opt = threshold_fixed
        threshold_type = "fixed_fallback"

    pred_labels_fixed = [1 if p >= threshold_fixed else 0 for p in ys_prob]
    pred_labels_opt = [1 if p >= threshold_opt else 0 for p in ys_prob]

    # save predictions.csv
    preds_path = os.path.join(output_dir, 'predictions.csv')
    try:
        with open(preds_path, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            writer.writerow(['cycle_id', 'label', 'pred_proba', 'pred_label', 'pred_label_opt'])
            for cid, ytrue, p, pl_fixed, pl_opt in zip(ids_all, ys_true, ys_prob, pred_labels_fixed, pred_labels_opt):
                writer.writerow([int(cid), int(ytrue), float(p), int(pl_fixed), int(pl_opt)])
    except Exception as e:
        print(f'Failed to write predictions.csv: {e}', file=sys.stderr)
        raise

    # compute metrics
    metrics: Dict[str, float] = {}
    try:
        metrics['accuracy'] = float(accuracy_score(ys_true, pred_labels_fixed))
        metrics['precision'] = float(precision_score(ys_true, pred_labels_fixed, zero_division=0))
        metrics['recall'] = float(recall_score(ys_true, pred_labels_fixed, zero_division=0))
        metrics['f1'] = float(f1_score(ys_true, pred_labels_fixed, zero_division=0))
        metrics['roc_auc'] = float(roc_auc_score(ys_true, ys_prob)) if len(set(ys_true)) > 1 else 0.0
        metrics['optimal_threshold'] = float(threshold_opt)
        metrics['threshold_type'] = threshold_type
    except Exception as e:
        print(f'Failed to compute metrics: {e}', file=sys.stderr)
        metrics = {k: 0.0 for k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']}
        metrics['optimal_threshold'] = float(threshold_fixed)
        metrics['threshold_type'] = 'fixed_fallback'

    # save metrics
    metrics_path = os.path.join(output_dir, 'metrics.json')
    try:
        with open(metrics_path, 'w', encoding='utf-8') as fh:
            json.dump(metrics, fh, indent=2)
    except Exception as e:
        print(f'Failed to write metrics.json: {e}', file=sys.stderr)
        raise

    # params.json
    params: Dict[str, Any] = {
        'checkpoint': checkpoint,
        'n_cycles': n_cycles,
        'n_points': n_points,
        'batch_size': batch_size,
        'shap': bool(shap),
        'shap_method': shap_method,
        'shap_subsample': shap_subsample,
        'save_attn': bool(save_attn),
        'seed': seed,
        'timestamp': datetime.now().astimezone().isoformat(),
        'git_sha': _git_sha_short(),
        'model_name': model.__class__.__name__,
        'data_summary': {'n_samples': len(dataset), 'feature_dim': D, 'time_steps': T},
    }
    params_path = os.path.join(output_dir, 'params.json')
    try:
        with open(params_path, 'w', encoding='utf-8') as fh:
            json.dump(params, fh, indent=2)
    except Exception as e:
        print(f'Failed to write params.json: {e}', file=sys.stderr)
        raise

    # attention saving
    attn_result: Optional[Dict[str, Any]] = None
    if save_attn and len(attn_list) > 0:
        if save_attention_batch is None:
            print('attention_saver not available; install or add src/explain/attention_saver.py', file=sys.stderr)
        else:
            try:
                # concatenate along batch dim
                all_attn = np.concatenate(attn_list, axis=0)
                attn_result = save_attention_batch(all_attn, output_dir, max_samples=16, average_heads=False)
            except Exception as e:
                print(f'Failed to save attention: {e}', file=sys.stderr)

    # shap
    shap_result: Optional[Dict[str, Any]] = None
    if shap:
        if compute_and_save_shap is None:
            print('shap_wrapper not available; install shap or add src/explain/shap_wrapper.py', file=sys.stderr)
        else:
            try:
                # For SHAP compute, pass model and dataloader (use same loader but note subsample)
                shap_result = compute_and_save_shap(model, loader, output_dir, method=shap_method, subsample=shap_subsample, baseline='mean', device=str(device))
            except ModuleNotFoundError as e:
                print(f'SHAP dependency missing: {e}', file=sys.stderr)
            except Exception as e:
                print(f'Failed to compute SHAP: {e}', file=sys.stderr)

    result = {
        'predictions': preds_path,
        'metrics': metrics_path,
        'params': params_path,
        'attention': attn_result,
        'shap': shap_result,
    }

    return result


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='outputs/sample_run')
    parser.add_argument('--n-cycles', type=int, default=512)
    parser.add_argument('--n-points', type=int, default=192)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--shap', action='store_true')
    parser.add_argument('--shap-method', type=str, default='shap', choices=['shap', 'captum'])
    parser.add_argument('--shap-subsample', type=int, default=32)
    parser.add_argument('--save-attn', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()
    try:
        res = predict_and_explain(
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            n_cycles=args.n_cycles,
            n_points=args.n_points,
            batch_size=args.batch_size,
            shap=args.shap,
            shap_method=args.shap_method,
            shap_subsample=args.shap_subsample,
            save_attn=args.save_attn,
            seed=args.seed,
        )
        print('Done. Results:')
        print(json.dumps(res, indent=2))
    except Exception as exc:
        print(f'Error during predict_and_explain: {exc}', file=sys.stderr)
        sys.exit(2)


# Acceptance criteria (human):
# 1) Running this script with a valid checkpoint creates predictions.csv, metrics.json, params.json in output-dir.
# 2) If --save-attn is specified and model returns attention, attention.npy and attention_meta.json are created (via attention_saver).
# 3) If --shap is specified and shap_wrapper is available, shap_values.npy and shap_summary.csv are created.
# 4) The predictions.csv has columns: cycle_id,label,pred_proba,pred_label
# 5) metrics.json contains accuracy, precision, recall, f1, roc_auc

