from __future__ import annotations

import argparse
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.data.dataset import SyntheticPressDataset, collate_fn
from src.models.pressfuse import PressFuseModel, CrossModalAttentionConfig


def apply_peak_drift_step_mix(
    X: torch.Tensor,
    y: torch.Tensor,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Apply simple anomaly mix (peak/drift/step) on positive samples only.

    Parameters
    ----------
    X : torch.Tensor
        Input tensor of shape (B, T, D)
    y : torch.Tensor
        Label tensor of shape (B, 1)
    rng : np.random.Generator
        RNG for deterministic-ish perturbation control
    """
    if X.ndim != 3:
        return X
    X_mod = X.clone()
    B, T, D = X_mod.shape
    pos_idx = (y.view(-1) > 0.5).nonzero(as_tuple=False).view(-1)
    if len(pos_idx) == 0:
        return X_mod

    for bi in pos_idx.tolist():
        mode = int(rng.integers(0, 3))  # 0:peak, 1:drift, 2:step
        feat = int(rng.integers(0, D))
        if mode == 0:
            center = int(rng.integers(max(2, T // 6), max(3, T - T // 6)))
            width = max(2, T // 20)
            left = max(0, center - width)
            right = min(T, center + width)
            X_mod[bi, left:right, feat] += 0.25
        elif mode == 1:
            drift = torch.linspace(0.0, 0.3, T, device=X_mod.device)
            X_mod[bi, :, feat] += drift
        else:
            step_t = int(rng.integers(max(1, T // 4), max(2, (3 * T) // 4)))
            X_mod[bi, step_t:, feat] += 0.2
    return X_mod


def focal_loss_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Binary focal loss based on BCEWithLogits(reduction='none')."""
    bce_loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight,
        reduction="none",
    )
    p_t = torch.exp(-bce_loss)
    focal = ((1.0 - p_t) ** gamma) * bce_loss
    return focal.mean()


def train(args: argparse.Namespace) -> None:
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # dataset
    dataset = SyntheticPressDataset(n_cycles=args.n_cycles, n_points=args.n_points, anomaly_prob=args.anomaly_prob, seed=args.seed)
    n = len(dataset)
    indices = np.arange(n)
    np.random.seed(args.seed)
    np.random.shuffle(indices)
    n_train = int(n * args.train_frac)
    n_val = int(n * args.val_frac)
    train_idx = indices[:n_train].tolist()
    val_idx = indices[n_train:n_train + n_val].tolist()
    test_idx = indices[n_train + n_val:].tolist()

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # model
    # infer input dim from dataset
    sample_x, _, _ = dataset[0]
    T, D = sample_x.shape
    cfg = CrossModalAttentionConfig(ts_input_dim=D, d_model=args.d_model, num_heads=args.num_heads)
    model = PressFuseModel(config=cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Auto-compute pos_weight on train split for class imbalance handling.
    train_labels = np.array([dataset.labels[i] for i in train_idx], dtype=np.int64)
    n_pos = int((train_labels == 1).sum())
    n_neg = int((train_labels == 0).sum())
    if n_pos > 0:
        pos_weight_value = float(n_neg / max(1, n_pos))
    else:
        pos_weight_value = 1.0
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"Using pos_weight={pos_weight_value:.4f} (n_neg={n_neg}, n_pos={n_pos})")

    rng = np.random.default_rng(args.seed)

    train_history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for X, y, _ids in train_loader:
            X = X.to(device)
            y = y.to(device)

            if args.anomaly_mix == "peak_drift_step":
                X = apply_peak_drift_step_mix(X, y, rng)

            # events not used; pass X as events to satisfy forward
            outputs = model(X, events=X)
            logits = outputs["binary_logits"]
            if args.loss == "focal":
                loss = focal_loss_from_logits(logits, y, pos_weight, args.gamma)
            else:
                loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # validation
        model.eval()
        val_losses = []
        ys_true = []
        ys_prob = []
        with torch.no_grad():
            for X, y, _ids in val_loader:
                X = X.to(device)
                y = y.to(device)
                outputs = model(X, events=X)
                logits = outputs["binary_logits"]
                if args.loss == "focal":
                    loss = focal_loss_from_logits(logits, y, pos_weight, args.gamma)
                else:
                    loss = criterion(logits, y)
                val_losses.append(loss.item())
                probs = torch.sigmoid(logits).cpu().numpy().ravel()
                ys_prob.extend(probs.tolist())
                ys_true.extend(y.cpu().numpy().ravel().tolist())

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        # metrics
        try:
            val_pred = [1 if p >= 0.5 else 0 for p in ys_prob]
            val_acc = accuracy_score(ys_true, val_pred)
            val_prec = precision_score(ys_true, val_pred, zero_division=0)
            val_rec = recall_score(ys_true, val_pred, zero_division=0)
            val_f1 = f1_score(ys_true, val_pred, zero_division=0)
            val_auc = roc_auc_score(ys_true, ys_prob)
        except Exception:
            val_acc = val_prec = val_rec = val_f1 = val_auc = 0.0

        current_lr = float(optimizer.param_groups[0]["lr"])

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_auc={val_auc:.4f} lr={current_lr:.8f}"
        )
        train_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": float(val_acc),
            "current_lr": current_lr,
        })

        if scheduler is not None:
            scheduler.step()

    # save model
    ckpt_path = os.path.join(out_dir, "model_mvp.ckpt")
    torch.save(model.state_dict(), ckpt_path)

    # evaluate on test set
    model.eval()
    ys_true = []
    ys_prob = []
    ids_all = []
    with torch.no_grad():
        for X, y, ids in test_loader:
            X = X.to(device)
            outputs = model(X, events=X)
            probs = torch.sigmoid(outputs["binary_logits"]).cpu().numpy().ravel()
            ys_prob.extend(probs.tolist())
            ys_true.extend([int(v) for v in y.cpu().numpy().ravel().tolist()])
            ids_all.extend(ids)

    # compute metrics
    pred_labels = [1 if p >= 0.5 else 0 for p in ys_prob]
    metrics = {
        "accuracy": float(accuracy_score(ys_true, pred_labels)),
        "precision": float(precision_score(ys_true, pred_labels, zero_division=0)),
        "recall": float(recall_score(ys_true, pred_labels, zero_division=0)),
        "f1": float(f1_score(ys_true, pred_labels, zero_division=0)),
        "roc_auc": float(roc_auc_score(ys_true, ys_prob)) if len(set(ys_true)) > 1 else 0.0,
    }

    # save metrics and predictions
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    import csv

    preds_path = os.path.join(out_dir, "predictions.csv")
    with open(preds_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["cycle_id", "label", "pred_proba", "pred_label"])
        for cid, ytrue, p in zip(ids_all, ys_true, ys_prob):
            writer.writerow([cid, ytrue, float(p), int(p >= 0.5)])

    # save train history
    hist_path = os.path.join(out_dir, "train_history.csv")
    import pandas as pd

    pd.DataFrame(train_history).to_csv(hist_path, index=False)

    print(f"Saved checkpoint: {ckpt_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved predictions: {preds_path}")
    print(f"Saved train history: {hist_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cycles", type=int, default=512)
    parser.add_argument("--n-points", type=int, default=192)
    parser.add_argument("--anomaly-prob", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss", type=str, default="bce", choices=["bce", "focal"])
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--lr-schedule", type=str, default="none", choices=["none", "cosine"])
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--anomaly-mix", type=str, default="none", choices=["none", "peak_drift_step"])
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/sample_run")
    args = parser.parse_args()
    train(args)

