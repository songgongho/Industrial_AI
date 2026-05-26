from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data.dataset import SyntheticPressDataset, collate_fn
from src.models.pressfuse import PressFuseModel, CrossModalAttentionConfig


def predict(args: argparse.Namespace) -> None:
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    dataset = SyntheticPressDataset(n_cycles=args.n_cycles, n_points=args.n_points, anomaly_prob=args.anomaly_prob, seed=args.seed)
    # use full dataset or subset
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # model
    sample_x, _, _ = dataset[0]
    T, D = sample_x.shape
    cfg = CrossModalAttentionConfig(ts_input_dim=D, d_model=args.d_model, num_heads=args.num_heads)
    model = PressFuseModel(config=cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # load checkpoint
    ckpt = args.checkpoint
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    ids_all = []
    ys_true = []
    ys_prob = []
    ys_attn = []
    with torch.no_grad():
        for X, y, ids in loader:
            X = X.to(device)
            outputs = model(X, events=X)
            probs = torch.sigmoid(outputs["binary_logits"]).cpu().numpy().ravel().tolist()
            ids_all.extend(ids)
            ys_true.extend([int(v) for v in y.cpu().numpy().ravel().tolist()])
            ys_prob.extend(probs)
            # attention is tensor: (B, num_heads, T, T) or (B, num_heads, T, T) depending
            attn = outputs.get("attention")
            if attn is not None:
                # move to cpu and extract mean across batch
                ys_attn.append(attn.cpu().numpy())

    # flatten attn if present
    if ys_attn:
        # concatenate along batch axis and save first sample per batch as example
        all_attn = np.concatenate(ys_attn, axis=0)
        # save as npy for UI to load
        attn_path = os.path.join(out_dir, "attention.npy")
        np.save(attn_path, all_attn)
        print(f"Saved attention npy: {attn_path}")

    # save predictions CSV
    preds_path = os.path.join(out_dir, "predictions.csv")
    with open(preds_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["cycle_id", "label", "pred_proba", "pred_label"])
        for cid, ytrue, p in zip(ids_all, ys_true, ys_prob):
            writer.writerow([cid, ytrue, float(p), int(p >= 0.5)])

    print(f"Saved predictions to {preds_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n-cycles", type=int, default=512)
    parser.add_argument("--n-points", type=int, default=192)
    parser.add_argument("--anomaly-prob", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/sample_run")
    args = parser.parse_args()
    predict(args)

