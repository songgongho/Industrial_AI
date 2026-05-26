#!/usr/bin/env python
import json
import os

import numpy as np
import pandas as pd


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IN_PATH = os.path.join(ROOT, "data", "customer", "processed", "master_synchronized.parquet")
OUT_DIR = os.path.join(ROOT, "outputs", "sample_run")
os.makedirs(OUT_DIR, exist_ok=True)


def _pick_numeric_columns(df: pd.DataFrame, max_cols: int = 12) -> list[str]:
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    ignore = {"cycle_id", "label", "pred_label"}
    cols = [c for c in numeric if c not in ignore]
    return cols[:max_cols]


def run_correlation_fallback(df: pd.DataFrame) -> dict:
    cols = _pick_numeric_columns(df)
    if len(cols) < 2:
        return {"status": "no_numeric_data", "used_columns": cols}

    corr = df[cols].corr().fillna(0.0)
    # Build simple directed edges by upper triangle magnitude
    edges = []
    for i, src in enumerate(cols):
        for j, tgt in enumerate(cols):
            if i == j:
                continue
            w = float(abs(corr.iloc[i, j]))
            if w >= 0.2:
                edges.append((src, tgt, w))

    edges_df = pd.DataFrame(edges, columns=["source", "target", "weight"]).sort_values("weight", ascending=False)
    edges_path = os.path.join(OUT_DIR, "edge_scores.csv")
    adj_path = os.path.join(OUT_DIR, "adjacency.csv")
    corr.to_csv(adj_path)
    edges_df.to_csv(edges_path, index=False)

    return {
        "status": "correlation_fallback",
        "used_columns": cols,
        "adjacency_path": adj_path,
        "edge_scores_path": edges_path,
        "n_edges": int(len(edges_df)),
    }


def main() -> None:
    if not os.path.exists(IN_PATH):
        out = {
            "status": "input_missing",
            "notes": "No synchronized input found. Run scripts/synchronize_customer_data.py first.",
            "input_path": IN_PATH,
        }
        out_path = os.path.join(ROOT, "outputs", "pcmci_result.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"Wrote {out_path}")
        return

    df = pd.read_parquet(IN_PATH)
    result = {}

    # Try tigramite availability, but keep robust fallback.
    try:
        import tigramite  # type: ignore  # noqa: F401
        # Minimal note: real PCMCI pipeline can be wired here later.
        result = {
            "status": "tigramite_available",
            "notes": "tigramite detected; currently using correlation fallback artifact generation.",
        }
        result.update(run_correlation_fallback(df))
    except Exception:
        result = run_correlation_fallback(df)

    out_path = os.path.join(ROOT, "outputs", "pcmci_result.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"Wrote {out_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

