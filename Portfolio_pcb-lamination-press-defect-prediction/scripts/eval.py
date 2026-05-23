"""Evaluation CLI for the semiconductor PCB lamination scaffold."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.schema import PressFeatureSpec
from src.data.synthpress import generate_press_cycle
from src.eval.metrics import classification_report_dict


def _read_tabular(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def _heuristic_predict(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Simple baseline for Press data: flag likely anomalies from process drift."""

    if "VACUUM" not in df.columns or "HPPRESS_PV" not in df.columns:
        raise KeyError("Heuristic baseline requires VACUUM and HPPRESS_PV columns")
    grouped = df.groupby("cycle_id", sort=False)
    means = grouped[["VACUUM", "HPPRESS_PV"]].mean()
    scores = 0.6 * (1.0 - means["VACUUM"]) + 0.4 * ((18.0 - means["HPPRESS_PV"]).abs() / 18.0)
    preds = (scores > scores.median()).astype(int)
    labels = grouped["label"].first().astype(int)
    return labels, preds


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a labeled Press dataset or a synthetic demo set")
    parser.add_argument("--input", type=Path, help="CSV/parquet file with labels")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--pred-col", default=None, help="Optional prediction column to evaluate directly")
    parser.add_argument("--cost-fn", type=float, default=100.0)
    parser.add_argument("--cost-fp", type=float, default=5.0)
    parser.add_argument("--synthetic-cycles", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="Optional markdown report path")
    return parser


def _synth_demo(n_cycles: int, seed: int) -> pd.DataFrame:
    rows = []
    for idx in range(n_cycles):
        frame, _, _ = generate_press_cycle(idx + 1, 1000 + idx, anomaly_prob=0.5, seed=seed + idx)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.input is None:
        df = _synth_demo(args.synthetic_cycles, args.seed)
    else:
        df = _read_tabular(args.input)

    if args.pred_col is not None:
        if args.label_col not in df.columns or args.pred_col not in df.columns:
            raise KeyError(f"Expected columns '{args.label_col}' and '{args.pred_col}'")
        y_true = df[args.label_col].astype(int)
        y_pred = df[args.pred_col].astype(int)
        y_score = df[args.pred_col].astype(float)
    elif {"cycle_id", "VACUUM", "HPPRESS_PV", args.label_col}.issubset(df.columns):
        y_true, y_pred = _heuristic_predict(df)
        y_score = None
    else:
        raise KeyError(
            "Input must either contain a prediction column or Press cycle columns such as cycle_id, VACUUM, HPPRESS_PV, label"
        )

    report = classification_report_dict(y_true, y_pred, y_score, cost_fn=args.cost_fn, cost_fp=args.cost_fp)
    lines = ["# Evaluation Report", ""] + [f"- {key}: {value}" for key, value in report.items()]
    text = "\n".join(lines)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

