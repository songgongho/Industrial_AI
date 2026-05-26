from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import precision_recall_fscore_support, roc_curve


def discover_experiments(output_root: Path) -> list[Path]:
    """Discover experiment folders with pattern outputs/exp*/."""
    return sorted([p for p in output_root.glob("exp*/") if p.is_dir()], key=lambda p: p.name)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_figure(fig: go.Figure, html_path: Path, png_path: Path) -> None:
    """Save Plotly figure as HTML and PNG.

    PNG export requires kaleido (`pip install kaleido`).
    """
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    try:
        fig.write_image(str(png_path), scale=2)
    except Exception as exc:
        print(f"[warn] PNG 저장 실패 ({png_path.name}): {exc}. `pip install kaleido` 필요할 수 있습니다.")


def read_experiment(exp_dir: Path) -> dict[str, Any] | None:
    train_path = exp_dir / "train_history.csv"
    metrics_path = exp_dir / "metrics.json"
    pred_path = exp_dir / "predictions.csv"
    params_path = exp_dir / "params.json"

    missing = [
        p.name
        for p in [train_path, metrics_path, pred_path]
        if not p.exists()
    ]
    if missing:
        print(f"[warn] {exp_dir.name} skip: missing {missing}")
        return None

    try:
        train_df = pd.read_csv(train_path)
        pred_df = pd.read_csv(pred_path)
        metrics = load_json(metrics_path)
        params = load_json(params_path)
    except Exception as exc:
        print(f"[warn] {exp_dir.name} skip: load 실패 ({exc})")
        return None

    required_train = {"epoch", "train_loss", "val_loss", "val_acc"}
    required_pred = {"label", "pred_proba"}
    if not required_train.issubset(set(train_df.columns)):
        print(f"[warn] {exp_dir.name} skip: train_history.csv 컬럼 부족")
        return None
    if not required_pred.issubset(set(pred_df.columns)):
        print(f"[warn] {exp_dir.name} skip: predictions.csv 컬럼 부족")
        return None

    y_true = pred_df["label"].astype(int).to_numpy()
    y_prob = pred_df["pred_proba"].astype(float).to_numpy()
    if len(np.unique(y_true)) < 2:
        print(f"[warn] {exp_dir.name} skip: label class 단일")
        return None

    return {
        "name": exp_dir.name,
        "train": train_df,
        "pred": pred_df,
        "metrics": metrics,
        "params": params,
        "y_true": y_true,
        "y_prob": y_prob,
    }


def build_fig_a(experiments: list[dict[str, Any]]) -> go.Figure:
    """Figure A: 2x1 subplot (loss curves + val_acc)."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Train/Val Loss", "Validation Accuracy"),
    )

    for exp in experiments:
        df = exp["train"]
        name = exp["name"]
        fig.add_trace(
            go.Scatter(
                x=df["epoch"],
                y=df["train_loss"],
                mode="lines+markers",
                name=f"{name} train_loss",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["epoch"],
                y=df["val_loss"],
                mode="lines+markers",
                line=dict(dash="dash"),
                name=f"{name} val_loss",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["epoch"],
                y=df["val_acc"],
                mode="lines+markers",
                name=f"{name} val_acc",
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        title="Figure A - 학습 곡선",
        template="plotly_white",
        height=850,
    )
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=2, col=1)
    return fig


def build_fig_b(experiments: list[dict[str, Any]]) -> go.Figure:
    """Figure B: ROC curves + optimal-threshold marker + random baseline."""
    fig = go.Figure()

    for exp in experiments:
        y_true = exp["y_true"]
        y_prob = exp["y_prob"]
        name = exp["name"]
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)

        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} ROC"))

        opt_thr = float(exp["metrics"].get("optimal_threshold", 0.5))
        idx = int(np.argmin(np.abs(thresholds - opt_thr)))
        fig.add_trace(
            go.Scatter(
                x=[float(fpr[idx])],
                y=[float(tpr[idx])],
                mode="markers+text",
                marker=dict(color="red", size=9),
                text=[f"{name}\nthr={opt_thr:.4f}"],
                textposition="top center",
                name=f"{name} optimal",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="gray", dash="dash"),
            name="random",
        )
    )
    fig.update_layout(
        title="Figure B - ROC 곡선 + Optimal Threshold",
        template="plotly_white",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=650,
    )
    return fig


def build_fig_c(experiments: list[dict[str, Any]]) -> go.Figure:
    """Figure C: per-experiment histogram subplots with threshold lines."""
    n = len(experiments)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[e["name"] for e in experiments])

    for i, exp in enumerate(experiments):
        r = i // cols + 1
        c = i % cols + 1
        pred_df = exp["pred"]
        y0 = pred_df[pred_df["label"] == 0]["pred_proba"]
        y1 = pred_df[pred_df["label"] == 1]["pred_proba"]

        fig.add_trace(
            go.Histogram(
                x=y0,
                marker_color="blue",
                opacity=0.55,
                name="label=0",
                showlegend=(i == 0),
            ),
            row=r,
            col=c,
        )
        fig.add_trace(
            go.Histogram(
                x=y1,
                marker_color="red",
                opacity=0.55,
                name="label=1",
                showlegend=(i == 0),
            ),
            row=r,
            col=c,
        )

        # Draw threshold lines using add_shape with subplot axis refs
        axis_index = i + 1
        xref = "x" if axis_index == 1 else f"x{axis_index}"
        yref = "paper"

        fig.add_shape(
            type="line",
            x0=0.5,
            x1=0.5,
            y0=0,
            y1=1,
            xref=xref,
            yref=yref,
            line=dict(color="gray", dash="dash"),
        )

        opt_thr = float(exp["metrics"].get("optimal_threshold", 0.5))
        fig.add_shape(
            type="line",
            x0=opt_thr,
            x1=opt_thr,
            y0=0,
            y1=1,
            xref=xref,
            yref=yref,
            line=dict(color="green", dash="dot"),
        )

    fig.update_layout(
        title="Figure C - Pred_Proba 분포",
        barmode="overlay",
        template="plotly_white",
        height=max(500, rows * 320),
    )
    return fig


def build_fig_d(experiments: list[dict[str, Any]]) -> go.Figure:
    """Figure D: F1/Precision/Recall vs threshold with optimal-point marker."""
    fig = go.Figure()
    thresholds = np.linspace(0.0, 1.0, 201)

    for exp in experiments:
        y_true = exp["y_true"]
        y_prob = exp["y_prob"]
        name = exp["name"]

        p_vals: list[float] = []
        r_vals: list[float] = []
        f_vals: list[float] = []

        for thr in thresholds:
            y_pred = (y_prob >= thr).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(
                y_true,
                y_pred,
                average="binary",
                zero_division=0,
            )
            p_vals.append(float(p))
            r_vals.append(float(r))
            f_vals.append(float(f1))

        fig.add_trace(go.Scatter(x=thresholds, y=f_vals, mode="lines", name=f"{name} F1"))
        fig.add_trace(
            go.Scatter(
                x=thresholds,
                y=p_vals,
                mode="lines",
                line=dict(dash="dot"),
                name=f"{name} Precision",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=thresholds,
                y=r_vals,
                mode="lines",
                line=dict(dash="dash"),
                name=f"{name} Recall",
            )
        )

        opt_thr = float(exp["metrics"].get("optimal_threshold", 0.5))
        opt_idx = int(np.argmin(np.abs(thresholds - opt_thr)))
        fig.add_trace(
            go.Scatter(
                x=[float(thresholds[opt_idx])],
                y=[float(f_vals[opt_idx])],
                mode="markers+text",
                marker=dict(size=8, color="green"),
                text=[f"{name}\nopt={opt_thr:.4f}"],
                textposition="top center",
                name=f"{name} optimal",
            )
        )

    fig.update_layout(
        title="Figure D - F1/Precision/Recall vs Threshold",
        template="plotly_white",
        xaxis_title="Threshold",
        yaxis_title="Score",
        height=700,
    )
    return fig


def print_summary_table(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        print("[info] summary is empty")
        return
    cols = [
        "exp_name",
        "f1",
        "roc_auc",
        "optimal_threshold",
        "epochs",
        "lr",
        "loss_type",
        "gamma",
        "lr_schedule",
    ]
    print("\n=== Experiment Summary ===")
    print(summary_df[cols].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=str, default="outputs")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    exp_dirs = discover_experiments(output_root)
    if not exp_dirs:
        print("[warn] outputs/exp*/ 패턴에 해당하는 폴더가 없습니다.")
        summary_path = output_root / "experiment_summary.csv"
        pd.DataFrame(
            columns=[
                "exp_name",
                "f1",
                "roc_auc",
                "optimal_threshold",
                "epochs",
                "lr",
                "loss_type",
                "gamma",
                "lr_schedule",
            ]
        ).to_csv(summary_path, index=False)
        print(f"Saved empty summary: {summary_path}")
        return

    experiments: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for exp_dir in exp_dirs:
        rec = read_experiment(exp_dir)
        if rec is None:
            continue
        experiments.append(rec)

        metrics = rec["metrics"]
        params = rec["params"]
        train_df = rec["train"]

        epochs = params.get("epochs")
        if epochs is None:
            epochs = int(train_df["epoch"].max())
        lr = params.get("lr")
        if lr is None and "current_lr" in train_df.columns:
            lr = float(train_df["current_lr"].iloc[0])

        rows.append(
            {
                "exp_name": rec["name"],
                "f1": float(metrics.get("f1", np.nan)),
                "roc_auc": float(metrics.get("roc_auc", np.nan)),
                "optimal_threshold": float(metrics.get("optimal_threshold", 0.5)),
                "epochs": epochs,
                "lr": lr,
                "loss_type": params.get("loss", "bce"),
                "gamma": params.get("gamma", 2.0),
                "lr_schedule": params.get("lr_schedule", "none"),
            }
        )

    if not experiments:
        print("[warn] 유효한 exp 폴더를 찾지 못했습니다.")
        summary_path = output_root / "experiment_summary.csv"
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        print(f"Saved summary: {summary_path}")
        return

    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig_a = build_fig_a(experiments)
    save_figure(fig_a, figures_dir / "fig_A_loss_curves.html", figures_dir / "fig_A_loss_curves.png")

    fig_b = build_fig_b(experiments)
    save_figure(fig_b, figures_dir / "fig_B_roc.html", figures_dir / "fig_B_roc.png")

    fig_c = build_fig_c(experiments)
    save_figure(fig_c, figures_dir / "fig_C_proba_dist.html", figures_dir / "fig_C_proba_dist.png")

    fig_d = build_fig_d(experiments)
    save_figure(fig_d, figures_dir / "fig_D_metric_curves.html", figures_dir / "fig_D_metric_curves.png")

    summary_df = pd.DataFrame(
        rows,
        columns=[
            "exp_name",
            "f1",
            "roc_auc",
            "optimal_threshold",
            "epochs",
            "lr",
            "loss_type",
            "gamma",
            "lr_schedule",
        ],
    )
    summary_path = output_root / "experiment_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print_summary_table(summary_df)
    print(f"\nSaved figures to: {figures_dir}")
    print(f"Saved summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
