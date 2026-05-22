from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.labeling.labeling import (
    DEFAULT_CONFIG,
    build_state_label,
    infer_available_columns,
    summarize_label_distribution,
)

sns.set(style="whitegrid")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "data" / "data_R2.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"


def load_input_csv(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, header=None, na_values=["\\N", "\\N.1"])


def add_forced_eqp_status(cfg: dict, eqp_status_col: Optional[int]) -> dict:
    new_cfg = dict(cfg)
    candidates = dict(new_cfg.get("column_candidates", {}))
    if eqp_status_col is not None:
        candidates["eqp_status"] = [eqp_status_col]
    new_cfg["column_candidates"] = candidates
    return new_cfg


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def series_summary(series: pd.Series) -> pd.Series:
    s = safe_numeric(series)
    return s.describe()


def render_report(
    labeled: pd.DataFrame,
    mapping: dict,
    summary: pd.DataFrame,
    out_dir: Path,
    csv_name: str,
) -> None:
    lines = []
    total = len(labeled)
    top_label = summary.sort_values("count", ascending=False).iloc[0]
    alert_count = int(summary.loc[summary["state_label"] == "alert_state", "count"].sum()) if "alert_state" in summary["state_label"].values else 0
    fault_count = int(summary.loc[summary["state_label"] == "suspected_sensor_fault", "count"].sum()) if "suspected_sensor_fault" in summary["state_label"].values else 0

    lines.append(f"# 데이터 분석 리포트\n")
    lines.append(f"- 입력 파일: `{csv_name}`")
    lines.append(f"- 총 행 수: **{total:,}**")
    lines.append(f"- 최빈 라벨: **{top_label['state_label']}** ({int(top_label['count']):,}건, {float(top_label['ratio']):.2%})")
    lines.append(f"- `alert_state` 수: **{alert_count:,}**")
    lines.append(f"- `suspected_sensor_fault` 수: **{fault_count:,}**")
    lines.append("")

    lines.append("## 1) 라벨 분포")
    for _, row in summary.sort_values("count", ascending=False).iterrows():
        lines.append(f"- {row['state_label']}: {int(row['count']):,}건 ({float(row['ratio']):.2%})")
    lines.append("")

    lines.append("## 2) 컬럼 매핑 결과")
    for k, v in mapping.items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    lines.append("## 3) 핵심 해석")
    if float(top_label["ratio"]) >= 0.8:
        lines.append("- 한 개의 라벨이 전체의 대부분을 차지하여 분포 편향이 큽니다.")
    if alert_count / total >= 0.5:
        lines.append("- `alert_state` 비율이 높아 안전 알람 또는 이상 신호가 자주 감지되는 패턴입니다.")
    if fault_count / total >= 0.05:
        lines.append("- `suspected_sensor_fault` 비율도 높아 센서 고착, 0값 지속, 결측 반복 여부를 추가 점검할 필요가 있습니다.")
    if mapping.get("eqp_status") is None:
        lines.append("- `eqp_status`가 자동 인식되지 않아 장비 상태 기반의 `idle` 판정이 충분히 반영되지 않았을 가능성이 있습니다.")
    lines.append("- 보고서에는 라벨 분포 막대그래프와 power-current 산점도를 함께 넣으면 해석력이 높아집니다.")
    lines.append("")

    lines.append("## 4) 시각화 파일")
    lines.append("- `results/plots/label_distribution.png`")
    lines.append("- `results/plots/power_current_scatter.png`")
    lines.append("- `results/plots/device_*_timeline.png`")
    lines.append("")

    report_path = out_dir / "analysis_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8-sig")
    print(f"Saved report to: {report_path}")


def save_label_distribution_plot(summary: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(7, 4))
    sns.barplot(
        data=summary.sort_values("count", ascending=False),
        x="state_label",
        y="count",
        hue="state_label",
        legend=False,
    )
    plt.xticks(rotation=30, ha="right")
    plt.title("State Label Distribution")
    plt.tight_layout()
    path = out_dir / "label_distribution.png"
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"Saved plot to: {path}")


def save_power_current_scatter(labeled: pd.DataFrame, mapping: dict, out_dir: Path, sample_n: int = 200000) -> None:
    pcol = mapping.get("power")
    ccol = mapping.get("current")
    if pcol is None or ccol is None:
        print("power/current 컬럼을 찾지 못해 산점도는 저장하지 않았습니다.")
        return

    plot_df = labeled[[pcol, ccol, "state_label"]].copy()
    plot_df[pcol] = pd.to_numeric(plot_df[pcol], errors="coerce")
    plot_df[ccol] = pd.to_numeric(plot_df[ccol], errors="coerce")
    plot_df = plot_df.dropna(subset=[pcol, ccol, "state_label"])

    if len(plot_df) > sample_n:
        plot_df = plot_df.sample(sample_n, random_state=42)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=plot_df, x=pcol, y=ccol, hue="state_label", s=10, alpha=0.55)
    plt.title("Power vs Current by Label")
    plt.tight_layout()
    path = out_dir / "power_current_scatter.png"
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"Saved plot to: {path}")


def save_device_timeline(labeled: pd.DataFrame, mapping: dict, out_dir: Path, top_n: int = 5) -> None:
    device_col = mapping.get("device_id")
    ts_col = "__timestamp__"

    if device_col is None or device_col not in labeled.columns:
        print("device_id 컬럼을 찾지 못해 타임라인은 저장하지 않았습니다.")
        return

    df = labeled.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])
    if df.empty:
        print("타임스탬프를 찾지 못해 타임라인은 저장하지 않았습니다.")
        return

    top_devices = df[device_col].value_counts().head(top_n).index.tolist()

    for dev in top_devices:
        sub = df[df[device_col] == dev].sort_values(ts_col).copy()
        if sub.empty:
            continue

        sub["group"] = (sub["state_label"] != sub["state_label"].shift()).cumsum()

        fig, ax = plt.subplots(figsize=(12, 2.8))
        for _, g in sub.groupby("group"):
            label = g["state_label"].iloc[0]
            start = g[ts_col].iloc[0]
            end = g[ts_col].iloc[-1]
            ax.axvspan(start, end, alpha=0.65, label=label)

        ax.set_title(f"Device {dev} state timeline")
        ax.set_yticks([])
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
        handles, leg_labels = ax.get_legend_handles_labels()
        unique = dict(zip(leg_labels, handles))
        ax.legend(unique.values(), unique.keys(), bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()

        safe_name = str(dev).replace("/", "_").replace("\\", "_").replace(" ", "_")
        path = out_dir / f"device_{safe_name}_timeline.png"
        plt.savefig(path, dpi=160)
        plt.close()
        print(f"Saved plot to: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV))
    parser.add_argument("--out", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--eqp-status-col", type=int, default=None)
    parser.add_argument("--top-devices", type=int, default=5)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    frame = load_input_csv(csv_path)
    cfg = add_forced_eqp_status(DEFAULT_CONFIG, args.eqp_status_col)
    labeled = build_state_label(frame, cfg)

    labeled_path = out_dir / "labeled_data.csv"
    labeled.to_csv(labeled_path, index=False, encoding="utf-8-sig")
    print(f"Saved labeled data to: {labeled_path}")

    mapping = infer_available_columns(labeled)
    labeled["__timestamp__"] = pd.NaT
    ts_col = mapping.get("timestamp")
    if ts_col is not None and ts_col in labeled.columns:
        labeled["__timestamp__"] = pd.to_datetime(labeled[ts_col], errors="coerce")

    summary = summarize_label_distribution(labeled)
    summary_path = out_dir / "label_distribution.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Saved label distribution summary to: {summary_path}")

    print("\n---- Inferred column mapping ----")
    for k, v in mapping.items():
        print(f"{k:12s} -> {v}")

    print("\n---- Label distribution summary ----")
    print(summary.to_string(index=False))

    save_label_distribution_plot(summary, plots_dir)
    save_power_current_scatter(labeled, mapping, plots_dir)
    save_device_timeline(labeled, mapping, plots_dir, top_n=args.top_devices)
    render_report(labeled, mapping, summary, out_dir, csv_path.name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())