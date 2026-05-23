from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")


def _summarize(df: pd.DataFrame) -> dict[str, float | int]:
    total_count = int(len(df))
    defect_count = int(df["is_defect"].sum())
    defect_rate_pct = round((defect_count / total_count * 100.0) if total_count else 0.0, 4)

    anomaly_q95 = float(df["anomaly_score"].quantile(0.95)) if "anomaly_score" in df.columns else 0.0

    return {
        "total_count": total_count,
        "defect_count": defect_count,
        "defect_rate_pct": defect_rate_pct,
        "anomaly_score_q95": round(anomaly_q95, 4),
    }


def _save_defect_pareto(defect_type_counts: pd.DataFrame, figs_dir: Path) -> None:
    if defect_type_counts.empty:
        return

    top = defect_type_counts.head(15).copy()
    top["cum_pct"] = top["count"].cumsum() / max(float(top["count"].sum()), 1.0) * 100.0

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.bar(top["defect_type"], top["count"], color="#4E79A7")
    ax1.set_ylabel("count")
    ax1.set_title("Defect Type Pareto (Top 15)")
    ax1.tick_params(axis="x", rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(top["defect_type"], top["cum_pct"], color="#E15759", marker="o")
    ax2.set_ylabel("cumulative %")
    ax2.set_ylim(0, 110)

    fig.tight_layout()
    fig.savefig(figs_dir / "defect_type_pareto.png", dpi=140)
    plt.close(fig)


def _write_markdown_report(
    report_path: Path,
    summary: dict[str, float | int],
    meta: dict[str, object],
    defect_type_counts: pd.DataFrame,
    machine_rates: pd.DataFrame,
    source_rates: pd.DataFrame,
    feature_contrib: pd.DataFrame,
    pm_events: pd.DataFrame,
    pm_machine_risk: pd.DataFrame,
    pm_lead_summary: pd.DataFrame,
) -> None:
    def append_table(headers: list[str], rows: list[list[object]]) -> None:
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for row in rows:
            lines.append("| " + " | ".join(str(v) for v in row) + " |")

    top_types = defect_type_counts.head(5)
    top_machines = machine_rates.head(5)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: list[str] = []
    lines.append("# Press 공정 불량/이상 탐색 보고서")
    lines.append("")
    lines.append(f"- 생성 시각: `{generated_at}`")
    lines.append(f"- 데이터 건수: `{summary['total_count']}`")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append(
        f"> 전체 {summary['total_count']}건 중 {summary['defect_count']}건({summary['defect_rate_pct']}%)이 불량/이상으로 탐지되었습니다. "
        f"현재 라벨 전략은 `{meta.get('label_strategy')}` 입니다."
    )
    lines.append("")
    lines.append("## 1. 핵심 KPI")
    append_table(
        ["지표", "값"],
        [
            ["총 데이터 수", summary["total_count"]],
            ["불량(또는 이상 추정) 수", summary["defect_count"]],
            ["불량률(%)", summary["defect_rate_pct"]],
            ["이상점수 상위 5% 경계(q95)", summary["anomaly_score_q95"]],
        ],
    )

    lines.append("")
    lines.append("## 2. 분석 설정 및 추론 정보")
    append_table(
        ["항목", "값"],
        [
            ["라벨 전략", meta.get("label_strategy")],
            ["결과 컬럼 추론", meta.get("result_col")],
            ["불량 컬럼 추론", meta.get("defect_col")],
            ["시간 컬럼 추론", meta.get("time_col")],
            ["설비 컬럼 추론", meta.get("machine_col")],
            ["이상 임계치", meta.get("anomaly_threshold")],
            ["PM 경보 임계치(%)", meta.get("pm_alert_threshold_pct")],
            ["PM 이벤트 수", meta.get("pm_event_count")],
        ],
    )

    lines.append("")
    lines.append("## 3. 불량 유형 분석")
    if top_types.empty:
        lines.append("- 불량/이상 행이 감지되지 않았습니다.")
    else:
        rows = []
        total_def = max(int(summary["defect_count"]), 1)
        for _, row in top_types.iterrows():
            rows.append([row["defect_type"], int(row["count"]), round(int(row["count"]) / total_def * 100, 2)])
        append_table(["불량 유형", "건수", "비중(%)"], rows)

    lines.append("")
    lines.append("![불량 유형 상위 분포](figures/top_defect_types.png)")
    lines.append("")
    lines.append("> 그림 1. 상위 불량 유형의 건수 분포입니다. 특정 유형 쏠림 여부를 빠르게 확인할 수 있습니다.")
    lines.append("")
    lines.append("![불량 유형 파레토](figures/defect_type_pareto.png)")
    lines.append("")
    lines.append("> 그림 2. 파레토 차트로 핵심 불량 유형 집중도를 확인할 수 있습니다.")

    lines.append("")
    lines.append("## 4. 설비/파일 단위 위험도")
    if top_machines.empty:
        lines.append("- 설비 위험도 계산 결과가 없습니다.")
    else:
        append_table(
            ["설비", "불량률(%)"],
            [[row["machine_id"], round(float(row["defect_rate_pct"]), 4)] for _, row in top_machines.iterrows()],
        )

    lines.append("")
    if source_rates.empty:
        lines.append("- 파일별 불량률을 계산할 source_file 컬럼이 없습니다.")
    else:
        append_table(
            ["원본 파일", "불량률(%)"],
            [[row["source_file"], round(float(row["defect_rate_pct"]), 4)] for _, row in source_rates.head(5).iterrows()],
        )

    lines.append("")
    lines.append("## 5. 이상 기여 센서")
    if feature_contrib.empty:
        lines.append("- 이상 기여 센서 데이터가 없습니다.")
    else:
        append_table(
            ["센서(태그)", "건수", "평균 이상점수"],
            [
                [row["primary_anomaly_feature"], int(row["count"]), round(float(row["avg_anomaly_score"]), 3)]
                for _, row in feature_contrib.head(8).iterrows()
            ],
        )

    lines.append("")
    lines.append("## 6. 예지보전 이벤트 관찰")
    if pm_events.empty:
        lines.append("- PM 이벤트 타임라인이 추출되지 않았습니다.")
    else:
        append_table(
            ["이벤트", "설비", "시작", "지속(min)", "불량행", "최대위험점수"],
            [
                [
                    int(row["pm_event_id"]),
                    row["machine_id"],
                    row["start_time"],
                    round(float(row["duration_min"]), 3),
                    int(row["defect_rows"]),
                    round(float(row["max_pm_risk_score"]), 3),
                ]
                for _, row in pm_events.head(8).iterrows()
            ],
        )

    lines.append("")
    if not pm_machine_risk.empty:
        append_table(
            ["설비", "경보횟수", "최대위험점수", "불량건수"],
            [
                [
                    row["machine_id"],
                    int(row["alert_count"]),
                    round(float(row["max_pm_risk_score"]), 3),
                    int(row["defect_count"]),
                ]
                for _, row in pm_machine_risk.head(5).iterrows()
            ],
        )

    lines.append("")
    if not pm_lead_summary.empty:
        lead = pm_lead_summary.iloc[0]
        append_table(
            ["리드타임 지표", "값(분)"],
            [
                ["샘플 수", int(lead["count"])],
                ["평균", round(float(lead["mean"]), 3)],
                ["중앙값(p50)", round(float(lead["p50"]), 3)],
                ["상위 90%(p90)", round(float(lead["p90"]), 3)],
            ],
        )

    lines.append("")
    lines.append("![PM 위험도 타임라인](figures/pm_risk_timeline.png)")
    lines.append("")
    lines.append("> 그림 3. 시간축 기준 PM 위험도 추세와 경보 임계치를 함께 표시합니다.")

    lines.append("")
    lines.append("## 7. 권고 사항")
    lines.append("1. `primary_anomaly_feature_contrib.csv` 상위 센서를 중심으로 설비 점검 우선순위를 설정하세요.")
    lines.append("2. `pm_event_timeline.csv`의 이벤트 시작 시점을 기준으로 사전 정비 리드타임 정책을 정의하세요.")
    lines.append("3. 라벨 데이터 확보 시 `explicit_labels` 전략으로 전환해 분류 정확도를 검증하세요.")

    lines.append("")
    lines.append("## 8. 산출물 파일")
    lines.append("- `tables/summary.csv`")
    lines.append("- `tables/defect_type_counts.csv`")
    lines.append("- `tables/defect_type_pareto.csv`")
    lines.append("- `tables/defect_rate_by_machine.csv`")
    lines.append("- `tables/defect_rate_by_source_file.csv`")
    lines.append("- `tables/primary_anomaly_feature_contrib.csv`")
    lines.append("- `tables/pm_event_timeline.csv`")
    lines.append("- `tables/pm_machine_risk_summary.csv`")
    lines.append("- `tables/pm_alert_leadtime_summary.csv`")
    lines.append("- `figures/top_defect_types.png`")
    lines.append("- `figures/defect_type_pareto.png`")
    lines.append("- `figures/pm_risk_timeline.png`")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def export_reports(
    prepared_df: pd.DataFrame,
    output_dir: Path,
    meta: dict[str, object],
) -> dict[str, object]:
    """Write analysis tables/figures and return report payload."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    figs_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    summary = _summarize(prepared_df)
    pd.DataFrame([summary]).to_csv(tables_dir / "summary.csv", index=False)

    defect_type_counts = (
        prepared_df.loc[prepared_df["is_defect"], "defect_type"]
        .value_counts(dropna=False)
        .rename_axis("defect_type")
        .reset_index(name="count")
    )
    defect_type_counts.to_csv(tables_dir / "defect_type_counts.csv", index=False)

    defect_type_pareto = defect_type_counts.copy()
    if not defect_type_pareto.empty:
        defect_type_pareto["cum_count"] = defect_type_pareto["count"].cumsum()
        defect_type_pareto["cum_pct"] = defect_type_pareto["cum_count"] / defect_type_pareto["count"].sum() * 100.0
    defect_type_pareto.to_csv(tables_dir / "defect_type_pareto.csv", index=False)

    machine_rates = (
        prepared_df.groupby("machine_id", dropna=False)["is_defect"]
        .mean()
        .mul(100)
        .reset_index(name="defect_rate_pct")
        .sort_values("defect_rate_pct", ascending=False)
    )
    machine_rates.to_csv(tables_dir / "defect_rate_by_machine.csv", index=False)

    source_rates = pd.DataFrame(columns=["source_file", "defect_rate_pct"])
    if "source_file" in prepared_df.columns:
        source_rates = (
            prepared_df.groupby("source_file", dropna=False)["is_defect"]
            .mean()
            .mul(100)
            .reset_index(name="defect_rate_pct")
            .sort_values("defect_rate_pct", ascending=False)
        )
    source_rates.to_csv(tables_dir / "defect_rate_by_source_file.csv", index=False)

    feature_contrib = pd.DataFrame(columns=["primary_anomaly_feature", "count", "avg_anomaly_score"])
    if "primary_anomaly_feature" in prepared_df.columns:
        feature_contrib = (
            prepared_df.loc[prepared_df["is_defect"]]
            .groupby("primary_anomaly_feature", dropna=False)
            .agg(count=("is_defect", "size"), avg_anomaly_score=("anomaly_score", "mean"))
            .reset_index()
            .sort_values(["count", "avg_anomaly_score"], ascending=False)
        )
    feature_contrib.to_csv(tables_dir / "primary_anomaly_feature_contrib.csv", index=False)

    pm_events = pd.DataFrame(
        columns=["pm_event_id", "machine_id", "start_time", "end_time", "duration_min", "alert_rows", "defect_rows", "max_pm_risk_score"]
    )
    if "pm_event_id" in prepared_df.columns:
        pm_alert_rows = prepared_df[prepared_df["pm_event_id"].notna()].copy()
        if not pm_alert_rows.empty:
            pm_events = (
                pm_alert_rows.groupby(["pm_event_id", "machine_id"], dropna=False)
                .agg(
                    start_time=("event_time", "min"),
                    end_time=("event_time", "max"),
                    alert_rows=("pm_alert_flag", "size"),
                    defect_rows=("is_defect", "sum"),
                    max_pm_risk_score=("pm_risk_score", "max"),
                )
                .reset_index()
            )
            pm_events["duration_min"] = (
                pm_events["end_time"] - pm_events["start_time"]
            ).dt.total_seconds().div(60).fillna(0.0)
            pm_events = pm_events.sort_values("start_time")
    pm_events.to_csv(tables_dir / "pm_event_timeline.csv", index=False)

    pm_machine_risk = (
        prepared_df.groupby("machine_id", dropna=False)
        .agg(
            avg_pm_risk_score=("pm_risk_score", "mean"),
            max_pm_risk_score=("pm_risk_score", "max"),
            alert_count=("pm_alert_flag", "sum"),
            defect_count=("is_defect", "sum"),
        )
        .reset_index()
        .sort_values(["alert_count", "max_pm_risk_score"], ascending=False)
    )
    pm_machine_risk.to_csv(tables_dir / "pm_machine_risk_summary.csv", index=False)

    pm_lead = pd.to_numeric(prepared_df.get("pm_lead_time_min"), errors="coerce")
    pm_lead = pm_lead[(pm_lead >= 0) & pm_lead.notna()]
    pm_lead_summary = pd.DataFrame(columns=["count", "mean", "p50", "p90"])
    if not pm_lead.empty:
        pm_lead_summary = pd.DataFrame(
            [
                {
                    "count": int(pm_lead.count()),
                    "mean": float(pm_lead.mean()),
                    "p50": float(pm_lead.quantile(0.5)),
                    "p90": float(pm_lead.quantile(0.9)),
                }
            ]
        )
    pm_lead_summary.to_csv(tables_dir / "pm_alert_leadtime_summary.csv", index=False)

    if "event_hour" in prepared_df.columns and prepared_df["event_hour"].notna().any():
        hour_rates = (
            prepared_df.groupby("event_hour", dropna=False)["is_defect"]
            .mean()
            .mul(100)
            .reset_index(name="defect_rate_pct")
            .sort_values("event_hour")
        )
        hour_rates.to_csv(tables_dir / "defect_rate_by_hour.csv", index=False)

    if not defect_type_counts.empty:
        top = defect_type_counts.head(15)
        plt.figure(figsize=(10, 5))
        plt.bar(top["defect_type"], top["count"])
        plt.title("Top Defect Types")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(figs_dir / "top_defect_types.png", dpi=140)
        plt.close()

    _save_defect_pareto(defect_type_counts, figs_dir)

    if "event_time" in prepared_df.columns and prepared_df["event_time"].notna().any():
        timeline = prepared_df.sort_values("event_time")
        plt.figure(figsize=(11, 4))
        plt.plot(timeline["event_time"], timeline["pm_risk_score"], linewidth=1.0, color="#4E79A7")
        plt.axhline(float(meta.get("pm_alert_threshold_pct", 97.0)), linestyle="--", color="#E15759")
        plt.title("PM Risk Timeline")
        plt.ylabel("pm_risk_score")
        plt.tight_layout()
        plt.savefig(figs_dir / "pm_risk_timeline.png", dpi=140)
        plt.close()

    prepared_df.to_csv(tables_dir / "combined_dataset.csv", index=False)

    _write_markdown_report(
        output_dir / "report.md",
        summary,
        meta,
        defect_type_counts,
        machine_rates,
        source_rates,
        feature_contrib,
        pm_events,
        pm_machine_risk,
        pm_lead_summary,
    )

    return {
        "summary": summary,
        "meta": meta,
        "output_dir": str(output_dir),
    }

