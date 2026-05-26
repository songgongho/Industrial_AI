"""Quality intelligence dashboard for PCB press / defect analysis.

Run:
  streamlit run scripts/quality_intelligence_ui.py
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    import pyodbc
except Exception:  # pragma: no cover - optional in some environments
    pyodbc = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs" / "quality_ui"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="PCB 품질 인텔리전스 대시보드",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
.block-container { padding-top: 1.25rem; padding-bottom: 1.5rem; }
.hero {
    padding: 1.2rem 1.4rem;
    border-radius: 20px;
    background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 45%, #38bdf8 100%);
    color: white;
    box-shadow: 0 14px 40px rgba(15, 23, 42, 0.18);
    margin-bottom: 1rem;
}
.hero h1 { margin: 0; font-size: 2.0rem; }
.hero p { margin: 0.4rem 0 0; opacity: 0.92; }
.kpi-box {
    border-radius: 18px;
    background: white;
    border: 1px solid rgba(148, 163, 184, 0.25);
    padding: 1rem 1.1rem;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
}
.kpi-label { color: #64748b; font-size: 0.85rem; margin-bottom: 0.2rem; }
.kpi-value { color: #0f172a; font-size: 1.7rem; font-weight: 700; line-height: 1.05; }
.kpi-delta { color: #0369a1; font-size: 0.83rem; margin-top: 0.15rem; }
.section-card {
    border-radius: 18px;
    background: white;
    border: 1px solid rgba(148, 163, 184, 0.20);
    padding: 1rem 1rem 0.5rem;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
    margin-bottom: 1rem;
}
.small-note { color: #64748b; font-size: 0.9rem; }
.badge {
    display: inline-block;
    padding: 0.22rem 0.55rem;
    border-radius: 999px;
    background: #eff6ff;
    color: #1d4ed8;
    font-size: 0.78rem;
    margin-right: 0.35rem;
    margin-bottom: 0.3rem;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


@dataclass(frozen=True)
class LocalDataBundle:
    defect_master: pd.DataFrame | None
    quality_summary: pd.DataFrame | None
    press_logs: dict[str, pd.DataFrame]
    source_notes: list[str]


REQUEST_ITEMS = [
    {
        "우선순위": "필수",
        "데이터": "일별/주간/월별 품질 현황",
        "권장 컬럼": "date, line, product, defect_code, defect_name, qty, scrap_qty, rework_qty, ppm",
        "이유": "추세, 계절성, 급격한 품질 악화 시점 탐지",
    },
    {
        "우선순위": "필수",
        "데이터": "설비 가동/비가동 현황",
        "권장 컬럼": "machine_id, process_id, start_time, end_time, state, reason_code, downtime_min",
        "이유": "가동률, 병목, 비가동 원인 분석",
    },
    {
        "우선순위": "필수",
        "데이터": "공정별 설비 가동율",
        "권장 컬럼": "date, process_id, machine_id, uptime_rate, planned_time, actual_time",
        "이유": "PRESS 포함 공정별 안정성 비교와 병목 탐지",
    },
    {
        "우선순위": "필수",
        "데이터": "알람 내역",
        "권장 컬럼": "alarm_time, alarm_code, alarm_name, severity, machine_id, duration_sec, cleared_time",
        "이유": "불량 전조 신호, 공정 이상 이벤트와 불량의 시점 정합",
    },
    {
        "우선순위": "필수",
        "데이터": "PRESS 공정 레시피/설정값 이력",
        "권장 컬럼": "cycle_id, set_pressure, set_temp, set_vacuum, actual_pressure, actual_temp, actual_vacuum",
        "이유": "원인-결과 학습과 공정 조건 최적화",
    },
    {
        "우선순위": "권장",
        "데이터": "LOT / PANEL / CYCLE 매핑키",
        "권장 컬럼": "lot_id, panel_id, cycle_id, product_code, process_route",
        "이유": "품질-설비-알람 데이터를 한 번에 조인하기 위한 핵심 키",
    },
    {
        "우선순위": "권장",
        "데이터": "검사 상세 결과",
        "권장 컬럼": "inspect_time, defect_code, defect_position, inspector, photo_id, recheck_flag",
        "이유": "불량 유형 세분화 및 위치 기반 분석",
    },
    {
        "우선순위": "권장",
        "데이터": "유지보수/교정 이력",
        "권장 컬럼": "maintenance_time, machine_id, action_type, replaced_part, calibration_result",
        "이유": "성능 저하, 반복 불량, 교정 전후 효과 분석",
    },
    {
        "우선순위": "있으면 매우 좋음",
        "데이터": "교대/작업자/라인 정보",
        "권장 컬럼": "shift, operator_id, line_id, team, start_time, end_time",
        "이유": "인적 요인과 교대별 품질 편차 확인",
    },
    {
        "우선순위": "있으면 매우 좋음",
        "데이터": "자재/원자재 lot 정보",
        "권장 컬럼": "incoming_lot, supplier, material_batch, receipt_date, quality_grade",
        "이유": "원자재 불량과 후공정 품질 상관 확인",
    },
    {
        "우선순위": "있으면 매우 좋음",
        "데이터": "환경 데이터",
        "권장 컬럼": "temperature, humidity, dust, timestamp, area_id",
        "이유": "외생 변수로 인한 품질 변동 보정",
    },
]


def _fmt_int(value: Any) -> str:
    try:
        return f"{int(float(value)):,}"
    except Exception:
        return "-"


def _clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False).replace("-", "0"), errors="coerce").fillna(0)


def _excel_files() -> list[Path]:
    return sorted(RAW_DIR.glob("*.xls"), key=lambda p: p.stat().st_size)


def _local_defect_master_path() -> Path | None:
    for path in _excel_files():
        try:
            df = pd.read_excel(path, engine="xlrd", nrows=3)
        except Exception:
            continue
        cols = {str(c).strip() for c in df.columns}
        if {"불량유형코드", "공정명", "사용여부"}.issubset(cols):
            return path
    return None


def _local_quality_path() -> Path | None:
    for path in _excel_files():
        try:
            df = pd.read_excel(path, engine="xlrd", nrows=3)
        except Exception:
            continue
        cols = {str(c).strip() for c in df.columns}
        if "TOTAL" in cols and any(str(c).endswith("_PPM") for c in cols):
            return path
    return None


@st.cache_data(show_spinner=False)
def load_excel_any(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xls", ".xlsx", ".xlsm"}:
        return pd.read_excel(path, engine="xlrd" if path.suffix.lower() == ".xls" else "openpyxl")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported spreadsheet file: {path.suffix}")


@st.cache_data(show_spinner=False)
def load_quality_summary(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        path = _local_quality_path()
    if path is None:
        return None
    df = load_excel_any(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


@st.cache_data(show_spinner=False)
def load_defect_master(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        path = _local_defect_master_path()
    if path is None:
        return None
    df = load_excel_any(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


@st.cache_data(show_spinner=False)
def load_mdb_table(path: str | Path) -> pd.DataFrame:
    if pyodbc is None:
        raise RuntimeError("pyodbc is required to load MDB files")
    conn = pyodbc.connect(rf"DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={Path(path)};")
    try:
        return pd.read_sql("SELECT * FROM [Data]", conn)
    finally:
        conn.close()


def load_press_logs(uploaded_mdb: Any | None = None) -> dict[str, pd.DataFrame]:
    logs: dict[str, pd.DataFrame] = {}
    if uploaded_mdb is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_mdb.name).suffix) as tmp:
            tmp.write(uploaded_mdb.getbuffer())
            temp_path = Path(tmp.name)
        try:
            logs[uploaded_mdb.name] = load_mdb_table(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)
        return logs

    for path in sorted(RAW_DIR.glob("*.mdb")):
        try:
            logs[path.name] = load_mdb_table(path)
        except Exception:
            continue
    return logs


def summarize_quality(df: pd.DataFrame) -> dict[str, Any]:
    df = df.copy()
    if df.empty:
        return {}
    period_col = df.columns[0]
    period_text = df[period_col].astype(str)
    df = df[~period_text.str.contains(r"조회|합계|total|계", case=False, na=False)].copy()
    count_cols = [
        c
        for c in df.columns
        if c not in {period_col, "총 판넬수량", "총 PCS", "TOTAL", "PPM"} and not str(c).endswith("_PPM")
    ]
    ppm_cols = [c for c in df.columns if str(c).endswith("_PPM")]
    if len(count_cols) >= 1:
        df[count_cols] = df[count_cols].apply(_clean_numeric)
    if len(ppm_cols) >= 1:
        df[ppm_cols] = df[ppm_cols].apply(_clean_numeric)
    total_ppm = _clean_numeric(df["PPM"]) if "PPM" in df.columns else pd.Series(dtype=float)
    total_counts = _clean_numeric(df["TOTAL"]) if "TOTAL" in df.columns else pd.Series(dtype=float)
    return {
        "period_col": period_col,
        "df": df,
        "count_cols": count_cols,
        "ppm_cols": ppm_cols,
        "total_ppm": total_ppm,
        "total_counts": total_counts,
        "mix": {c: float(df[c].sum()) for c in count_cols if c in df.columns},
        "ppm_by_col": {c: float(df[c].mean()) for c in ppm_cols if c in df.columns},
    }


def summarize_defects(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or df.empty:
        return {}
    process_col = df.columns[2]
    category_col = df.columns[7]
    use_col = df.columns[10]
    press_rows = df[df[process_col].astype(str).str.contains("PRESS", case=False, na=False)].copy()
    return {
        "process_col": process_col,
        "category_col": category_col,
        "use_col": use_col,
        "process_counts": df[process_col].value_counts(dropna=False),
        "category_counts": df[category_col].value_counts(dropna=False),
        "use_counts": df[use_col].value_counts(dropna=False),
        "press_rows": press_rows,
    }


def summarize_press_log(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {}
    num = df.copy()
    if "Data Time" in num.columns:
        num["Data Time"] = pd.to_datetime(num["Data Time"])
    for c in num.columns:
        if c != "Data Time":
            num[c] = pd.to_numeric(num[c], errors="coerce")
    key_cols = [c for c in ["6HPPRESSPV", "6HPPRESSSV", "6FHPPRESSPV", "6FHPPRESSSV", "6VACUUM", "6HPTEMPSV"] if c in num.columns]
    pt_cols = [c for c in ["6PT1", "6PT2", "6PT3", "6PT4", "6PT5", "6PT6", "6PT7", "6PT8", "6PT9"] if c in num.columns]
    corr = num[key_cols + pt_cols].corr() if len(key_cols) >= 2 else pd.DataFrame()
    return {
        "df": num,
        "key_cols": key_cols,
        "pt_cols": pt_cols,
        "corr": corr,
        "start": num["Data Time"].min() if "Data Time" in num.columns else None,
        "end": num["Data Time"].max() if "Data Time" in num.columns else None,
        "interval_s": float(num["Data Time"].diff().dt.total_seconds().median()) if "Data Time" in num.columns and len(num) > 1 else None,
    }


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>PCB Press 품질 인텔리전스 대시보드</h1>
          <p>고객사 품질 현황 · PRESS 공정 상관성 · 추가 요청 데이터까지 한 화면에서 정리하는 분석 UI</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi(label: str, value: str, delta: str = "") -> None:
    st.markdown(
        f"""
        <div class="kpi-box">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-delta">{delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_data_request_table() -> None:
    request_df = pd.DataFrame(REQUEST_ITEMS)
    st.dataframe(request_df, use_container_width=True, hide_index=True)


def render_quality_section(summary: dict[str, Any]) -> None:
    if not summary:
        st.info("품질 요약 데이터를 찾지 못했습니다. `data/raw`에 월간 품질 파일을 넣거나 업로드해 주세요.")
        return

    df = summary["df"]
    period_col = summary["period_col"]
    count_cols = summary["count_cols"]

    total_ppm = summary["total_ppm"]
    total_counts = summary["total_counts"]
    periods = df[period_col].astype(str)

    top_count = max(summary["mix"].items(), key=lambda kv: kv[1]) if summary["mix"] else ("-", 0)
    latest_ppm = float(total_ppm.iloc[-1]) if len(total_ppm) else 0.0
    first_ppm = float(total_ppm.iloc[0]) if len(total_ppm) else 0.0
    yoy = ((latest_ppm - first_ppm) / first_ppm * 100.0) if first_ppm else 0.0
    total_sum = float(total_counts.sum()) if len(total_counts) else 0.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi("총 PPM(최신)", f"{latest_ppm:,.1f}", f"기준 대비 {yoy:+.1f}%")
    with c2:
        render_kpi("총 불량수", _fmt_int(total_sum), f"상위 유형: {top_count[0]}")
    with c3:
        render_kpi("분석 기간", f"{len(df)}개", f"기준 컬럼: {period_col}")
    with c4:
        render_kpi("불량 유형 수", f"{len(count_cols)}개", "누적 구성비 기준")

    left, right = st.columns([1.05, 1])
    with left:
        fig = go.Figure()
        if len(total_ppm):
            fig.add_trace(go.Scatter(x=periods, y=total_ppm, mode="lines+markers", name="TOTAL PPM", line=dict(width=3, color="#2563eb")))
        fig.update_layout(
            title="기간별 총 PPM 추이",
            template="plotly_white",
            height=380,
            margin=dict(l=10, r=10, t=55, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        mix = pd.DataFrame([(k, v) for k, v in summary["mix"].items()], columns=["불량유형", "수량"])
        mix = mix.sort_values("수량", ascending=False)
        fig2 = px.pie(mix, names="불량유형", values="수량", hole=0.55, title="3개년 누적 불량 구성비")
        fig2.update_layout(template="plotly_white", height=380, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 연도별 불량 상세")
    show_cols = [period_col] + count_cols + [c for c in ["TOTAL", "PPM"] if c in df.columns]
    st.dataframe(df[show_cols], use_container_width=True, hide_index=True)


def render_defect_section(summary: dict[str, Any]) -> None:
    if not summary:
        st.info("불량 마스터를 찾지 못했습니다. `data/raw`에 불량 유형 파일을 넣거나 업로드해 주세요.")
        return

    process_counts = summary["process_counts"]
    category_counts = summary["category_counts"]
    press_rows = summary["press_rows"]

    left, center, right = st.columns([1.1, 1.1, 1])
    with left:
        proc_df = process_counts.rename_axis("공정명").reset_index(name="건수").sort_values("건수", ascending=False).head(12)
        fig = px.bar(proc_df, x="건수", y="공정명", orientation="h", title="공정별 불량 등록 수")
        fig.update_layout(template="plotly_white", height=420, margin=dict(l=10, r=10, t=55, b=10), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
    with center:
        cat_df = category_counts.rename_axis("불량구분").reset_index(name="건수")
        fig2 = px.pie(cat_df, names="불량구분", values="건수", hole=0.58, title="제조불량 vs 검사불량")
        fig2.update_layout(template="plotly_white", height=420, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig2, use_container_width=True)
    with right:
        st.markdown("### PRESS 관련 항목")
        st.caption("공정 이상 원인으로 직접 추적할 수 있는 항목")
        if not press_rows.empty:
            st.dataframe(press_rows.iloc[:, [3, 4, 2, 7, 10]].rename(columns={press_rows.columns[3]: "코드", press_rows.columns[4]: "명칭", press_rows.columns[2]: "공정", press_rows.columns[7]: "구분", press_rows.columns[10]: "사용여부"}), use_container_width=True, hide_index=True)
        else:
            st.info("PRESS 관련 행이 없습니다.")

    st.markdown("### 고객사 불량 마스터 요약")
    st.write("- 제조불량 / 검사불량 분포와 공정별 불량 밀도를 함께 보면, PRESS 전후 공정의 기여도를 빠르게 파악할 수 있습니다.")
    st.write("- `PRESS` 항목은 수는 적지만, 원인성 높은 제어 항목(압력/온도/진공/설비/프로그램)을 포함합니다.")


def render_press_section(press_logs: dict[str, pd.DataFrame]) -> None:
    if not press_logs:
        st.info("PRESS 로그가 없습니다. `data/raw`의 MDB 파일을 업로드하거나 배치해 주세요.")
        return

    selected = st.selectbox("분석할 PRESS 로그", list(press_logs.keys()))
    summary = summarize_press_log(press_logs[selected])
    if not summary:
        st.warning("선택한 로그에서 요약 가능한 데이터가 없습니다.")
        return

    df = summary["df"]
    key_cols = summary["key_cols"]
    pt_cols = summary["pt_cols"]

    c1, c2, c3 = st.columns(3)
    with c1:
        render_kpi("데이터 포인트", _fmt_int(len(df)), f"샘플 간격 {summary['interval_s']:.0f}초" if summary["interval_s"] else "")
    with c2:
        render_kpi("시작 시각", str(summary["start"]), "")
    with c3:
        render_kpi("종료 시각", str(summary["end"]), "")

    if key_cols:
        corr = summary["corr"]
        heat = corr.loc[key_cols, key_cols]
        fig = px.imshow(
            heat,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="PRESS 핵심 변수 상관관계",
        )
        fig.update_layout(template="plotly_white", height=520, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    if "Data Time" in df.columns:
        ts_cols = [c for c in key_cols[:3] if c in df.columns]
        if ts_cols:
            line_df = df[["Data Time"] + ts_cols].melt("Data Time", var_name="변수", value_name="값")
            fig2 = px.line(line_df, x="Data Time", y="값", color="변수", title="주요 PRESS 변수 시계열")
            fig2.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 변수 범위 요약")
    rng_rows = []
    for c in key_cols + pt_cols:
        s = df[c]
        rng_rows.append({"변수": c, "min": float(s.min()), "max": float(s.max()), "mean": float(s.mean()), "std": float(s.std())})
    st.dataframe(pd.DataFrame(rng_rows), use_container_width=True, hide_index=True)

    st.caption("해석 포인트: 압력 변수끼리는 거의 동조, 온도는 6HPTEMPSV와 PT 채널이 강하게 연결, 진공은 특정 구간에서 압력과 반대 방향으로 움직이는 경향이 나타납니다.")


def render_request_section() -> None:
    st.markdown("### 고객사 추가 요청 데이터 정리")
    st.write("아래 데이터가 들어오면 일별/주간/월별 품질 분석과 PRESS 공정 이상 탐지가 훨씬 정교해집니다.")
    render_data_request_table()

    st.markdown("### 왜 필요한가")
    cols = st.columns(2)
    left_points = [
        "일/주/월 품질 현황은 추세와 계절성, 품질 급변 구간을 찾는 데 가장 중요합니다.",
        "가동/비가동 및 설비 가동률은 불량이 발생한 시점의 설비 상태를 설명합니다.",
        "알람 로그는 불량의 직전 전조 신호를 학습시키는 핵심 라벨 역할을 합니다.",
        "레시피/설정값 이력은 PRESS 공정의 원인-결과 모델과 조건 최적화에 필요합니다.",
    ]
    right_points = [
        "LOT/PANEL/CYCLE 키가 있어야 품질, 설비, 알람, 검사 결과를 한 번에 조인할 수 있습니다.",
        "유지보수/교정 이력은 모델이 '장비 노후'와 '일시 이상'을 구분하는 데 도움을 줍니다.",
        "작업자/교대/라인 정보는 사람 또는 운영 조건에 따른 변동성을 설명합니다.",
        "원자재/환경 데이터는 외생 요인을 보정해 오탐을 줄입니다.",
    ]
    with cols[0]:
        for item in left_points:
            st.markdown(f"- {item}")
    with cols[1]:
        for item in right_points:
            st.markdown(f"- {item}")

    st.markdown("### 고객사 요청 메시지 예시")
    st.code(
        """안녕하세요.\n\n추가 분석을 위해 아래 데이터 제공을 요청드립니다.\n1) 일별/주간/월별 품질 현황\n2) 설비 가동/비가동 현황 및 공정별 가동률\n3) PRESS 공정 알람 이력 및 레시피/설정값 변화 이력\n4) LOT/PANEL/CYCLE 매핑 키\n5) 유지보수/교정 이력, 교대/작업자 정보\n\n제공 시점/컬럼 정의가 있으면 함께 전달 부탁드립니다.\n감사합니다.""",
        language="text",
    )


def main() -> None:
    st.set_page_config(
        page_title="PCB 품질 인텔리전스 대시보드",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("품질 분석 옵션")
    st.sidebar.caption("로컬 RAW 파일을 기본으로 읽고, 필요하면 업로드 파일로 덮어쓸 수 있습니다.")
    use_local = st.sidebar.toggle("로컬 `data/raw` 사용", value=True)
    st.sidebar.markdown("---")
    quality_upload = st.sidebar.file_uploader("월별/주별/일별 품질 파일", type=["xls", "xlsx", "csv"], key="quality")
    defect_upload = st.sidebar.file_uploader("불량 유형 파일", type=["xls", "xlsx", "csv"], key="defect")
    mdb_upload = st.sidebar.file_uploader("PRESS 로그 파일 (.mdb)", type=["mdb"], key="mdb")

    quality_df = load_quality_summary(None if use_local and quality_upload is None else quality_upload)
    defect_df = load_defect_master(None if use_local and defect_upload is None else defect_upload)
    press_logs = load_press_logs(None if use_local and mdb_upload is None else mdb_upload)

    bundle = LocalDataBundle(
        defect_master=defect_df,
        quality_summary=quality_df,
        press_logs=press_logs,
        source_notes=[
            f"품질 파일: {'로컬 RAW' if use_local and quality_upload is None else (quality_upload.name if quality_upload else '없음')}",
            f"불량 파일: {'로컬 RAW' if use_local and defect_upload is None else (defect_upload.name if defect_upload else '없음')}",
            f"PRESS 로그: {'로컬 RAW' if use_local and mdb_upload is None else (mdb_upload.name if mdb_upload else '없음')}",
        ],
    )

    st.sidebar.markdown("### 현재 데이터 소스")
    for note in bundle.source_notes:
        st.sidebar.markdown(f"- {note}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 빠른 체크")
    st.sidebar.markdown(f"- 품질 요약: **{'있음' if quality_df is not None else '없음'}**")
    st.sidebar.markdown(f"- 불량 마스터: **{'있음' if defect_df is not None else '없음'}**")
    st.sidebar.markdown(f"- PRESS 로그: **{len(press_logs)}개 파일**")

    render_hero()

    st.markdown(
        """
        <div class="small-note">
          이 대시보드는 고객사로부터 받을 <b>일별/주간/월별 품질 현황</b>, <b>설비 가동·비가동</b>, <b>공정별 가동률</b>, <b>알람 내역</b>을 함께 조합해
          PRESS 공정과 불량률의 상관관계를 읽기 쉽게 보여주는 것을 목표로 합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    quality_summary = summarize_quality(quality_df) if quality_df is not None else {}
    defect_summary = summarize_defects(defect_df) if defect_df is not None else {}

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi("품질 파일", "로드됨" if quality_df is not None else "대기", f"행 수: {_fmt_int(len(quality_df)) if quality_df is not None else '-'}")
    with k2:
        render_kpi("불량 마스터", "로드됨" if defect_df is not None else "대기", f"행 수: {_fmt_int(len(defect_df)) if defect_df is not None else '-'}")
    with k3:
        render_kpi("PRESS 로그", _fmt_int(len(press_logs)), "MDB 기반 시계열")
    with k4:
        render_kpi("요청 데이터", f"{len(REQUEST_ITEMS)}개", "고객사 전달용 체크리스트")

    st.markdown("---")
    tabs = st.tabs(["품질 요약", "불량 유형", "PRESS 상관성", "고객사 요청 데이터"])
    with tabs[0]:
        render_quality_section(quality_summary)
    with tabs[1]:
        render_defect_section(defect_summary)
    with tabs[2]:
        render_press_section(press_logs)
    with tabs[3]:
        render_request_section()

    st.markdown("---")
    st.caption("권장 실행: streamlit run scripts/quality_intelligence_ui.py")


if __name__ == "__main__":
    main()


