"""Local Streamlit dashboard for the semiconductor PCB lamination project.

Run with:
  streamlit run scripts/ui.py

The UI shows:
 - project goals and milestone summary
 - dataset / report / implementation readiness
 - quick actions to run downloads, EDA, and DVC stages
 - next recommended work items

Run with:
  streamlit run scripts/ui.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol, cast

# Set up large file upload support (10GB limit)
# This MUST be set before any streamlit imports
os.environ.setdefault("STREAMLIT_SERVER_MAX_UPLOAD_SIZE", "10000")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

class _ColumnLike(Protocol):
    def __enter__(self) -> Any: ...

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool | None: ...


class _FormLike(Protocol):
    def __enter__(self) -> Any: ...

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool | None: ...


class StreamlitAPI(Protocol):
    session_state: dict[str, Any]
    sidebar: Any

    def title(self, body: str) -> Any: ...

    def header(self, body: str) -> Any: ...

    def subheader(self, body: str) -> Any: ...

    def markdown(self, body: str) -> Any: ...

    def caption(self, body: str) -> Any: ...

    def write(self, body: Any) -> Any: ...

    def code(self, body: str) -> Any: ...

    def info(self, body: str) -> Any: ...

    def success(self, body: str) -> Any: ...

    def error(self, body: str) -> Any: ...

    def warning(self, body: str) -> Any: ...

    def json(self, body: Any) -> Any: ...

    def text(self, body: Any) -> Any: ...

    def progress(self, value: float) -> Any: ...

    def button(self, label: str) -> bool: ...

    def rerun(self) -> None: ...

    def dataframe(self, data: Any, *, use_container_width: bool = False) -> Any: ...

    def expander(self, label: str, expanded: bool = False) -> _FormLike: ...

    def columns(self, n: int) -> list[_ColumnLike]: ...

    def form(self, key: str) -> _FormLike: ...

    def file_uploader(self, label: str, *, type: list[str]) -> Any: ...

    def text_input(self, label: str, *, value: str) -> str: ...

    def slider(self, label: str, *, min_value: int, max_value: int, value: int, step: int) -> int: ...

    def form_submit_button(self, label: str) -> bool: ...


st: StreamlitAPI
try:
    import streamlit as streamlit_module
    st = cast(StreamlitAPI, streamlit_module)
except ModuleNotFoundError:  # pragma: no cover - test environment fallback
    class _StreamlitFallback:
        session_state: dict[str, Any] = {}

        def __getattr__(self, name: str):
            raise ModuleNotFoundError("streamlit is required to run the web UI")

    st = cast(StreamlitAPI, _StreamlitFallback())

from src.data.dataset_inspector import DatasetAnalysis, analyze_dataset_source, save_analysis_report
from src.data.preprocess import preprocess_tabular_dataset
from src.research.guide import build_up_items, glossary_terms, paper_insights, project_direction, security_principles

from src.research.bibliography import (
    list_notes,
    read_note,
    save_note,
    list_note_backups,
    restore_note_backup,
)

DATA_RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
INTERIM_UPLOADS = ROOT / "data" / "interim" / "uploads"

SECOM_OUT = DATA_RAW / "secom"
DEEPPCB_OUT = DATA_RAW / "deeppcb"
BOSCH_OUT = DATA_RAW / "bosch"

SECOM_REPORT = REPORTS / "eda_secom.md"
DEEPPCB_REPORT = REPORTS / "eda_deeppcb.md"


@dataclass(frozen=True)
class ArtifactStatus:
    name: str
    path: Path
    ready: bool
    note: str = ""


@dataclass(frozen=True)
class DashboardState:
    goals: list[str]
    artifacts: list[ArtifactStatus]
    ready_count: int
    total_count: int
    progress_pct: int
    next_steps: list[str]
    summary: str


@dataclass(frozen=True)
class ConnectionInfo:
    local_url: str
    public_url: str | None


def run_cmd(cmd: list[str]) -> tuple[int, str]:
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return res.returncode, (res.stdout + "\n" + res.stderr).strip()
    except Exception as e:
        return 1, str(e)


def file_status(p: Path) -> bool:
    return p.exists()


def _artifact(name: str, path: Path, note: str = "") -> ArtifactStatus:
    return ArtifactStatus(name=name, path=path, ready=file_status(path), note=note)


def _artifact_rows(artifacts: Iterable[ArtifactStatus]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in artifacts:
        rows.append(
            {
                "상태": "✅" if item.ready else "⬜",
                "항목": item.name,
                "경로": str(item.path),
                "비고": item.note or "",
            }
        )
    return rows


def _progress_pct(items: Iterable[ArtifactStatus]) -> tuple[int, int, int]:
    items_list = list(items)
    total = len(items_list)
    ready = sum(1 for item in items_list if item.ready)
    pct = int(round((ready / total) * 100)) if total else 0
    return ready, total, pct


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def _read_public_url(root: Path = ROOT) -> str | None:
    file_path = root / "logs" / "public_url.txt"
    if not file_path.exists():
        return None
    value = file_path.read_text(encoding="utf-8").strip()
    return value or None


def build_connection_info(root: Path = ROOT, port: int = 8501) -> ConnectionInfo:
    return ConnectionInfo(local_url=f"http://127.0.0.1:{port}", public_url=_read_public_url(root))


def show_connection_info(info: ConnectionInfo) -> None:
    st.header("접속 정보")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**로컬 접속**")
        st.code(info.local_url)
        st.caption("같은 PC에서 확인할 때 사용합니다.")
    with col2:
        st.markdown("**외부 접속**")
        if info.public_url:
            st.code(info.public_url)
            st.success("외부 URL이 감지되었습니다.")
            st.caption("`scripts/start_public_ui.ps1`가 생성한 링크입니다.")
        else:
            st.info("외부 URL이 아직 없습니다. `scripts/start_public_ui.ps1`로 생성하면 자동 표시됩니다.")


def show_sidebar_summary(state: DashboardState, connection_info: ConnectionInfo) -> None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 빠른 현황")
    st.sidebar.markdown(f"- 준비도: **{state.ready_count}/{state.total_count}**")
    st.sidebar.markdown(f"- 진행률: **{state.progress_pct}%**")
    st.sidebar.markdown(f"- 로컬: `{connection_info.local_url}`")
    if connection_info.public_url:
        st.sidebar.markdown(f"- 외부: `{connection_info.public_url}`")
    else:
        st.sidebar.markdown("- 외부: 대기 중")


def _analyze_uploaded_file(uploaded_file) -> DatasetAnalysis:
    suffix = Path(uploaded_file.name).suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_path = Path(tmp.name)
    try:
        return analyze_dataset_source(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


def _persist_uploaded_file(uploaded_file) -> Path:
    INTERIM_UPLOADS.mkdir(parents=True, exist_ok=True)
    safe_name = Path(uploaded_file.name).name
    target = INTERIM_UPLOADS / safe_name
    target.write_bytes(uploaded_file.getbuffer())
    return target


def build_dashboard_state(root: Path = ROOT) -> DashboardState:
    data_raw = root / "data" / "raw"
    reports = root / "reports"
    scripts_dir = root / "scripts"

    goals = [
        "공개 데이터(SECOM, DeepPCB)를 활용해 사전학습/베이스라인을 안정화한다.",
        "반도체 PCB 적층 공정 실데이터가 오면 Press 사이클, P013, P019 라벨을 정합한다.",
        "MS-CDPNet 구조를 통해 멀티모달 융합 + 인과 전파 예측 + 설명 가능성을 확보한다.",
        "평가 지표(F1, AUROC, FAR@Recall, Cost-Aware)를 표준화하고 리포트한다.",
    ]

    artifacts = [
        _artifact("SECOM 데이터", data_raw / "secom", "공개 반도체 공정 데이터"),
        _artifact("DeepPCB 데이터", data_raw / "deeppcb", "PCB 결함 이미지 데이터"),
        _artifact("Bosch 데이터", data_raw / "bosch", "Kaggle CLI 필요"),
        _artifact("SECOM 리포트", reports / "eda_secom.md", "EDA 결과"),
        _artifact("DeepPCB 리포트", reports / "eda_deeppcb.md", "EDA 결과"),
        _artifact("데이터 다운로드 스크립트", scripts_dir / "download_datasets.py"),
        _artifact("학습 진입점", scripts_dir / "train.py"),
        _artifact("평가 진입점", scripts_dir / "eval.py"),
        _artifact("데이터 감사", scripts_dir / "audit.py"),
    ]

    ready_count, total_count, progress_pct = _progress_pct(artifacts)
    next_steps = [
        "업로드 또는 폴더 경로 기반 데이터 분석 결과를 바탕으로 전처리 템플릿을 자동화한다.",
        "`scripts/eda_secom.py`와 `scripts/eda_deeppcb.py`를 복구해 EDA 리포트를 다시 생성한다.",
        "`MLflow`를 학습 모듈과 `train.py`에 연결해 실험 추적을 시작한다.",
        "반도체 PCB 적층 공정에서 받은 사이클↔Panel/LOT 매칭 키를 반영해 실제 데이터 로더를 정식화한다.",
        "`Hydra` 실험 설정을 정리하고, 공개 데이터 베이스라인을 재현 가능하게 만든다.",
    ]

    summary = f"현재 상태: {ready_count}/{total_count}개 핵심 항목이 준비되었습니다."
    return DashboardState(
        goals=goals,
        artifacts=artifacts,
        ready_count=ready_count,
        total_count=total_count,
        progress_pct=progress_pct,
        next_steps=next_steps,
        summary=summary,
    )


def show_overview(state: DashboardState) -> None:
    st.header("개발 목표")
    for goal in state.goals:
        st.markdown(f"- {goal}")

    st.header("진행 현황")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("준비된 항목", f"{state.ready_count}/{state.total_count}")
    col2.metric("진행률", f"{state.progress_pct}%")
    col3.metric("다음 작업", f"{len(state.next_steps)}개")
    col4.metric("상태", "모니터링 중")

    st.progress(state.progress_pct / 100 if state.total_count else 0.0)
    st.caption(state.summary)


def show_status(state: DashboardState) -> None:
    st.header("데이터 / 리포트 / 스크립트 상태")
    cols = st.columns(3)
    visible = [
        ("SECOM", SECOM_OUT),
        ("DeepPCB", DEEPPCB_OUT),
        ("Bosch", BOSCH_OUT),
    ]
    for idx, (title, path) in enumerate(visible):
        with cols[idx]:
            st.write(f"**{title}**")
            st.code(str(path))
            if file_status(path):
                st.success("Ready")
            else:
                st.error("Missing")

    st.subheader("핵심 산출물")
    st.caption("상태를 표로 정리해 핵심 산출물을 한눈에 볼 수 있습니다.")
    st.dataframe(_artifact_rows(state.artifacts), use_container_width=True)


def show_dataset_analyzer() -> None:
    st.header("데이터셋 분석")
    st.caption("파일을 업로드하거나, 로컬 폴더 경로를 입력하면 자동으로 기본 분석을 수행합니다.")
    st.info("📂 **파일 업로드 제한**: 최대 10GB\n\n로컬 폴더 경로로 더 큰 데이터를 분석할 수도 있습니다.")

    with st.form("dataset_analysis_form"):
        uploaded = st.file_uploader(
            "분석할 파일 업로드 (최대 10GB)",
            type=["csv", "tsv", "txt", "data", "parquet", "pq", "xlsx", "xls", "png", "jpg", "jpeg", "bmp", "tif", "tiff", "zip"],
        )
        folder_input = st.text_input("또는 로컬 폴더 경로", value=str(DATA_RAW))
        max_rows = st.slider("미리보기 행 수", min_value=5, max_value=50, value=10, step=5)
        submitted = st.form_submit_button("분석 시작")

    if submitted:
        if uploaded is not None:
            source_path = _persist_uploaded_file(uploaded)
            analysis = analyze_dataset_source(source_path, max_preview_rows=max_rows)
        else:
            folder_path = Path(folder_input).expanduser()
            source_path = folder_path
            analysis = analyze_dataset_source(folder_path, max_preview_rows=max_rows)
        st.session_state["latest_dataset_analysis"] = analysis
        st.session_state["latest_dataset_source_path"] = str(source_path)

    analysis = st.session_state.get("latest_dataset_analysis")
    if not isinstance(analysis, DatasetAnalysis):
        st.info("파일을 업로드하거나 폴더 경로를 입력한 뒤 분석 시작을 눌러주세요.")
        return

    st.subheader("분석 요약")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("유형", analysis.source_type)
    col2.metric("파일 수", str(analysis.total_files))
    col3.metric("폴더 수", str(analysis.total_dirs))
    col4.metric("총 용량", _format_bytes(analysis.total_size_bytes))

    st.markdown(analysis.to_markdown())

    if analysis.preview_frame is not None:
        st.subheader("미리보기")
        st.dataframe(analysis.preview_frame, use_container_width=True)

    if analysis.dtype_frame is not None:
        st.subheader("컬럼 자료형")
        st.dataframe(analysis.dtype_frame, use_container_width=True)

    if analysis.numeric_summary is not None:
        st.subheader("수치형 요약 통계")
        st.dataframe(analysis.numeric_summary, use_container_width=True)

    if analysis.image_info is not None:
        st.subheader("이미지 정보")
        st.json(analysis.image_info)

    if analysis.archive_members:
        st.subheader("압축 파일 내부 항목")
        st.write(analysis.archive_members)

    if st.button("분석 리포트 저장"):
        saved = save_analysis_report(analysis, REPORTS / "analysis", report_name=analysis.source.stem or analysis.source_type)
        st.success("분석 리포트가 저장되었습니다.")
        st.json({key: str(path) for key, path in saved.items()})

    if st.button("전처리 실행"):
        source_path_str = st.session_state.get("latest_dataset_source_path")
        if not source_path_str:
            st.warning("먼저 파일 업로드 또는 폴더 경로 분석을 실행하세요.")
            return
        source_path = Path(source_path_str)
        try:
            result = preprocess_tabular_dataset(source_path, ROOT / "data" / "processed" / source_path.stem)
        except Exception as exc:
            st.error(f"전처리 실패: {exc}")
        else:
            st.success("전처리가 완료되었습니다.")
            st.json(
                {
                    "input_path": str(result.input_path),
                    "output_dir": str(result.output_dir),
                    "cleaned_csv": str(result.cleaned_csv),
                    "cleaned_parquet": str(result.cleaned_parquet),
                    "report_md": str(result.report_md),
                    "metadata_json": str(result.metadata_json),
                    "rows_before": result.rows_before,
                    "rows_after": result.rows_after,
                }
            )


def controls():
    st.header("Controls")
    st.caption("버튼을 누르면 다운로드/EDA/DVC 재실행을 시도합니다.")
    if st.button("Download datasets (SECOM + DeepPCB + Bosch)"):
        st.info("Running scripts/download_datasets.py --all (this may take time and require kaggle/git) ...")
        code, out = run_cmd(["python", str(ROOT / "scripts" / "download_datasets.py"), "--all"])
        if code == 0:
            st.success("Download script finished")
        else:
            st.error(f"Download script failed (exit {code})")
        st.text(out)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Run SECOM EDA"):
            st.info("Running scripts/eda_secom.py ...")
            code, out = run_cmd(["python", str(ROOT / "scripts" / "eda_secom.py")])
            if code == 0:
                st.success("SECOM EDA finished")
            else:
                st.error(f"SECOM EDA failed (exit {code})")
            st.text(out)
    with col2:
        if st.button("Run DeepPCB EDA"):
            st.info("Running scripts/eda_deeppcb.py ...")
            code, out = run_cmd(["python", str(ROOT / "scripts" / "eda_deeppcb.py")])
            if code == 0:
                st.success("DeepPCB EDA finished")
            else:
                st.error(f"DeepPCB EDA failed (exit {code})")
            st.text(out)

    st.markdown("---")
    if st.button("DVC repro (download_datasets)"):
        st.info("Running: dvc repro download_datasets")
        code, out = run_cmd(["dvc", "repro", "download_datasets"])
        if code == 0:
            st.success("DVC repro succeeded")
        else:
            st.error(f"DVC repro failed (exit {code})")
        st.text(out)

    if st.button("DVC repro (eda_secom)"):
        st.info("Running: dvc repro eda_secom")
        code, out = run_cmd(["dvc", "repro", "eda_secom"])
        if code == 0:
            st.success("DVC repro succeeded")
        else:
            st.error(f"DVC repro failed (exit {code})")
        st.text(out)

    if st.button("새로고침"):
        st.rerun()


def show_reports_preview():
    st.header("Reports preview")
    if SECOM_REPORT.exists():
        with st.expander("SECOM EDA", expanded=False):
            st.markdown(SECOM_REPORT.read_text(encoding="utf-8"))
    else:
        st.info("SECOM report not found. Run SECOM EDA to generate it.")

    if DEEPPCB_REPORT.exists():
        with st.expander("DeepPCB EDA", expanded=False):
            st.markdown(DEEPPCB_REPORT.read_text(encoding="utf-8"))
    else:
        st.info("DeepPCB report not found. Run DeepPCB EDA to generate it.")


def show_next_steps(state: DashboardState) -> None:
    st.header("다음 진행할 일")
    for idx, step in enumerate(state.next_steps, start=1):
        st.markdown(f"{idx}. {step}")


def show_notes() -> None:
    st.header("운영 메모")
    st.markdown(
        """
        - 이 대시보드는 로컬 개발용입니다.
        - 공개 데이터는 `data/raw/` 아래에 두고, 대용량 원본은 DVC로 관리합니다.
        - 반도체 PCB 적층 공정 실데이터가 도착하면 `cycle_id`, `panel_id`, `lot_id` 매칭부터 확정합니다.
        - 노트 저장 시 `paper/notes/.history/`에 자동 백업이 남습니다.
        """
    )


def render_home_page(state: DashboardState, connection_info: ConnectionInfo) -> None:
    st.title("반도체 PCB 적층 공정 - 개발 모니터링 웹")
    st.markdown("개발 목표, 진행률, 데이터 상태, 다음 작업을 한 화면에서 확인합니다.")
    show_connection_info(connection_info)
    st.markdown("---")
    show_overview(state)
    st.markdown("---")
    show_status(state)
    st.markdown("---")
    show_dataset_analyzer()
    st.markdown("---")
    controls()
    st.markdown("---")
    show_reports_preview()
    st.markdown("---")
    show_next_steps(state)
    st.markdown("---")
    show_notes()


def render_research_direction_page() -> None:
    direction = project_direction()
    st.title("연구 방향 / 논문 주제")
    st.markdown(direction.one_liner)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("우리가 풀고 싶은 문제")
        st.markdown(direction.problem)
        st.subheader("왜 중요한가")
        st.markdown(direction.why_it_matters)
    with col2:
        st.subheader("제안하는 방향")
        st.markdown(direction.proposed_solution)
        st.subheader("지금 범위")
        st.markdown(direction.scope_for_now)

    st.subheader("기대 기여")
    st.markdown(direction.expected_contribution)

    st.subheader("보안 원칙")
    for item in direction.security_notes:
        st.markdown(f"- {item}")

    st.subheader("후속 아이디어")
    for item in direction.next_ideas:
        st.markdown(f"- {item}")


def render_references_page() -> None:
    st.title("참고 자료 / SCI 분석")
    st.markdown("아래는 논문 빌드업을 위한 핵심 참고 논문과, 각 논문이 가진 약점·보완 방향입니다.")

    st.subheader("논문 빌드업 목록")
    for item in build_up_items():
        with st.expander(item.title, expanded=False):
            st.markdown(f"- 왜 필요한가: {item.why_needed}")
            st.markdown(f"- 산출물: {item.deliverable}")

    st.subheader("핵심 SCI/상위 학회 논문 분석")
    for paper in paper_insights():
        with st.expander(f"{paper.title} ({paper.year})", expanded=False):
            st.markdown(f"- 저자: {paper.authors}")
            st.markdown(f"- 게재처: {paper.venue}")
            st.markdown(f"- 무엇을 하는가: {paper.what_it_does}")
            st.markdown(f"- 분석 기법: {paper.technique}")
            st.markdown(f"- 강점: {paper.strength}")
            st.markdown(f"- 약점: {paper.weakness}")
            st.markdown(f"- 우리는 어떻게 보완하는가: {paper.our_response}")

    # --- Bibliography controls and generated notes preview
    st.markdown("---")
    st.subheader("자동 생성된 논문 노트 및 BibTeX")
    notes_dir = ROOT / "paper" / "notes"
    bib_path = ROOT / "paper" / "references.bib"

    col_a, col_b = st.columns([2, 1])
    with col_a:
        md_files = list_notes(notes_dir)
        st.markdown(f"생성된 노트: {len(md_files)}개")
        if md_files:
            selected = st.selectbox("노트 선택", ["--선택--"] + md_files)
            if selected and selected != "--선택--":
                if st.button("열기"):
                    try:
                        content = read_note(selected, notes_dir)
                    except Exception as exc:
                        st.error(f"파일을 열 수 없습니다: {exc}")
                    else:
                        st.code(content)

                if "edit_note" not in st.session_state:
                    st.session_state["edit_note"] = None

                if st.button("편집 시작"):
                    try:
                        st.session_state["edit_note"] = read_note(selected, notes_dir)
                        st.session_state["edit_note_name"] = selected
                    except Exception as exc:
                        st.error(f"편집을 시작할 수 없습니다: {exc}")

                edit_content = st.session_state.get("edit_note")
                if edit_content is not None and st.session_state.get("edit_note_name") == selected:
                    new_text = st.text_area("편집", value=edit_content, height=420)
                    if st.button("저장"):
                        try:
                            save_note(selected, new_text, notes_dir)
                        except Exception as exc:
                            st.error(f"저장 실패: {exc}")
                        else:
                            st.success("저장되었습니다.")
                            st.session_state["edit_note"] = None

                    with st.expander("변경 이력 보기"):
                        try:
                            backups = list_note_backups(selected, notes_dir)
                        except Exception as exc:
                            st.error(f"이력 조회 실패: {exc}")
                            backups = []

                        if not backups:
                            st.info("백업 파일이 없습니다.")
                        else:
                            chosen = st.selectbox("복원할 백업 선택", ["--선택--"] + backups)
                            if chosen and chosen != "--선택--":
                                if st.button("복원"):
                                    try:
                                        restored = restore_note_backup(selected, chosen, notes_dir)
                                    except Exception as exc:
                                        st.error(f"복원 실패: {exc}")
                                    else:
                                        st.success(f"복원 완료: {restored.name}")
                                        # Refresh the edit buffer with restored content
                                        try:
                                            st.session_state["edit_note"] = read_note(selected, notes_dir)
                                        except Exception:
                                            st.session_state["edit_note"] = None
        else:
            st.info("논문 노트 디렉토리가 없습니다. 먼저 자동생성 스크립트를 실행하세요.")

    with col_b:
        if bib_path.exists():
            st.markdown(f"BibTeX: `{bib_path.name}`")
            if st.button("BibTeX 파일 열기"):
                st.code(bib_path.read_text(encoding="utf-8"))
        else:
            st.info("BibTeX 파일이 없습니다.")

        if st.button("Regenerate bibliography (generate_bibliography.py)"):
            st.info("참고문헌 및 노트 자동생성을 실행합니다. 완료되면 페이지를 새로고침하세요.")
            code, out = run_cmd(["python", str(ROOT / "scripts" / "generate_bibliography.py"), "--overwrite"])
            if code == 0:
                st.success("생성 완료")
            else:
                st.error(f"생성 실패 (exit {code})")
            st.text(out)


def render_glossary_page() -> None:
    st.title("용어 사전")
    st.markdown("기획자가 바로 읽을 수 있도록, 어려운 용어를 짧고 쉬운 말로 정리했습니다.")
    for term in glossary_terms():
        with st.expander(term.term, expanded=False):
            st.markdown(f"- 정의: {term.short_definition}")
            st.markdown(f"- 예시: {term.easy_example}")


def render_security_page() -> None:
    st.title("보안 / 운영 원칙")
    st.markdown("산업 데이터는 보안이 연구만큼 중요합니다. 아래 원칙은 앞으로도 계속 지켜야 합니다.")
    for rule in security_principles():
        st.subheader(rule.title)
        st.markdown(rule.description)
        st.info(rule.practical_rule)


def render_pages(page: str, state: DashboardState, connection_info: ConnectionInfo) -> None:
    if page == "대시보드":
        render_home_page(state, connection_info)
    elif page == "연구 방향":
        render_research_direction_page()
    elif page == "참고 자료":
        render_references_page()
    elif page == "용어 사전":
        render_glossary_page()
    elif page == "보안 / 운영 원칙":
        render_security_page()
    else:
        render_home_page()


def main():
    st.set_page_config(page_title="Semiconductor PCB Lamination Thesis", page_icon="🧪", layout="wide")
    state = build_dashboard_state()
    connection_info = build_connection_info()
    page = st.sidebar.radio(
        "페이지",
        ["대시보드", "연구 방향", "참고 자료", "용어 사전", "보안 / 운영 원칙"],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("기획자용 안내: 각 페이지는 연구 방향, 참고 자료, 용어, 보안 원칙을 따로 볼 수 있게 나눴습니다.")
    show_sidebar_summary(state, connection_info)
    # Optional simple access control: set environment variable PCB_UI_TOKEN to require a token
    token_required = os.environ.get("PCB_UI_TOKEN")
    if token_required:
        provided = st.sidebar.text_input("Access token", value="", type="password")
        if provided != token_required:
            st.sidebar.error("토큰이 올바르지 않습니다. 접속 권한이 필요합니다.")
            return
    render_pages(page, state, connection_info)


if __name__ == "__main__":
    main()

