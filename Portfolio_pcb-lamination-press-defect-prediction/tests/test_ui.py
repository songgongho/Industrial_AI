from pathlib import Path

from scripts.ui import (
    ArtifactStatus,
    ConnectionInfo,
    DashboardState,
    _artifact_rows,
    build_connection_info,
    build_dashboard_state,
)


def test_build_dashboard_state_reports_expected_progress(tmp_path: Path) -> None:
    root = tmp_path
    scripts_dir = root / "scripts"
    data_raw = root / "data" / "raw"
    reports = root / "reports"
    scripts_dir.mkdir(parents=True)
    (data_raw / "secom").mkdir(parents=True)
    (data_raw / "deeppcb").mkdir(parents=True)
    (reports).mkdir(parents=True)
    (scripts_dir / "download_datasets.py").write_text("print('ok')", encoding="utf-8")
    (scripts_dir / "train.py").write_text("print('ok')", encoding="utf-8")
    (scripts_dir / "eval.py").write_text("print('ok')", encoding="utf-8")
    (scripts_dir / "audit.py").write_text("print('ok')", encoding="utf-8")
    (reports / "eda_secom.md").write_text("# SECOM", encoding="utf-8")

    state = build_dashboard_state(root)

    assert isinstance(state, DashboardState)
    assert state.ready_count >= 5
    assert state.total_count >= state.ready_count
    assert 0 <= state.progress_pct <= 100
    assert any("SECOM" in goal for goal in state.goals)
    assert any(item.name == "SECOM 데이터" and item.ready for item in state.artifacts)
    assert any(item.name == "DeepPCB 데이터" and item.ready for item in state.artifacts)
    assert any(item.name == "SECOM 리포트" and item.ready for item in state.artifacts)


def test_build_dashboard_state_marks_missing_items(tmp_path: Path) -> None:
    root = tmp_path
    (root / "scripts").mkdir(parents=True)
    state = build_dashboard_state(root)

    missing = {item.name for item in state.artifacts if not item.ready}
    assert "SECOM 데이터" in missing
    assert "DeepPCB 데이터" in missing
    assert "Bosch 데이터" in missing


def test_build_connection_info_reads_public_url_file(tmp_path: Path) -> None:
    root = tmp_path
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True)
    (logs_dir / "public_url.txt").write_text("https://example.ngrok-free.dev\n", encoding="utf-8")

    info = build_connection_info(root, port=8501)

    assert isinstance(info, ConnectionInfo)
    assert info.local_url == "http://127.0.0.1:8501"
    assert info.public_url == "https://example.ngrok-free.dev"


def test_build_connection_info_without_public_url(tmp_path: Path) -> None:
    info = build_connection_info(tmp_path, port=8501)

    assert info.local_url == "http://127.0.0.1:8501"
    assert info.public_url is None


def test_artifact_rows_render_status_and_notes() -> None:
    rows = _artifact_rows(
        [
            ArtifactStatus(name="SECOM", path=Path("data/raw/secom"), ready=True, note="ready"),
            ArtifactStatus(name="DeepPCB", path=Path("data/raw/deeppcb"), ready=False, note=""),
        ]
    )

    assert rows == [
        {"상태": "✅", "항목": "SECOM", "경로": str(Path("data/raw/secom")), "비고": "ready"},
        {"상태": "⬜", "항목": "DeepPCB", "경로": str(Path("data/raw/deeppcb")), "비고": ""},
    ]


