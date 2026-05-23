from pathlib import Path

from tools import (
    analyze_history,
    assess_severity,
    build_markdown_report,
    infer_causes,
    parse_input_file,
    suggest_actions,
)


def main() -> None:
    data = parse_input_file(Path("sample_input.json"))
    stats = analyze_history(data.historical_data)
    severity = assess_severity(data.anomaly_score, stats["change_rate"])
    causes = infer_causes(data.feature_importance, stats)
    actions = suggest_actions(severity)
    report = build_markdown_report(data, stats, severity, causes, actions)

    assert "이상탐지 보고서" in report
    assert data.sensor_id in report
    print("smoke test passed")


if __name__ == "__main__":
    main()

