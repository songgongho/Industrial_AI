# 스마트팩토리 이상탐지 해석 에이전트 구현
# 역할: 제조 설비/공정 데이터의 이상탐지 모델 결과를 입력받아,
#       왜 이상으로 판단됐는지 원인 분석, 심각도 평가, 대응 제안, 보고서 초안을 생성하는 Python CLI 에이전트.
# 입력 형식 (JSON 또는 pandas DataFrame):
# {
#   "timestamp": "2026-04-06 09:00",
#   "anomaly_score": 0.92,  # 0~1, >0.8 고위험
#   "feature_importance": {"sensor_temp": 0.45, "vibration": 0.32, "pressure": 0.23},
#   "historical_data": [0.1, 0.2, 0.3, 0.5, 0.9],  # 최근 점수 시계열
#   "sensor_id": "Line1_SENSOR_A"
# }
# 기능 단계:
# 1. 입력 검증 및 데이터 로드 (pandas 사용)
# 2. 이상 원인 분석: 중요 feature별 해석
# 3. 심각도 분류: Low/Medium/High (score + 변화율 기준)
# 4. 대응 제안: 실무형 액션 생성
# 5. 보고서 생성: 마크다운 테이블 + 3문장 요약
# 출력 형식: 한국어 마크다운 보고서
# 예시 1 입력: anomaly_score=0.92, feature_importance={"temp":0.45}
# 예시 1 출력: High 위험, 온도 과열 원인, 즉시 점검 권고
# 예시 2 입력: anomaly_score=0.65, feature_importance={"vib":0.60}
# 예시 2 출력: Medium, 진동 패턴 변화, 모니터링 지속
# 구현 제약:
# - pandas로 historical 분석 (평균, 변화율)
# - LLM 해석 함수는 도구 함수로 분리 (openai/ollama 선택)
# - argparse 기반 CLI: python main.py --input sample_input.json
# - 예외처리/로그 메시지 포함

from __future__ import annotations

import argparse
from pathlib import Path

from tools import (
    analyze_history,
    assess_severity,
    build_markdown_report,
    infer_causes,
    optional_llm_interpretation,
    parse_input_file,
    suggest_actions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="스마트팩토리 이상탐지 결과를 한국어 보고서로 해석합니다."
    )
    parser.add_argument("--input", required=True, help="입력 파일 경로 (json/csv)")
    parser.add_argument("--output", help="보고서 저장 경로 (.md)")
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="LLM 해석을 추가합니다 (provider/model 필요)",
    )
    parser.add_argument(
        "--llm-provider",
        default="ollama",
        choices=["ollama", "openai"],
        help="LLM 제공자 선택",
    )
    parser.add_argument("--llm-model", default="llama3.1", help="LLM 모델명")
    return parser.parse_args()


def run() -> str:
    args = parse_args()
    anomaly_input = parse_input_file(Path(args.input))

    stats = analyze_history(anomaly_input.historical_data)
    severity = assess_severity(anomaly_input.anomaly_score, stats["change_rate"])
    causes = infer_causes(anomaly_input.feature_importance, stats)
    actions = suggest_actions(severity)

    llm_text = None
    if args.use_llm:
        llm_text = optional_llm_interpretation(
            feature_importance=anomaly_input.feature_importance,
            score=anomaly_input.anomaly_score,
            provider=args.llm_provider,
            model=args.llm_model,
        )

    report = build_markdown_report(
        anomaly_input=anomaly_input,
        stats=stats,
        severity=severity,
        causes=causes,
        actions=actions,
        llm_text=llm_text,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(report, encoding="utf-8")
        print(f"[INFO] 보고서를 저장했습니다: {output_path}")
    else:
        print(report)

    return report


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"[ERROR] 실행 중 문제가 발생했습니다: {exc}")
