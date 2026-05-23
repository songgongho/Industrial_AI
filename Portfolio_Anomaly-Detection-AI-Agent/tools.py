from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class AnomalyInput:
    timestamp: str
    anomaly_score: float
    feature_importance: Dict[str, float]
    historical_data: List[float]
    sensor_id: str


def parse_input_file(input_path: Path) -> AnomalyInput:
    if not input_path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")

    if input_path.suffix.lower() == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    elif input_path.suffix.lower() == ".csv":
        payload = _read_csv_payload(input_path)
    else:
        raise ValueError("지원하지 않는 파일 형식입니다. JSON 또는 CSV를 사용하세요.")

    return validate_input(payload)


def _read_csv_payload(input_path: Path) -> Dict[str, Any]:
    df = pd.read_csv(str(input_path))
    if df.empty:
        raise ValueError("CSV 파일이 비어 있습니다.")

    row = df.iloc[0]
    feature_importance = _parse_json_cell(row.get("feature_importance", "{}"), default={})
    historical_data = _parse_json_cell(row.get("historical_data", "[]"), default=[])

    return {
        "timestamp": str(row.get("timestamp", "")),
        "anomaly_score": float(row.get("anomaly_score", 0.0)),
        "feature_importance": feature_importance,
        "historical_data": historical_data,
        "sensor_id": str(row.get("sensor_id", "UNKNOWN_SENSOR")),
    }


def _parse_json_cell(value: Any, default: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return default

    text = str(value).strip()
    if not text:
        return default

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def validate_input(payload: Dict[str, Any]) -> AnomalyInput:
    required_keys = {
        "timestamp",
        "anomaly_score",
        "feature_importance",
        "historical_data",
        "sensor_id",
    }
    missing = required_keys - set(payload.keys())
    if missing:
        raise ValueError(f"필수 입력 키가 누락되었습니다: {sorted(missing)}")

    feature_importance = payload["feature_importance"]
    historical_data = payload["historical_data"]

    if not isinstance(feature_importance, dict) or not feature_importance:
        raise ValueError("feature_importance는 비어 있지 않은 dict여야 합니다.")

    if not isinstance(historical_data, list) or not historical_data:
        raise ValueError("historical_data는 비어 있지 않은 list여야 합니다.")

    score = float(payload["anomaly_score"])
    if score < 0 or score > 1:
        raise ValueError("anomaly_score는 0~1 범위여야 합니다.")

    normalized_feature_importance = {
        str(k): float(v) for k, v in feature_importance.items()
    }
    normalized_historical = [float(v) for v in historical_data]

    return AnomalyInput(
        timestamp=str(payload["timestamp"]),
        anomaly_score=score,
        feature_importance=normalized_feature_importance,
        historical_data=normalized_historical,
        sensor_id=str(payload["sensor_id"]),
    )


def analyze_history(historical_data: List[float]) -> Dict[str, float]:
    series = pd.Series(historical_data, dtype="float64")
    mean_value = float(series.mean())
    std_value = float(series.std(ddof=0)) if len(series) > 1 else 0.0
    latest = float(series.iloc[-1])
    prev = float(series.iloc[-2]) if len(series) > 1 else latest

    change_rate = 0.0
    if abs(prev) > 1e-9:
        change_rate = (latest - prev) / abs(prev)

    mean_ratio = latest / mean_value if abs(mean_value) > 1e-9 else 0.0

    return {
        "mean": mean_value,
        "std": std_value,
        "latest": latest,
        "change_rate": change_rate,
        "mean_ratio": mean_ratio,
    }


def assess_severity(score: float, change_rate: float) -> str:
    if score >= 0.8 or (score >= 0.7 and change_rate >= 0.3):
        return "High"
    if score >= 0.55 or change_rate >= 0.15:
        return "Medium"
    return "Low"


def infer_causes(feature_importance: Dict[str, float], stats: Dict[str, float]) -> List[str]:
    top_features = sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)[:3]
    causes: List[str] = []

    for name, weight in top_features:
        causes.append(f"{name} 기여도 {weight:.0%}")

    if stats["mean_ratio"] >= 1.5:
        causes.append(f"최근 점수가 과거 평균 대비 {stats['mean_ratio']:.2f}배 상승")
    if stats["change_rate"] >= 0.2:
        causes.append(f"직전 대비 변화율 {stats['change_rate']:.1%}로 급격한 증가")

    return causes


def suggest_actions(severity: str) -> List[str]:
    action_map = {
        "High": [
            "즉시 현장 점검을 수행하고 필요 시 설비를 안전 모드로 전환",
            "백업 라인 가동 여부를 검토하고 불량 확산을 차단",
            "유지보수 담당자에게 10분 이내 알림 전파",
        ],
        "Medium": [
            "다음 30분 동안 점수 추이를 집중 모니터링",
            "상위 기여 센서의 보정 상태를 확인",
            "동일 공정의 최근 알람 이력을 대조",
        ],
        "Low": [
            "정기 모니터링 주기 내에서 추세를 관찰",
            "센서 드리프트 여부를 주간 단위로 점검",
        ],
    }
    return action_map.get(severity, ["운영 표준 절차에 따라 모니터링"])


def optional_llm_interpretation(
    feature_importance: Dict[str, float],
    score: float,
    provider: str,
    model: str,
    timeout: int = 20,
) -> str:
    prompt = (
        "다음 제조 이상탐지 결과를 한국어로 3문장 해석하세요. "
        "문장에는 원인, 위험도, 권고 조치를 포함하세요.\n"
        f"anomaly_score={score}\n"
        f"feature_importance={feature_importance}"
    )

    if provider == "ollama":
        body = {
            "model": model,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
        }
        request = urllib.request.Request(
            "http://localhost:11434/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return _read_llm_response(request, timeout, provider)

    if provider == "openai":
        api_key = Path(".openai_api_key").read_text(encoding="utf-8").strip() if Path(".openai_api_key").exists() else ""
        if not api_key:
            return "LLM 해석 생략: .openai_api_key 파일이 없어 OpenAI 호출을 건너뜁니다."

        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        return _read_llm_response(request, timeout, provider)

    return f"LLM 해석 생략: 지원하지 않는 provider입니다 ({provider})."


def _read_llm_response(request: urllib.request.Request, timeout: int, provider: str) -> str:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            payload = json.loads(raw)

            if provider == "ollama":
                return payload.get("message", {}).get("content", "LLM 응답 파싱 실패")
            if provider == "openai":
                choices = payload.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "LLM 응답 파싱 실패")
                return "LLM 응답이 비어 있습니다."

            return "LLM 응답 파싱 실패"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return f"LLM 해석 실패: {exc}"


def build_markdown_report(
    anomaly_input: AnomalyInput,
    stats: Dict[str, float],
    severity: str,
    causes: List[str],
    actions: List[str],
    llm_text: Optional[str] = None,
) -> str:
    main_cause = ", ".join(causes[:2]) if causes else "원인 정보 부족"
    main_action = actions[0] if actions else "대응 권고 없음"

    lines = [
        f"## 이상탐지 보고서 [{anomaly_input.sensor_id}]",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 발생 시각 | {anomaly_input.timestamp} |",
        f"| 이상점수 | {anomaly_input.anomaly_score:.2f} ({severity}) |",
        f"| 주요원인 | {main_cause} |",
        f"| 대응 | {main_action} |",
        "",
        "### 상세 분석",
        f"- 과거 평균 점수: {stats['mean']:.3f}",
        f"- 최근 점수: {stats['latest']:.3f}",
        f"- 직전 대비 변화율: {stats['change_rate']:.1%}",
        "",
        "### 권고 조치",
    ]

    for action in actions:
        lines.append(f"- {action}")

    summary = (
        f"요약: {anomaly_input.sensor_id}에서 {severity} 수준 이상 징후가 감지되었습니다. "
        f"핵심 원인은 {main_cause}이며, {main_action} 조치가 우선 권고됩니다."
    )
    lines.extend(["", summary])

    if llm_text:
        lines.extend(["", "### LLM 해석", llm_text])

    return "\n".join(lines)

