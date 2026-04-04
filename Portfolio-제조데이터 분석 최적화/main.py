"""
제조데이터 분석과 최적화(4주차) 실습 런처
- OPC UA 서버/클라이언트, 데이터 파이프라인, 환경 점검을 한 파일에서 실행
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SCENARIOS = {
    "server-basic": ROOT / "opcua_basic" / "opc_server.py",
    "client-basic": ROOT / "opcua_basic" / "opc_client.py",
    "server-mfg": ROOT / "opcua_basic" / "opc_server_mfg.py",
    "client-mfg": ROOT / "opcua_basic" / "opc_client_mfg.py",
    "pipeline": ROOT / "data_pipeline" / "data_pipeline.py",
    "pipeline-check": ROOT / "data_pipeline" / "data_pipeline_check.py",
    "adv-server": ROOT / "information_model" / "advanced_server.py",
    "adv-client": ROOT / "information_model" / "advanced_client.py",
    "verify-env": ROOT / "verify_environment.py",
}


def list_scenarios() -> None:
    print("사용 가능한 실습 시나리오")
    for key, path in SCENARIOS.items():
        print(f"- {key:13} -> {path.relative_to(ROOT)}")


def run_scenario(name: str) -> int:
    script = SCENARIOS.get(name)
    if script is None:
        print(f"오류: '{name}' 시나리오를 찾을 수 없습니다.")
        list_scenarios()
        return 2

    if not script.exists():
        print(f"오류: 실행 파일이 없습니다 -> {script}")
        return 3

    print(f"실행 시작: {name} ({script.relative_to(ROOT)})")
    completed = subprocess.run([sys.executable, str(script)], cwd=str(ROOT))
    return completed.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="4주차 제조데이터 수집/처리 실습 통합 실행기"
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        help="실행할 시나리오 이름 (예: server-mfg, pipeline-check)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="실행 가능한 시나리오 목록 출력",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.list or not args.scenario:
        list_scenarios()
        if not args.scenario:
            return 0

    return run_scenario(args.scenario)


if __name__ == "__main__":
    raise SystemExit(main())

