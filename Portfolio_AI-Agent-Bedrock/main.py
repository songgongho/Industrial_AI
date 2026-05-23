"""로컬 실습용 간단한 실행 안내 스크립트."""

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    print("Bedrock Agent 실습 프로젝트 준비 상태")
    print(f"- 프로젝트 경로: {root}")
    print(f"- 앱 파일 존재: {(root / 'app.py').exists()}")
    print(f"- 사전 점검 스크립트 존재: {(root / 'scripts' / 'preflight_check.py').exists()}")
    print("\n다음 순서로 실행하세요:")
    print("1) python -m scripts.preflight_check")
    print("2) streamlit run app.py")


if __name__ == "__main__":
    main()
