"""
실습 환경 점검 스크립트
- 필수 라이브러리 import
- 버전 출력
- OPENCV contrib 기능(SIFT) 사용 가능 여부 확인
"""

from __future__ import annotations

import importlib
import sys

REQUIRED_MODULES = [
    "asyncua",
    "numpy",
    "pandas",
    "sklearn",
    "matplotlib",
    "cv2",
]


def check_modules() -> int:
    print("[환경 점검] 필수 모듈 import 확인")
    missing = []

    for module_name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "버전 정보 없음")
            print(f"- OK   {module_name:10} | {version}")
        except Exception as exc:  # noqa: BLE001
            print(f"- FAIL {module_name:10} | {exc}")
            missing.append(module_name)

    if missing:
        print("\n누락 모듈이 있습니다:", ", ".join(missing))
        print("`pip install -r requirements.txt` 실행 후 다시 점검하세요.")
        return 1

    print("\n모든 필수 모듈 import 성공")
    return 0


def check_opencv_contrib() -> int:
    print("\n[환경 점검] OpenCV contrib 기능 확인")
    try:
        import cv2

        has_sift = hasattr(cv2, "SIFT_create")
        has_xfeatures = hasattr(cv2, "xfeatures2d")

        print(f"- cv2.__version__: {cv2.__version__}")
        print(f"- SIFT_create 지원: {'예' if has_sift else '아니오'}")
        print(f"- xfeatures2d 지원: {'예' if has_xfeatures else '아니오'}")

        if not (has_sift or has_xfeatures):
            print("contrib 기능이 비활성일 수 있습니다. opencv-contrib-python 설치를 확인하세요.")
            return 2

        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"OpenCV 점검 실패: {exc}")
        return 3


def main() -> int:
    code = check_modules()
    if code != 0:
        return code

    return check_opencv_contrib()


if __name__ == "__main__":
    raise SystemExit(main())

