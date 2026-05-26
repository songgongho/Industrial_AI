#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PressXAI Dashboard Launcher
더블클릭으로 실행: Streamlit 대시보드를 로컬에서 시작하고 브라우저를 자동으로 엽니다.

사용:
  1. Windows에서 이 파일(.py)을 더블클릭
  2. 또는 터미널에서: python run_dashboard.py
  3. 또는 패키징: pyinstaller로 .exe로 변환 가능
"""

import os
import sys
import time
import webbrowser
import subprocess
from pathlib import Path

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).resolve().parent
STREAMLIT_APP = PROJECT_ROOT / "app" / "streamlit_app.py"


def check_dependencies():
    """필요 패키지 확인"""
    try:
        import streamlit
        return True
    except ImportError:
        print("❌ Streamlit이 설치되지 않았습니다.")
        print("다음 명령으로 설치하세요:")
        print("  pip install streamlit")
        print("\n또는 프로젝트 루트에서:")
        print("  pip install -r requirements.txt")
        return False


def run_dashboard():
    """Streamlit 대시보드 실행"""

    if not check_dependencies():
        sys.exit(1)

    # Streamlit 앱 경로 확인
    if not STREAMLIT_APP.exists():
        print(f"❌ Streamlit 앱을 찾을 수 없습니다: {STREAMLIT_APP}")
        sys.exit(1)

    print("🚀 PressXAI Dashboard를 시작하는 중...")
    print(f"   앱 경로: {STREAMLIT_APP}")
    print(f"   작업 디렉토리: {PROJECT_ROOT}")

    # Streamlit 포트 (기본 8501)
    STREAMLIT_PORT = 8501
    STREAMLIT_URL = f"http://localhost:{STREAMLIT_PORT}"

    # 3초 후 브라우저 자동 오픈 (서버 시작 시간 확보)
    def open_browser():
        """지연 후 브라우저 오픈"""
        time.sleep(3)
        print(f"\n🌐 브라우저에서 열기: {STREAMLIT_URL}")
        webbrowser.open(STREAMLIT_URL)

    # 브라우저 오픈을 별도 스레드에서 실행
    import threading
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()

    print(f"\n✅ Streamlit 서버가 {STREAMLIT_URL} 에서 시작됩니다.")
    print("   터미널을 닫으면 서버가 중단됩니다.")
    print("   Ctrl+C를 누르면 서버를 중지할 수 있습니다.\n")

    # Streamlit 실행 (상대 경로 사용)
    os.chdir(PROJECT_ROOT)
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(STREAMLIT_APP),
        "--logger.level=warning",  # 로그 레벨 조정
    ]

    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        print("\n\n⏹️  서버가 중지되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        sys.exit(1)


def main():
    """메인 진입점"""
    try:
        run_dashboard()
    except Exception as e:
        print(f"\n❌ 예기치 않은 오류: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

