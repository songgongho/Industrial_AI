#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PressXAI 대시보드 런처
더블클릭 또는 python run_dashboard.py 로 실행하면:
1. Streamlit 대시보드가 자동으로 시작됩니다 (http://localhost:8501)
2. 기본 웹 브라우저가 자동으로 열립니다
3. 사용자는 즉시 대시보드를 사용할 수 있습니다

사용 방법:
  1. Windows에서 이 파일(.py)을 더블클릭 (가장 간단)
  2. 또는 run_dashboard.bat 파일을 더블클릭
  3. 또는 터미널에서: python run_dashboard.py
  4. 또는 직접: streamlit run app/streamlit_app.py

서버 중지:
  - Ctrl+C 를 누르거나 콘솔 창을 닫기
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
        print("\n❌ Streamlit이 설치되지 않았습니다.")
        print("\n다음 중 하나의 방법으로 설치하세요:")
        print("  방법 1 (추천):")
        print("    pip install -r requirements.txt")
        print("  방법 2:")
        print("    pip install streamlit")
        return False


def run_dashboard():
    """Streamlit 대시보드 실행"""

    if not check_dependencies():
        sys.exit(1)

    if not STREAMLIT_APP.exists():
        print(f"\n❌ 오류: Streamlit 앱을 찾을 수 없습니다.")
        print(f"   경로: {STREAMLIT_APP}")
        print(f"\n파일이 올바른 위치에 있는지 확인해주세요.")
        sys.exit(1)

    print("\n" + "="*60)
    print("🚀 PressXAI 대시보드 시작 중...")
    print("="*60)
    print(f"   앱 경로: {STREAMLIT_APP}")
    print(f"   프로젝트: {PROJECT_ROOT}")
    print(f"   포트: 8501 (http://localhost:8501)")
    print("="*60 + "\n")

    # Streamlit 포트 (기본 8501)
    STREAMLIT_PORT = 8501
    STREAMLIT_URL = f"http://localhost:{STREAMLIT_PORT}"

    # 3초 후 브라우저 자동 오픈 (서버 시작 시간 확보)
    def open_browser():
        """서버 시작 후 브라우저 자동 오픈"""
        time.sleep(3)
        try:
            print(f"🌐 브라우저 오픈 중: {STREAMLIT_URL}")
            webbrowser.open(STREAMLIT_URL)
        except Exception as e:
            print(f"⚠️  브라우저 자동 오픈 실패: {e}")
            print(f"   수동으로 접속해주세요: {STREAMLIT_URL}")

    # 브라우저 오픈을 별도 스레드에서 실행
    import threading
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()

    print(f"\n✅ Streamlit 서버가 {STREAMLIT_URL} 에서 시작되었습니다.")
    print("   상태: 서버 실행 중")
    print("   브라우저: 자동 오픈 대기 중")
    print("\n[안내]")
    print("   - 콘솔 창이 열린 상태에서 서버가 실행됩니다")
    print("   - 콘솔 창을 닫으면 서버가 중단됩니다")
    print("   - Ctrl+C 를 누르면 서버를 강제 종료할 수 있습니다")
    print("   - 브라우저가 자동으로 열리지 않으면 다음 주소로 수동 접속:")
    print(f"     👉 {STREAMLIT_URL}\n")

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
        print("\n\n⏹️  사용자가 서버를 중지했습니다.")
        print("   대시보드가 종료되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        print("   대시보드를 시작할 수 없습니다.")
        print("\n[해결 방법]")
        print("   1. 포트 8501이 다른 프로그램에서 사용 중인지 확인")
        print("   2. requirements.txt 의 모든 패키지가 설치되어 있는지 확인")
        print("   3. Python 버전이 3.8 이상인지 확인")
        sys.exit(1)


def main():
    """메인 진입점 - 대시보드 실행"""
    try:
        run_dashboard()
    except Exception as e:
        print(f"\n❌ 예기치 않은 오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

