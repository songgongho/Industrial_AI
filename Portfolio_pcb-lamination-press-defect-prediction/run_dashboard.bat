@echo off
REM -*- coding: utf-8 -*-
REM ============================================================
REM PressXAI Dashboard 실행 파일
REM 이 파일을 더블클릭하면 대시보드가 자동으로 시작됩니다.
REM ============================================================

chcp 65001 >nul
setlocal enabledelayedexpansion

echo.
echo ================================================================
echo  PressXAI 대시보드 실행 중...
echo ================================================================
echo.

REM 현재 디렉토리를 프로젝트 루트로 설정
cd /d "%~dp0"

REM Python 가상환경 활성화
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo [OK] 가상환경 활성화됨
) else (
    echo [경고] 가상환경을 찾을 수 없습니다.
    echo "python -m venv .venv" 로 가상환경을 먼저 생성해주세요.
    pause
    exit /b 1
)

REM Streamlit 설치 확인
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [경고] Streamlit이 설치되지 않았습니다.
    echo 설치를 진행합니다...
    pip install streamlit -q
)

REM 환경변수 설정
set STREAMLIT_SERVER_HEADLESS=true
set STREAMLIT_LOGGER_LEVEL=warning
set STREAMLIT_CLIENT_SHOWERRORDETAILS=false

REM 대시보드 실행
echo.
echo [시작] Streamlit 서버가 시작되었습니다.
echo [안내] 브라우저에서 http://localhost:8501 로 접속해주세요.
echo [중지] 이 창을 닫으면 서버가 중지됩니다.
echo.

python run_dashboard.py

pause

