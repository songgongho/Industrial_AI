param(
    [string]$PythonExe = "python",
    [string]$VenvName = ".venv"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot $VenvName
$RequirementsPath = Join-Path $ProjectRoot "requirements.txt"

Write-Host "[1/4] 프로젝트 루트: $ProjectRoot"
Write-Host "[2/4] 가상환경 생성: $VenvPath"
& $PythonExe -m venv $VenvPath

$ActivateScript = Join-Path $VenvPath "Scripts\Activate.ps1"
if (-not (Test-Path $ActivateScript)) {
    throw "가상환경 활성화 스크립트를 찾지 못했습니다: $ActivateScript"
}

Write-Host "[3/4] 가상환경 활성화"
. $ActivateScript

Write-Host "[4/4] 의존성 설치"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r $RequirementsPath

Write-Host "설치 완료"
Write-Host "다음 명령으로 점검하세요: python verify_environment.py"

