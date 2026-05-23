# Run Streamlit in background and log output
# Usage: .\scripts\run_streamlit_bg.ps1 [-Port 8501] [-Address '127.0.0.1']
param(
    [int]$Port = 8501,
    [string]$Address = '127.0.0.1'
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$scriptPath = Join-Path $projectRoot "scripts\ui.py"
$logDir = Join-Path $projectRoot "logs"
$logFile = Join-Path $logDir "streamlit.log"

if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

Write-Host "Starting Streamlit → $logFile"
Write-Host "Max upload size: 10GB (via STREAMLIT_SERVER_MAX_UPLOAD_SIZE=10000)" -ForegroundColor Yellow
if (Test-Path $logFile) { Remove-Item $logFile -Force -ErrorAction SilentlyContinue }

$job = Start-Job -ScriptBlock {
    param($py, $ui, $port, $addr, $log)
    $env:STREAMLIT_SERVER_MAX_UPLOAD_SIZE = '10000'
    & $py -u -m streamlit run $ui --server.port $port --server.address $addr --server.headless true --server.enableCORS false --server.maxUploadSize=10000 *>> $log
} -ArgumentList $venvPython, $scriptPath, $Port, $Address, $logFile

Write-Host "Streamlit started (background job) JobId=$($job.Id). Tail logs with: Get-Content -Path $logFile -Wait"
