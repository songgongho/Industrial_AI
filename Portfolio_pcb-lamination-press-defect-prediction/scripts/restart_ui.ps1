# Restart Streamlit UI (Windows PowerShell)
# Usage: .\scripts\restart_ui.ps1

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$scriptPath = Join-Path $projectRoot "scripts\ui.py"

Write-Host "Stopping existing streamlit python processes bound to ui.py (if any)..."
# Find python processes running streamlit ui.py
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine -match 'streamlit' -and $_.CommandLine -match 'ui.py' } | ForEach-Object {
    Write-Host "Killing PID:" $_.ProcessId
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}

Start-Sleep -Seconds 1

Write-Host "Starting Streamlit in background (127.0.0.1:8501)..."
Start-Process -FilePath $venvPython -ArgumentList '-m','streamlit','run',$scriptPath,'--server.port','8501','--server.address','127.0.0.1','--server.headless','true','--server.enableCORS','false' -WindowStyle Hidden
Write-Host "Started. Open http://localhost:8501 in your browser."
