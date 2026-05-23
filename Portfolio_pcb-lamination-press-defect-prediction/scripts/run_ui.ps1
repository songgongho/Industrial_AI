# Run/Restart Streamlit UI (PowerShell)
# Usage: Open PowerShell in project root and run:
#   .\scripts\run_ui.ps1
# This script will:
#  - find any process listening on port 8501 and attempt to terminate it
#  - create logs/ and launch Streamlit using the project's venv python
#  - write stdout/stderr to logs/streamlit.out and logs/streamlit.err

$port = 8501
$defaultRoot = Join-Path $PSScriptRoot ".."
$projectRoot = if (Test-Path "C:\neotech-thesis") { "C:\neotech-thesis" } else { $defaultRoot }
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$uiScript = Join-Path $projectRoot "scripts\ui.py"
$logsDir = Join-Path $projectRoot "logs"
# project root used as working directory for Streamlit to avoid user-home temp path issues

if (-not (Test-Path $venvPython)) {
    Write-Host "ERROR: venv python not found at $venvPython" -ForegroundColor Red
    Write-Host "Activate venv or adjust script to point to your python." -ForegroundColor Yellow
    exit 1
}

# Ensure logs directory exists
if (-not (Test-Path $logsDir)) { New-Item -ItemType Directory -Path $logsDir | Out-Null }
$outLog = (Join-Path $logsDir "streamlit.out")
$errLog = (Join-Path $logsDir "streamlit.err")

# Find processes listening on port
$net = netstat -ano | findstr ":$port"
if ($net) {
    Write-Host "Found listening entries for port $port. Attempting to extract PIDs..."
    $lines = $net -split "`n"
    $pids = @()
    foreach ($l in $lines) {
        $parts = ($l -split '\s+') -ne ''
        if ($parts.Length -ge 5) { $thePid = $parts[-1]; if ($thePid -match '^[1-9][0-9]*$') { $pids += $thePid } }
    }
    $pids = $pids | Select-Object -Unique
    foreach ($thePid in $pids) {
        try {
            Write-Host "Killing PID $thePid..."
            taskkill /PID $thePid /F | Out-Null
        } catch {
            Write-Host "Failed to kill ${thePid}: $_" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "No existing listener on port $port found."
}

Start-Sleep -Milliseconds 500

# Set environment variables for large file uploads (10GB limit)
$env:STREAMLIT_CLIENT_MAX_UPLOAD_SIZE = '10000'
$env:STREAMLIT_SERVER_MAX_UPLOAD_SIZE = '10000'

# Start Streamlit via cmd.exe so the command line is evaluated in a plain shell context.
Write-Host "Starting Streamlit UI with 10GB upload limit..." -ForegroundColor Cyan
Write-Host "Command: $venvPython -m streamlit run $uiScript --server.port $port --server.address 127.0.0.1 --server.headless true --server.enableCORS false --server.maxUploadSize=10000" -ForegroundColor Yellow
$streamlitCmd = "set STREAMLIT_SERVER_MAX_UPLOAD_SIZE=10000 & `"$venvPython`" -m streamlit run `"$uiScript`" --server.port $port --server.address 127.0.0.1 --server.headless true --server.enableCORS false --server.maxUploadSize=10000"
Start-Process -FilePath cmd.exe -ArgumentList '/c', $streamlitCmd -RedirectStandardOutput $outLog -RedirectStandardError $errLog -WorkingDirectory $projectRoot -WindowStyle Hidden

Write-Host "Streamlit started (logs: $outLog, $errLog). Give it a few seconds and open http://127.0.0.1:$port/ in your browser." -ForegroundColor Green

# Tail the last few lines of output (non-blocking)
Get-Content $outLog -Tail 20

