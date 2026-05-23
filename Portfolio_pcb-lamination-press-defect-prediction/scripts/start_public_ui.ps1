# Start Streamlit locally and expose it via localtunnel (Windows PowerShell)
# Usage: Open PowerShell in project root and run:
#   .\scripts\start_public_ui.ps1 [-Port 8501]
# Requirements: node & npx available on PATH (localtunnel will be installed temporarily via npx)

param(
    [int]$Port = 8501,
    [switch]$UseNgrok
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$logsDir = Join-Path $projectRoot "logs"
New-Item -ItemType Directory -Path $logsDir -Force | Out-Null
$ltLog = Join-Path $logsDir "localtunnel.log"
$publicUrlFile = Join-Path $logsDir "public_url.txt"
if (Test-Path $publicUrlFile) { Remove-Item $publicUrlFile -Force -ErrorAction SilentlyContinue }

Write-Host "1) Starting Streamlit (background) on port $Port..."
# Use existing helper which starts Streamlit in background
& (Join-Path $projectRoot "scripts\run_streamlit_bg.ps1") -Port $Port -Address '127.0.0.1'

Start-Sleep -Seconds 2

$found = $null

# If user requested ngrok (or authtoken env var present), try ngrok first
$ngrokToken = $env:NGROK_AUTHTOKEN
if ($UseNgrok -or $ngrokToken) {
    Write-Host "2) Attempting to launch ngrok tunnel (preferred)..."
    $venvNgrok = Join-Path $projectRoot ".venv\Scripts\ngrok.exe"
    $ngrokExeCandidates = @($venvNgrok, "ngrok")
    $ngrokExe = $null
    foreach ($c in $ngrokExeCandidates) {
        try { $cmd = Get-Command $c -ErrorAction SilentlyContinue; if ($cmd) { $ngrokExe = $cmd.Source; break } } catch { }
    }

    if (-not $ngrokExe) {
        Write-Host "ngrok not found. Falling back to localtunnel." -ForegroundColor Yellow
    } else {
        # If token env var exists, assume ngrok already configured. Otherwise warn.
        if (-not $ngrokToken) { Write-Host "No NGROK_AUTHTOKEN env var found. Ensure you've run scripts/setup_ngrok.ps1 or set NGROK_AUTHTOKEN." -ForegroundColor Yellow }

        $ngrokLog = Join-Path $logsDir 'ngrok.log'
        if (Test-Path $ngrokLog) { Remove-Item $ngrokLog -Force -ErrorAction SilentlyContinue }
        Start-Process -FilePath $ngrokExe -ArgumentList 'http', $Port, '--log=stdout','--log-format=logfmt' -RedirectStandardOutput $ngrokLog -RedirectStandardError $ngrokLog -WindowStyle Hidden -PassThru | Out-Null

        # Wait and parse ngrok log for a public URL
        for ($i=0; $i -lt 30; $i++) {
            Start-Sleep -Seconds 1
            if (Test-Path $ngrokLog) {
                $txt = Get-Content $ngrokLog -ErrorAction SilentlyContinue | Out-String
                if ($txt -match 'https?://[\w\-]+\.ngrok\.io') { $found = $matches[0]; break }
                if ($txt -match 'url=(https?://\S+ngrok\.io)') { $found = $matches[1]; break }
            }
        }

        if ($found) {
            Write-Host "Public URL (ngrok):" -ForegroundColor Green; Write-Host $found -ForegroundColor Cyan
            Write-Host "Local Streamlit: http://127.0.0.1:$Port/"
            Write-Host "ngrok logs: $ngrokLog"
            Set-Content -Path $publicUrlFile -Value $found -Encoding UTF8
            Write-Host "Done."
            return
        }
        Write-Host "ngrok did not produce a public URL in time; falling back to localtunnel." -ForegroundColor Yellow
    }
}

Write-Host "2) Launching localtunnel via npx (this may take a few seconds)..."
# Start localtunnel in a background job and direct output to logs/localtunnel.log
if (Test-Path $ltLog) { Remove-Item $ltLog -Force -ErrorAction SilentlyContinue }
$job = Start-Job -ScriptBlock { param($p,$log) npx --yes localtunnel --port $p *> $log } -ArgumentList $Port,$ltLog

# Wait for localtunnel to emit a URL
$timeout = 40
for ($i=0; $i -lt $timeout; $i++) {
    Start-Sleep -Seconds 1
    if (Test-Path $ltLog) {
        $tail = Get-Content $ltLog -Tail 50 -ErrorAction SilentlyContinue
        if ($tail -match 'https?://[\w\-\.]+\.(loca\.lt|trycloudflare\.com|localtunnel\.me|lhrtunnel\.net)[^\s"]*') {
            $found = $matches[0]
            break
        }
        if ($tail -match 'your url is:\s*(https?://\S+)') {
            $found = $matches[1]
            break
        }
    }
}

if ($found) {
    Write-Host "Public URL:" -ForegroundColor Green; Write-Host $found -ForegroundColor Cyan
    Write-Host "Local Streamlit: http://127.0.0.1:$Port/"
    Write-Host "Localtunnel logs: $ltLog"
    Set-Content -Path $publicUrlFile -Value $found -Encoding UTF8
} else {
    Write-Host "Failed to detect public URL within timeout. Check logs: $ltLog" -ForegroundColor Yellow
    if ($job -and $job.State -eq 'Running') { Write-Host "Localtunnel job is still running (JobId=$($job.Id)). Use Get-Job / Receive-Job to inspect." }
}

Write-Host "Done."
