<#
Register ngrok authtoken for local usage
Usage:
  .\scripts\setup_ngrok.ps1 -Token 'YOUR_AUTHTOKEN'
This will run the ngrok executable found in .venv\Scripts\ngrok.exe if present,
otherwise it will attempt to run `ngrok` on PATH.
#>
param(
    [Parameter(Mandatory=$true)]
    [string]$Token
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvNgrok = Join-Path $projectRoot ".venv\Scripts\ngrok.exe"
$ngrokPaths = @($venvNgrok, "ngrok")

$ngrokExe = $null
foreach ($p in $ngrokPaths) {
    try {
        $cmd = Get-Command $p -ErrorAction SilentlyContinue
        if ($cmd) { $ngrokExe = $cmd.Source; break }
    } catch { }
}

if (-not $ngrokExe) {
    Write-Host "ngrok executable not found (.venv/Scripts/ngrok.exe or ngrok on PATH)." -ForegroundColor Yellow
    Write-Host "Please install ngrok or place ngrok.exe into .venv\Scripts and retry." -ForegroundColor Yellow
    exit 1
}

Write-Host "Registering authtoken with: $ngrokExe"
$proc = Start-Process -FilePath $ngrokExe -ArgumentList 'config','add-authtoken',$Token -Wait -NoNewWindow -PassThru
if ($proc.ExitCode -eq 0) {
    Write-Host "ngrok authtoken registered successfully." -ForegroundColor Green
    Write-Host "You can now run: .\scripts\start_public_ui.ps1 -UseNgrok`n(or set environment variable NGROK_AUTHTOKEN and run start_public_ui.ps1)"
} else {
    Write-Host "ngrok returned exit code $($proc.ExitCode). Check the executable and token." -ForegroundColor Red
    exit $proc.ExitCode
}

