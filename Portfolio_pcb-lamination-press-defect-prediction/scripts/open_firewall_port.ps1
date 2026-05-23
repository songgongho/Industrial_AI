# Open Windows Firewall port for Streamlit (requires admin)
# Usage: Run PowerShell as Administrator and execute:
#   .\scripts\open_firewall_port.ps1 -Port 8501 -RuleName "Streamlit 8501"
param(
    [int]$Port = 8501,
    [string]$RuleName = 'Streamlit 8501'
)

Write-Host "Adding inbound firewall rule for TCP port $Port (rule: $RuleName)"
try {
    New-NetFirewallRule -DisplayName $RuleName -Direction Inbound -Action Allow -Protocol TCP -LocalPort $Port -Profile Any -EdgeTraversalPolicy Allow -ErrorAction Stop
    Write-Host "Firewall rule added."
} catch {
    Write-Error "Failed to add firewall rule: $_"
}

