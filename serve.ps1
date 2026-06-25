<#
  Headless Tasni server for REMOTE use (SSH over Tailscale from the Mac).

  Unlike start.ps1 this opens NO browser window and does NOT block on one; it
  runs the backend in the FOREGROUND so whatever launched it owns its lifetime.
  Launched over SSH (`ssh -tt ... powershell -File serve.ps1`), closing the SSH
  session tears the server down. It also kills any prior Tasni/:8000 owner on
  start, so a stale remote server can never linger and block the port.

  Serves the built React UI on http://127.0.0.1:<port> (default 8000); pair it
  with an SSH local port-forward (-L 8000:localhost:8000) and open it on the Mac.

    powershell -NoProfile -ExecutionPolicy Bypass -File serve.ps1            # build + serve
    powershell -NoProfile -ExecutionPolicy Bypass -File serve.ps1 -NoBuild   # skip the rebuild
    powershell -NoProfile -ExecutionPolicy Bypass -File serve.ps1 -Stop      # just stop a running server
#>
[CmdletBinding()]
param([int]$Port = 8000, [switch]$NoBuild, [switch]$Stop)

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot
$webui = Join-Path $PSScriptRoot 'tasni\webui'

function Stop-Tasni {
    # kill any python running the tasni package...
    Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='py.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like '*tasni*' } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    # ...and whatever still owns the port
    Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique |
        ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
}

Stop-Tasni
if ($Stop) { Write-Host "[tasni] stopped." -ForegroundColor Cyan; return }

if (-not $NoBuild) {
    if (-not (Test-Path (Join-Path $webui 'node_modules'))) {
        Write-Host '[tasni] installing web UI deps...' -ForegroundColor Cyan
        Push-Location $webui; try { npm install } finally { Pop-Location }
    }
    Write-Host '[tasni] building web UI (so :' "$Port" ' serves the latest)...' -ForegroundColor Cyan
    Push-Location $webui; try { npm run build } finally { Pop-Location }
}

Write-Host "[tasni] serving http://127.0.0.1:$Port  --  end the SSH session (or -Stop) to halt." -ForegroundColor Green
# Foreground uvicorn: when the launching SSH session closes, Windows OpenSSH ends
# this process tree and the server stops.
#
# IMPORTANT: drop ErrorActionPreference back to Continue first. uvicorn logs to
# stderr, and under 'Stop' (esp. when stdout/err is redirected, as over SSH)
# PowerShell wraps the first stderr line as a terminating NativeCommandError and
# would kill the server the instant it started.
$ErrorActionPreference = 'Continue'
py -3.10 -m tasni --port $Port
