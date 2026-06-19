<#
  Start the tasni control panel (Windows / PowerShell).

    .\start.ps1          dev  - FastAPI (:8000) + Vite (:5173, hot reload).  Open :5173
    .\start.ps1 prod     build the React app, then serve everything from FastAPI (:8000)

  Requires the Python launcher (`py -3.10`) and node/npm on PATH. The backend
  opens in its own window in dev mode; closing Vite (Ctrl-C) stops the backend.
#>
[CmdletBinding()]
param([ValidateSet('dev', 'prod')][string]$Mode = 'dev')

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot
$webui = Join-Path $PSScriptRoot 'tasni\webui'

function Initialize-WebDeps {
    if (-not (Test-Path (Join-Path $webui 'node_modules'))) {
        Write-Host '[start] installing web UI deps...' -ForegroundColor Cyan
        Push-Location $webui
        try { npm install } finally { Pop-Location }
    }
}

if ($Mode -eq 'prod') {
    Initialize-WebDeps
    Write-Host '[start] building web UI...' -ForegroundColor Cyan
    Push-Location $webui
    try { npm run build } finally { Pop-Location }
    Write-Host '[start] serving on http://localhost:8000' -ForegroundColor Green
    py -3.10 -m tasni --port 8000
    return
}

# dev: backend in its own window + Vite in this one; kill the backend on exit.
Initialize-WebDeps
Write-Host '[start] backend  -> http://localhost:8000' -ForegroundColor Green
$backend = Start-Process -FilePath 'py' `
    -ArgumentList '-3.10', '-m', 'tasni', '--port', '8000' -PassThru
try {
    Write-Host '[start] UI (dev) -> http://localhost:5173  (open this one)' -ForegroundColor Green
    Push-Location $webui
    npm run dev
} finally {
    Pop-Location
    if ($backend -and -not $backend.HasExited) {
        Write-Host '[start] stopping backend...' -ForegroundColor Cyan
        taskkill /PID $backend.Id /T /F | Out-Null
    }
}
