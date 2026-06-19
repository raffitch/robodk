<#
  Start the tasni control panel (Windows / PowerShell).

    .\start.ps1          dev  - FastAPI (:8000) + Vite (:5173, hot reload). Opens :5173
    .\start.ps1 prod     build the React app, then serve everything from FastAPI (:8000)

  On start it kills any previous tasni backend/Vite still running, then launches
  fresh and opens the browser automatically. (RoboDK is left alone — reusing a
  running RoboDK avoids the slow 117 MB station reload.) Requires `py -3.10` and
  node/npm on PATH. Closing Vite (Ctrl-C) stops the backend.
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

function Stop-PriorInstances {
    # Vite / esbuild
    Get-Process node, esbuild -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    # our backend (python running `-m tasni`)
    Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='py.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like '*tasni*' } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    # anything still holding our ports
    $owners = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
        Where-Object { $_.LocalPort -in 8000, 5173 } |
        Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($procId in $owners) { Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue }
}

function Open-Browser($url) {
    # Wait (in the background) for the port to accept connections, then open the
    # default browser — so we don't open before the server is listening. Tries
    # both IPv4/IPv6 localhost, waits up to ~60s, and falls back to explorer.
    $port = ([uri]$url).Port
    Start-Job -ArgumentList $url, $port -ScriptBlock {
        param($url, $port)
        function Test-Up($p) {
            foreach ($h in @('127.0.0.1', 'localhost')) {
                try { $c = New-Object Net.Sockets.TcpClient; $c.Connect($h, $p); $u = $c.Connected; $c.Close()
                      if ($u) { return $true } } catch {}
            }
            return $false
        }
        for ($i = 0; $i -lt 120; $i++) {
            if (Test-Up $port) {
                Start-Sleep -Milliseconds 600   # let the server finish standing up
                try { Start-Process $url } catch { try { Start-Process explorer.exe $url } catch {} }
                return
            }
            Start-Sleep -Milliseconds 500
        }
    } | Out-Null
}

Stop-PriorInstances
Initialize-WebDeps

if ($Mode -eq 'prod') {
    Write-Host '[start] building web UI...' -ForegroundColor Cyan
    Push-Location $webui
    try { npm run build } finally { Pop-Location }
    Write-Host '[start] serving on http://localhost:8000' -ForegroundColor Green
    Open-Browser 'http://localhost:8000'
    py -3.10 -m tasni --port 8000
    return
}

# dev: backend in its own window + Vite in this one; kill the backend on exit.
Write-Host '[start] backend  -> http://localhost:8000' -ForegroundColor Green
$backend = Start-Process -FilePath 'py' `
    -ArgumentList '-3.10', '-m', 'tasni', '--port', '8000' -PassThru
Open-Browser 'http://localhost:5173'
try {
    Write-Host '[start] UI (dev) -> http://localhost:5173  (opening in your browser)' -ForegroundColor Green
    Push-Location $webui
    npm run dev
} finally {
    Pop-Location
    if ($backend -and -not $backend.HasExited) {
        Write-Host '[start] stopping backend...' -ForegroundColor Cyan
        taskkill /PID $backend.Id /T /F | Out-Null
    }
    Get-Job -ErrorAction SilentlyContinue | Remove-Job -Force -ErrorAction SilentlyContinue
}
