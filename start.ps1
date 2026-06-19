<#
  Start the Tasni control panel as a standalone app window (Windows / PowerShell).

    .\start.ps1          dev  - FastAPI (:8000) + Vite (:5173, hot reload)
    .\start.ps1 prod     build the React app, then serve everything from FastAPI (:8000)

  Servers run hidden in the background; the UI opens in a Chromium app-mode window
  (no tabs/address bar). CLOSING THE APP WINDOW STOPS EVERYTHING. On start it also
  kills any previous Tasni run. Requires `py -3.10`, node/npm, and Chrome or Edge.
#>
[CmdletBinding()]
param([ValidateSet('dev', 'prod')][string]$Mode = 'dev')

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot
$webui = Join-Path $PSScriptRoot 'tasni\webui'
$appProfile = Join-Path $env:TEMP 'tasni-appwin'

function Initialize-WebDeps {
    if (-not (Test-Path (Join-Path $webui 'node_modules'))) {
        Write-Host '[tasni] installing web UI deps...' -ForegroundColor Cyan
        Push-Location $webui; try { npm install } finally { Pop-Location }
    }
}

function Stop-PriorInstances {
    Get-Process node, esbuild -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='py.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like '*tasni*' } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    # any previous Tasni app window (matched by its dedicated profile dir)
    Get-CimInstance Win32_Process -Filter "Name='chrome.exe' OR Name='msedge.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like '*tasni-appwin*' } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    $owners = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
        Where-Object { $_.LocalPort -in 8000, 5173 } |
        Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($procId in $owners) { Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue }
}

function Find-Browser {
    foreach ($c in @(
            "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
            "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
            "$env:LocalAppData\Google\Chrome\Application\chrome.exe",
            "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe",
            "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe")) {
        if ($c -and (Test-Path $c)) { return $c }
    }
    return $null
}

function Wait-Port($port, $timeoutSec = 90) {
    for ($i = 0; $i -lt ($timeoutSec * 2); $i++) {
        foreach ($h in @('127.0.0.1', 'localhost')) {
            try { $c = New-Object Net.Sockets.TcpClient; $c.Connect($h, $port); $u = $c.Connected; $c.Close()
                  if ($u) { return $true } } catch {}
        }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

Stop-PriorInstances
Initialize-WebDeps
if ($Mode -eq 'prod') {
    Write-Host '[tasni] building web UI...' -ForegroundColor Cyan
    Push-Location $webui; try { npm run build } finally { Pop-Location }
}

$port = if ($Mode -eq 'prod') { 8000 } else { 5173 }
$servers = @()
Write-Host '[tasni] starting backend...' -ForegroundColor Green
$servers += Start-Process -FilePath 'py' -ArgumentList '-3.10', '-m', 'tasni', '--port', '8000' `
    -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput "$env:TEMP\tasni-backend.out.log" `
    -RedirectStandardError "$env:TEMP\tasni-backend.err.log"
if ($Mode -eq 'dev') {
    Write-Host '[tasni] starting web UI (dev)...' -ForegroundColor Green
    $servers += Start-Process -FilePath $env:ComSpec -ArgumentList '/c', 'npm run dev' `
        -WorkingDirectory $webui -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput "$env:TEMP\tasni-vite.out.log" `
        -RedirectStandardError "$env:TEMP\tasni-vite.err.log"
}

try {
    if (-not (Wait-Port $port)) {
        Write-Warning "server didn't come up on :$port — see $env:TEMP\tasni-*.log"
    }
    $url = "http://localhost:$port"
    $browser = Find-Browser
    if ($browser) {
        Write-Host "[tasni] Tasni is running. Close the app window to stop." -ForegroundColor Green
        $app = Start-Process -FilePath $browser -PassThru -ArgumentList `
            "--app=$url --user-data-dir=`"$appProfile`" --no-first-run --no-default-browser-check"
        $app.WaitForExit()
    } else {
        Start-Process $url
        Write-Host "[tasni] No Chrome/Edge found — opened in your default browser." -ForegroundColor Yellow
        Write-Host "[tasni] Close this window (Ctrl-C) to stop the servers." -ForegroundColor Yellow
        $servers[0].WaitForExit()
    }
} finally {
    Write-Host "[tasni] stopping servers..." -ForegroundColor Cyan
    foreach ($p in $servers) {
        try { if (-not $p.HasExited) { taskkill /PID $p.Id /T /F | Out-Null } } catch {}
    }
    # Robust fallback (also covers child processes if taskkill is unavailable):
    # kill whatever still owns our ports + any tasni-tagged node/python.
    Stop-PriorInstances
}
