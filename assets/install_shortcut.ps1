<#
  Create a "Tasni" shortcut on the Desktop that launches the control panel
  (start.ps1, dev mode) with the on-brand icon. Re-run any time to refresh it.

    powershell -ExecutionPolicy Bypass -File assets\install_shortcut.ps1
#>
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot           # assets\ -> repo root
$ico = Join-Path $repo 'assets\tasni.ico'
$ps = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
$desktop = [Environment]::GetFolderPath('Desktop')
$lnk = Join-Path $desktop 'Tasni.lnk'

$shell = New-Object -ComObject WScript.Shell
$sc = $shell.CreateShortcut($lnk)
$sc.TargetPath = $ps
$sc.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$repo\start.ps1`""
$sc.WorkingDirectory = $repo
$sc.IconLocation = "$ico,0"
$sc.Description = 'Tasni - robotic fabrication control panel'
$sc.WindowStyle = 1
$sc.Save()

Write-Host "Created shortcut: $lnk" -ForegroundColor Green
Write-Host "  target : $ps" -ForegroundColor DarkGray
Write-Host "  runs   : start.ps1 (dev) in $repo" -ForegroundColor DarkGray
Write-Host "  icon   : $ico" -ForegroundColor DarkGray
