@echo off
REM Convenience wrapper so you can double-click or run `start.bat [prod]`
REM without worrying about PowerShell execution policy.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
