@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0task07_demo\launch.ps1" %*
if errorlevel 1 pause
