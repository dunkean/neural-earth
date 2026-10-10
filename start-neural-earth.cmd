@echo off
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" "%~dp0launch_terrain.py" --gpu %*
if errorlevel 1 pause
