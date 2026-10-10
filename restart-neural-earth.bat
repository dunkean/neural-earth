@echo off
rem Kill every running Neural Earth server (and its child processes), then start a fresh one.
setlocal
cd /d "%~dp0"
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Create .venv and install dependencies first; see docs/installation.md.
    pause
    exit /b 1
)

echo Stopping Neural Earth servers...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ids = @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'terrain_server\.py|run_terrain_benchmark_server\.py|launch_terrain\.py' } | ForEach-Object ProcessId);" ^
  "$ids += @(Get-NetTCPConnection -State Listen -LocalPort 8765,8766 -ErrorAction SilentlyContinue | ForEach-Object OwningProcess);" ^
  "$ids = @($ids | Where-Object { $_ -and $_ -ne $PID } | Sort-Object -Unique);" ^
  "if (-not $ids) { Write-Host '  none running' }" ^
  "foreach ($id in $ids) { Write-Host ('  killing PID ' + $id); taskkill /T /F /PID $id 2>&1 | Out-Null }" ^
  "for ($i = 0; $i -lt 40 -and (Get-NetTCPConnection -State Listen -LocalPort 8765 -ErrorAction SilentlyContinue); $i++) { Start-Sleep -Milliseconds 250 }"

echo Starting a new Neural Earth server...
"%~dp0.venv\Scripts\python.exe" "%~dp0launch_terrain.py" --gpu %*
if errorlevel 1 pause
