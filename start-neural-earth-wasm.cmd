@echo off
cd /d "%~dp0"
if not exist wasm\dist\index.html (
  call npm run build:wasm
  if errorlevel 1 exit /b 1
)
call npm run serve:wasm
