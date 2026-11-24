@echo off
rem 控制台后端常驻进程（由任务计划程序或 start-console.cmd 调用）
rem 仅起 uvicorn；端口固定 8010，前端 dist 由后端静态托管。
setlocal
set "PY=%USERPROFILE%\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
if not exist "%PY%" (
  echo [ERROR] python not found: %PY%
  exit /b 1
)
cd /d "%~dp0webapp\backend"
"%PY%" -m uvicorn main:app --host 127.0.0.1 --port 8010
endlocal
