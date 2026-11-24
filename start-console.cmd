@echo off
chcp 65001 >nul
setlocal
set "PY=%USERPROFILE%\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
set "URL=http://127.0.0.1:8010/#/workbench"
if not exist "%PY%" (
  echo [ERROR] 找不到 python: %PY%
  pause
  exit /b 1
)
cd /d "%~dp0webapp\backend"

rem 端口已在监听就跳过启动，避免起出两个进程抢 8010（旧进程还占着时新进程会静默失败）
netstat -ano | findstr /c:"127.0.0.1:8010" | findstr /c:"LISTENING" >nul 2>&1
if errorlevel 1 (
  echo [1/3] 启动控制台后端 127.0.0.1:8010 ...
  start "" /b "%PY%" -m uvicorn main:app --host 127.0.0.1 --port 8010
) else (
  echo [1/3] 8010 已有后端在跑，跳过启动
  echo        ^(改过后端代码时请先关掉旧窗口再双击本文件^)
)

rem 用 /api/health 探活，比固定等 8 秒可靠：起得慢就多等，起不来就露报错
echo [2/3] 等待后端就绪 ...
set /a N=0
:wait
set /a N+=1
"%PY%" -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8010/api/health',timeout=1)" >nul 2>&1
if not errorlevel 1 goto ready
if %N% GEQ 30 goto failed
timeout /t 1 >nul
goto wait

:ready
echo       后端就绪（约 %N% 秒）
if not defined HC_NO_BROWSER start "" "%URL%"
echo [3/3] 页面：%URL%
echo.
echo 关掉这个窗口即停止后端；重新打开请再双击本文件。
pause >nul
exit /b 0

:failed
echo.
echo [ERROR] 30 秒内后端没起来。下面是被吞掉的报错（端口被占 / 依赖缺失都会显示在这里）：
echo ----------------------------------------------------------------
"%PY%" -m uvicorn main:app --host 127.0.0.1 --port 8010
echo ----------------------------------------------------------------
pause
exit /b 1
