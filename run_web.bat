@echo off
setlocal
cd /d "%~dp0"

set YOLO_DRT_API_URL=http://127.0.0.1:8080
set WEB_APP_HOST=127.0.0.1
set WEB_APP_PORT=8088

if not exist "WEB_app\server.py" (
  echo [ERROR] WEB_app\server.py not found in:
  echo   %CD%
  pause
  exit /b 1
)

echo WEB: http://127.0.0.1:%WEB_APP_PORT%/
echo API: %YOLO_DRT_API_URL%
echo DIR: %CD%
echo.

py -3 -m WEB_app.server 2>nul
if not errorlevel 1 goto :eof

python -m WEB_app.server
if errorlevel 1 (
  echo.
  echo [ERROR] Launch failed. Python must be installed and in PATH.
  pause
  exit /b 1
)
endlocal
