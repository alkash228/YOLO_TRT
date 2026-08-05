@echo off
cd /d "%~dp0.."

set YOLO_DRT_API_URL=http://127.0.0.1:8080
set WEB_APP_HOST=127.0.0.1
set WEB_APP_PORT=8088

echo WEB: http://127.0.0.1:8088
echo API: %YOLO_DRT_API_URL%
echo.

where python >nul 2>nul
if errorlevel 1 (
  echo Python not found in PATH.
  pause
  exit /b 1
)

python -m pip install -q -r WEB_app\requirements.txt
python -m WEB_app.server
pause
