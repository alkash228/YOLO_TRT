@echo off
setlocal EnableExtensions
cd /d "%~dp0"

REM WEB without host Python — needs Docker only.
REM API container must already listen on host :8080.

set "API_URL=http://host.docker.internal:8080"
set "WEB_PORT=8088"
set "OUT_DIR=%~dp0..\output"

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

echo [WEB] http://127.0.0.1:%WEB_PORT%/
echo [WEB] API -^> %API_URL%
echo [WEB] Ctrl+C to stop
echo.

docker run --rm -it ^
  --name yolo-drt-web ^
  -p %WEB_PORT%:8088 ^
  -e YOLO_DRT_API_URL=%API_URL% ^
  -e WEB_APP_HOST=0.0.0.0 ^
  -e WEB_APP_PORT=8088 ^
  -e YOLO_DRT_HOST_OUTPUT_DIR=/data/output ^
  -v "%~dp0WEB_app:/app/WEB_app:ro" ^
  -v "%~dp0app:/app/app:ro" ^
  -v "%OUT_DIR%:/data/output" ^
  -w /app ^
  python:3.11-slim-bookworm ^
  sh -c "pip install -q -r WEB_app/requirements.txt numpy opencv-python-headless imageio imageio-ffmpeg && python -m WEB_app.server"

if errorlevel 1 (
  echo.
  echo [WEB] Failed. Is Docker Desktop running? Is API on :8080?
  pause
)
endlocal
