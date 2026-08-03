@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [WEB_app] Virtual env not found: %PY%
    echo Run: scripts\setup_venv.ps1
    pause
    exit /b 1
)

set "WEB_HOST=127.0.0.1"
set "WEB_PORT=8088"

REM Prefer Docker API :8080, then desktop embedded API :8765.
set "API_URL="
"%PY%" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2)" 1>nul 2>nul
if not errorlevel 1 set "API_URL=http://127.0.0.1:8080"
if "%API_URL%"=="" (
    "%PY%" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=2)" 1>nul 2>nul
    if not errorlevel 1 set "API_URL=http://127.0.0.1:8765"
)
if "%API_URL%"=="" set "API_URL=http://127.0.0.1:8080"

echo [WEB_app] Installing deps if needed...
"%PY%" -m pip install -q -r "%~dp0WEB_app\requirements.txt"
if errorlevel 1 (
    echo [WEB_app] pip install failed.
    pause
    exit /b 1
)

echo.
echo [WEB_app] URL:  http://%WEB_HOST%:%WEB_PORT%/
echo [WEB_app] API:  %API_URL%  ^(Docker :8080 or desktop :8765^)
echo.

start "" "http://%WEB_HOST%:%WEB_PORT%/"

set "YOLO_DRT_API_URL=%API_URL%"
set "WEB_APP_HOST=%WEB_HOST%"
set "WEB_APP_PORT=%WEB_PORT%"

"%PY%" -m WEB_app.server
if errorlevel 1 (
    echo.
    echo [WEB_app] Server exited with error.
    pause
    exit /b 1
)

endlocal
