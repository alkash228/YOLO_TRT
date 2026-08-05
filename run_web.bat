@echo off
cd /d "%~dp0.."
set YOLO_DRT_API_URL=http://127.0.0.1:8080
set WEB_APP_HOST=127.0.0.1
set WEB_APP_PORT=8088
python -m WEB_app.server
