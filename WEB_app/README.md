# WEB_app — веб-интерфейс YOLO_DRT

Светлый UI: загрузка видео → анализ на API → клипы NO HELMET → **Word-акт** по каждому нарушителю.

## Как это связано с API

WEB **не считает** YOLO сам. Он:

1. Проксирует запросы на API (`/proxy/health`, `/proxy/jobs/upload`, …).
2. После job читает папку прогона с диска (`run_dir`) и локально собирает MP4 и `.docx`.

Поэтому API и WEB должны видеть **одну и ту же** `output` (на Windows — одна машина; в Docker — общий volume `../output` у обоих контейнеров).

## Запуск на Windows

**Терминал 1 — API:**

```powershell
cd D:\Projects\SAM3_construction\YOLO_DRT
.\run_api.bat
# или: python -m api.main  →  :8765
```

**Терминал 2 — WEB:**

```powershell
.\run_web_app.bat
# или: python -m WEB_app.server  →  :8088
```

Открой http://127.0.0.1:8088 (Ctrl+F5 после обновления статики).

## Запуск в Docker

Из каталога `YOLO_DOCKER`:

```bash
docker compose up -d
# API:  http://localhost:8080
# WEB:  http://localhost:8088
```

Код WEB в образе синхронизируется скриптом `YOLO_DOCKER/scripts/sync_from_root.ps1`.

## Word-отчёт

- Компания-тест: ООО «Обнал» / ООО «Рога и копыта»
- Дата и время инцидента — в форме на шаге 2
- В документ: организация, нарушение NO HELMET, ID, статистика, **фото** (средний кадр трека)

Зависимость: `pip install python-docx` (есть в `WEB_app/requirements.txt` и в Docker API image).

## Маршруты WEB

| Route | Назначение |
|-------|------------|
| `GET /` | UI |
| `POST /proxy/jobs/upload` | → API upload |
| `GET /proxy/jobs/{id}` | прогресс job |
| `POST /build-video` | клипы по violator ID |
| `POST /report/word` | акт Word |
| `GET /report/violators` | список ID после прогона |
| `GET /videos/...` | раздача MP4 |

## Переменные окружения

| Variable | Default |
|----------|---------|
| `YOLO_DRT_API_URL` | авто :8765, затем :8080 |
| `WEB_APP_HOST` | `0.0.0.0` |
| `WEB_APP_PORT` | `8088` |
