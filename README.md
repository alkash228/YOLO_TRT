# YOLO_DRT — Docker

Один контейнер **`yolo-drt-api`**: FastAPI + ядро пайплайна (`api/` + `app/`). GPU inference, очередь job'ов, результаты на диск.

Образ: **`yolo-drt-api:latest`** (тег в `docker-compose.yml`). Краткие команды пересборки: **[BUILD.md](BUILD.md)**.

---

## Что внутри образа

| Слой | Содержимое |
|------|------------|
| База | `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`, Python 3.11, ffmpeg |
| ML | PyTorch (nightly cu128), `requirements-api.txt`, deep-person-reid |
| Код | `app/`, `api/`, `config/` (профиль `ui_fast_profile.json`) |
| Модели | `COPY models → /data/models` при **build** (не mount) |
| Процесс | `uvicorn api.main:app --host 0.0.0.0 --port 8080` |

Порт контейнера: **8080**. Swagger: http://localhost:8080/docs

**Identity / ReID = person only.** Helmet (`helmet-26m.pt`) is a per-frame cross-check accessory detector (person∩helmet; missing det → `NO HELMET`) — no track IDs, no ReID embeds.

---

## Требования на хосте

- **Docker** + **Docker Compose**
- **NVIDIA Container Toolkit** (`runtime: nvidia`, одна GPU)
- Перед `docker compose build` — веса в `YOLO_DOCKER/models/`:
  - `YOLO/yolo26x.pt` (person bbox detect; legacy pose: `YOLO/yolo26x-pose.pt` optional)
  - `YOLO/Helmet/helmet-26m.pt` (helmet accessories — not tracked / not ReID)
  - `RD/solider_swin_small_msmt17.pth` (person Pass2 / re-entry; default)
  - optional rollback: `RD/osnet_ain_x1_0_…pth`
  - по желанию готовые TRT: `models/TRT/*.engine` (иначе соберутся при первом старте внутри контейнера)

---

## Сборка и запуск

```bash
cd YOLO_DOCKER
docker compose build
docker compose up -d --force-recreate yolo-drt-api
docker compose logs -f yolo-drt-api   # первый старт: возможна долгая сборка TRT
curl http://localhost:8080/health
```

Остановка:

```bash
docker compose down          # volume yolo_work / yolo_trt сохраняются
docker compose down -v       # + удалить named volumes (uploads/staging/TRT)
```

Пересборка после смены кода или моделей:

```bash
cd YOLO_DOCKER
docker compose build         # --no-cache если меняли models/ или Dockerfile
docker compose up -d --force-recreate yolo-drt-api
```

---

## Volumes и пути

| В контейнере | Откуда | Зачем |
|--------------|--------|--------|
| `/data/models` | **в образе** | Веса и TRT |
| `/data/output` | `../output` (bind) | Run-папки, JSON, manifest, `_source.mp4` |
| `/data/work` | volume **`yolo_work`** | Upload, staging, быстрый decode (Linux disk) |
| `/data/videos` | `./videos:ro` (опционально) | Path-job без upload |

Переменные каталогов (можно переопределить в `docker-compose.yml` или `.env`):

- `YOLO_DRT_OUTPUT_DIR=/data/output`
- `YOLO_DRT_WORK_DIR=/data/work`
- `YOLO_DRT_UPLOAD_DIR=/data/work/uploads`
- `YOLO_DRT_MODELS_DIR=/data/models`

На хосте артефакты job'ов лежат в **`../output`** относительно `YOLO_DOCKER/` (тот же bind, что `/data/output` в контейнере).

---

## API: как гнать видео

**Upload:**

```bash
curl -F "file=@clip.mp4" -F "prompt=person" http://localhost:8080/v1/jobs/upload
```

**Path (файл уже в контейнере):**

```bash
# положить clip.mp4 в YOLO_DOCKER/videos/
curl -X POST http://localhost:8080/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"path":"/data/videos/clip.mp4","prompt":"person"}'
```

Статус job: `GET /v1/jobs/{job_id}`. Артефакты: `GET /v1/jobs/{job_id}/artifacts`. Список прогонов — в `/data/output` на хосте.

**Длинные ролики (часы):** не гоняй через upload — клади файл в `YOLO_DOCKER/videos/` и path `/data/videos/...`. API **не копирует** файлы с `/data/videos` и файлы &gt; `YOLO_DRT_STAGE_MAX_COPY_GB` (по умолчанию 2 GiB) в volume — иначе «не грузится» из‑за копирования/диска.

**Админка (человекочитаемо):** http://localhost:8080/admin  
JSON: `GET /v1/admin/status`, отмена `POST /v1/admin/jobs/latest/cancel`, рестарт контейнера `POST /v1/admin/restart` (`{"mode":"docker"}` или `"exit"`).  
Логи контейнера как `.txt`: `GET /v1/admin/logs?tail=3000` (нужен `/var/run/docker.sock`).

Сборка итогового MP4 из контейнера по умолчанию **выключена** (`YOLO_DRT_ENCODE_MODE=manual`); JSON и manifest пишутся в run-папку.

---

## Настройки пайплайна в Docker

Источник по умолчанию: `config/ui_fast_profile.json` + дубли в `docker-compose.yml` → `api/env.py` (`YOLO_DRT_*`).

Кратко (профиль fast):

| Параметр | Значение |
|----------|----------|
| Pass 1 ID | SAM identity + **CPU identity gallery** (SQLite spill); SOLIDER re-entry без хранения эмбеддингов в VRAM |
| Pass 2 | Offline tracklet link + SOLIDER, **gap=unlimited** (весь ролик) |
| Rollback Pass2 | `YOLO_DRT_REID_BACKEND=osnet` + OSNet `.pth` in `models/RD/` |
| Кадры | `windowed`, batch **64**, TRT max batch **32** |
| Каска | person∩helmet cross-check (no ReID on helmet), сглаживание нарушений |
| WEB clip-by-ID | host `WEB_app/` — clips/reports per person `stable_id` (not in API image) |
| RAM (host в контейнере) | smart budget **10 GB**, окно до **4 GB**, preload cap **12 GB** |

Полный список env — `.env.example` и блок `environment:` в `docker-compose.yml`.

Пример смены только через compose (перезапуск контейнера):

```yaml
environment:
  YOLO_DRT_INFER_BATCH_SIZE: "32"
  YOLO_DRT_MAX_PROCESS_RAM_GB: "10.0"
```

Жёсткого лимита VRAM в env **нет** — упирается в карту, batch и TRT engines.

---

## Память и стабильность между job'ами

- **Host RAM:** `YOLO_DRT_SMART_RAM_BUDGET=true`, `YOLO_DRT_MAX_PROCESS_RAM_GB=10.0` — размер окон decode и очередей.
- **VRAM:** между job'ами pose TRT остаётся загруженным (скорость следующего прогона); полный сброс GPU — `docker compose restart yolo-drt-api`.
- Upload/decode идут в **`yolo_work`**, не в bind-mount Windows — так быстрее на Docker Desktop.

---

## Структура каталога

```
YOLO_DOCKER/
  Dockerfile
  docker-compose.yml
  BUILD.md             → rebuild / recreate commands
  requirements-api.txt
  .env.example
  models/              → COPY в образ → /data/models
  videos/              → optional bind → /data/videos
  app/                 → ядро pipeline
  api/                 → FastAPI
  config/              → ui_fast_profile.json
  WEB_app/             → host UI (clip-by-ID); not in API image
  notebooks/           → docker_api_benchmark.ipynb
```

---

## Отладка

| Симптом | Что проверить |
|---------|----------------|
| `health` не отвечает | `docker compose logs`, GPU driver + nvidia runtime |
| OOM при infer | уменьшить `YOLO_DRT_INFER_BATCH_SIZE` / `YOLO_DRT_TRT_MAX_BATCH`, пересобрать TRT |
| Нет run на хосте | bind `../output`, job в статусе `done`, manifest в run-папке |
| Медленный decode | upload через API (work volume), не класть большие файлы только в `./videos` bind без staging |
| Странное поведение после правок кода | пересборка образа `docker compose build` |

Бенч API: `notebooks/docker_api_benchmark.ipynb`.

---

## Ссылки

- Swagger: http://localhost:8080/docs
- TensorRT / batch по VRAM: `../models/TRT/TENSORRT_INSTRUCTIONS.md`

## SOLIDER Pass2 ReID weights

Download MSMT17 Swin-S checkpoint into `models/RD/solider_swin_small_msmt17.pth` (also under `YOLO_DOCKER/models/RD/` before image build):

```bash
python -m gdown "https://drive.google.com/uc?id=1C-aIZdFyjFsZX4W4feG-Ex39RU2Qvu3b" -O models/RD/solider_swin_small_msmt17.pth
```

Rollback to OSNet: `YOLO_DRT_REID_BACKEND=osnet` and point `YOLO_DRT_REID_MODEL` at the OSNet `.pth`.

