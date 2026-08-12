# Rebuild `yolo-drt-api` image

From this directory (`YOLO_DOCKER/`). Do **not** change volume mounts unless you intend to.

## Weights before build (required)

Place files under `./models/` (they are `COPY`'d into the image at `/data/models`):

| Path | Role |
|------|------|
| `models/YOLO/yolo26x.pt` | Person detect (bbox) |
| `models/YOLO/Helmet/helmet-26m.pt` | Helmet cross-check (accessories only — no ReID) |
| `models/RD/solider_swin_small_msmt17.pth` | Person Pass2 / re-entry ReID (SOLIDER) |
| `models/RD/osnet_ain_x1_0_….pth` | Optional OSNet rollback |
| `models/TRT/*.engine` | Optional prebuilt TRT (else built on first run) |

SOLIDER download (if missing):

```bash
python -m gdown "https://drive.google.com/uc?id=1C-aIZdFyjFsZX4W4feG-Ex39RU2Qvu3b" -O models/RD/solider_swin_small_msmt17.pth
```

## Rebuild + recreate API container

```bash
cd YOLO_DOCKER
docker compose build
docker compose up -d --force-recreate yolo-drt-api
docker compose logs -f yolo-drt-api
curl http://localhost:8080/health
```

Code-only change (no weight changes):

```bash
docker compose build
docker compose up -d --force-recreate yolo-drt-api
```

Weights / Dockerfile base change — clean rebuild:

```bash
docker compose build --no-cache
docker compose up -d --force-recreate yolo-drt-api
```

## Identity vs helmet (runtime)

- **Person:** `yolo26x.pt` → BoT-SORT / SAM identity / optional SOLIDER-OSNet ReID → `stable_id`
- **Helmet:** `helmet-26m.pt` → per-frame `predict` → person∩helmet verdict (`NO HELMET` if missing) — **no** track IDs, **no** ReID

`WEB_app/` is host-side UI (clip-by-ID); it is not baked into this API image.
