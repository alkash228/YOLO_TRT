# YOLO_DRT Docker API

GPU FastAPI service for YOLO detect/pose + helmet PPE + identity tracking.

## Default identity pipeline

| Setting | Env | Default |
|---------|-----|---------|
| SAM masklet F2F | `YOLO_DRT_USE_SAM_IDENTITY` | `true` |
| Live OSNet gallery | `YOLO_DRT_USE_REID` | `false` |
| Offline Pass 2 link | `YOLO_DRT_USE_OFFLINE_TRACKLET_LINK` | `true` |
| OSNet in Pass 2 only | `YOLO_DRT_TRACKLET_LINK_USE_REID` | `true` |
| SAM OSNet re-entry | `YOLO_DRT_SAM_OSNET_REENTRY` | `false` |
| Helmet cross-check | `YOLO_DRT_CROSS_CHECK_ENABLED` | `true` |

- **Pass 1 (live):** YOLO track + `SamMemoryTracker` (IoU / motion_id). OSNet is **not** loaded every frame.
- **Pass 2 (end of video):** `tracklet_linker` may load OSNet from `YOLO_DRT_REID_MODEL` to merge long-gap tracklets.
- **Helmet / PPE:** unchanged (`cross_check_*`).

OSNet `.pth` must still be present under `/data/models/RD/` (baked at image build) for Pass 2.

## Run

```bash
cd YOLO_DOCKER
# Put weights in ./models (YOLO pose, Helmet, RD/osnet_*.pth, optional TRT/*.engine)
docker compose build
docker compose up -d
curl http://localhost:8080/health
```

Upload a job:

```bash
curl -X POST http://localhost:8080/v1/jobs/upload \
  -F "file=@clip.mp4" \
  -F "prompt=person"
```

## Fall back to classic live OSNet ReID

Disable SAM identity and enable live ReID (old gallery path):

```yaml
# docker-compose.yml environment (or .env)
YOLO_DRT_USE_SAM_IDENTITY: "false"
YOLO_DRT_USE_REID: "true"
YOLO_DRT_USE_OFFLINE_TRACKLET_LINK: "false"   # optional
```

Or one-shot:

```bash
docker compose run --rm \
  -e YOLO_DRT_USE_SAM_IDENTITY=false \
  -e YOLO_DRT_USE_REID=true \
  yolo-drt-api
```

With live ReID, TensorRT will also expect / build an OSNet `.engine` at startup.

## Disable Pass 2 only

Keep SAM F2F, skip offline linking:

```bash
YOLO_DRT_USE_OFFLINE_TRACKLET_LINK=false
```

## Notes

- Settings are loaded once at container start from `YOLO_DRT_*` env (`api/env.py` → `PipelineSettings`).
- This tree is a Docker fork of `app/`; identity modules live in `app/core/sam_memory_tracker.py` and `app/core/tracklet_linker.py`.
- See `.env.example` for the full env template.
