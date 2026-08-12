# ReID weights

| File | Backend |
|------|---------|
| `solider_swin_small_msmt17.pth` | Pass2 default (`reid_backend=solider`) |
| `osnet_ain_x1_0_msmt17_…pth` | Rollback (`YOLO_DRT_REID_BACKEND=osnet`) |

Download SOLIDER Swin-S (MSMT17):

```bash
python -m gdown "https://drive.google.com/uc?id=1C-aIZdFyjFsZX4W4feG-Ex39RU2Qvu3b" -O solider_swin_small_msmt17.pth
```

Source: [tinyvision/SOLIDER-REID](https://github.com/tinyvision/SOLIDER-REID) Swin Small MSMT17 table.
