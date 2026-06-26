# Release 2026-06-26 — DEPLOYED

Snapshot of the production models. Copy these into `boarddetection/models/` to
deploy. `.pt` = GPU (ultralytics), `.onnx` = CPU (onnxruntime, `OCR_MODEL_FORMAT=onnx`).

| Model | Source | Arch | imgsz | Notes |
|---|---|---|---|---|
| `items.pt` | items_v20 | YOLO11s | 960 | 14 pieces + 4 landmarks, PIECE_CONF 0.25 |
| `board_seg.pt` | board_seg_v6_synth500 | YOLO11n-seg | 640 | board + palace; synth-filled empty boards |
| `items.onnx` | export of items.pt | — | 960 | opset 12 |
| `board_seg.onnx` | export of board_seg.pt | — | 640 | opset 12 |

## Bench (test/bench, 243 imgs = 219 straight + 24 skewed)
| Backend | exact-FEN |
|---|---|
| torch `.pt` (GPU) | 229/243 (207 straight / 22 skewed) |
| ONNX (CPU) | 230/243 |

Remaining fails: xe↔ma + tướng↔tượng opening misclass, a few skewed
classification/miss (244, 248). Not grid/board_seg issues.

## Provenance (backups in `models/backups/`)
- items: `items_v20_snap.pt`, `items_v20.onnx`
- board_seg: `board_seg_v6_synth500.pt`, `board_seg_v6_synth500.onnx`
- previous prod (rollback): `board_seg_predeploy_2026-06-26.pt` (v5)

## Re-export ONNX after retraining a `.pt`
See `boarddetection/ONNX.md`. VPS CPU deploy: `ocr-gpu-service/docs/onnx-cpu-vps-handoff.md`.
