# Weekly active-learning ingest

Grow `items_v19` from real OCR user-submissions each week, with a human review
gate so wrong pseudo-labels never reach the trainset un-checked.

```
weekly submitted imgs ──► v19 items.pt detect (conf≥0.25, imgsz960)
        │                         │ pseudo-labels (YOLO)
        ▼                         ▼
  data/incoming/<week>/    items_v19/incoming/{images,labels}/   (STAGING)
                                   │ build galleries
                                   ▼
                  data/label_review_incoming/<tag>/sheet_*.jpg
                                   │ human spot-fix (same flow as cleaning pass)
                                   ▼
                          merge → items_v19/train/  → retrain
```

## Steps

1. **Drop the week's images** into a folder, e.g. `data/incoming/2026-06-24/`.

2. **Ingest** (detect + stage + build review galleries). Run when GPU is free —
   detection at imgsz 960 is heavy and will contend with the prod OCR:
   ```
   python scripts/weekly_ingest.py --input data/incoming/2026-06-24
   ```
   → stages into `items_v19/incoming/`, galleries into
   `data/label_review_incoming/`. Prints pseudo-label class distribution.

3. **Review + fix** the new batch (same gallery workflow as the big clean pass,
   but pointed at the incoming review dir):
   ```
   python scripts/apply_review.py xeden '5,12=ma 7=phao' --dir data/label_review_incoming
   python scripts/apply_review.py totden '3=bo' --dir data/label_review_incoming   # bo = mark delete
   ```
   class tokens: `xe ma phao si tuong(voi) soai(tướng) tot` + color `-den/-do`
   (color defaults to the gallery's color; override e.g. `xe-do`, `phao-den`).

4. **Merge** into the trainset (purges `bo`/sentinel-99 lines, moves files):
   ```
   python scripts/weekly_ingest.py --merge
   ```

5. **Retrain** (batch 12 keeps ~9–10 GB so prod OCR coexists; see
   `scripts/train_items.py`):
   ```
   python scripts/train_items.py --data data/items_v19/data.yaml \
     --name items_vNEXT --img-size 960 --batch-size 12 --workers 2 --no-deploy
   ```
   Then add the new model to `scripts/eval_bench.py`, compare vs current
   (v19 = 136/224 on the 224-image FEN bench), deploy only if it beats it.

## Notes
- Pseudo-labels are only a starting point — the gallery review is where xe↔mã,
  tướng↔tượng etc. get corrected, exactly as in the v19 clean pass.
- Landmarks (board-conner/palace-*) are pseudo-labeled and kept for training but
  not surfaced in galleries (rarely mislabeled).
- Target: ~10–15k images for a near-perfect model (v19 ≈ 8.2k now).
- Staging dir `items_v19/incoming/` is NOT a train/val/test split, so it is
  ignored by training until `--merge`.
