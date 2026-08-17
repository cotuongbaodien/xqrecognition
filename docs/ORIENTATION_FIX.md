# Orientation & Rotation — Root-cause Analysis + Complete Solution

Status: **design / not yet implemented** (2026-06-06). Discovered while
auditing the 20 hard-core test failures: many palace-piece (tướng/sĩ)
errors are actually *orientation* bugs, not piece-classification bugs.

> **Xem thêm (2026-07-20):** [`ROT180_TRAINING_GAP.md`](./ROT180_TRAINING_GAP.md) —
> một lớp lỗi orientation **khác**, nằm TRƯỚC toàn bộ pipeline mô tả ở đây. Với bàn cờ
> lật ngược (đỏ ở trên), YOLO detector phân loại sai con tướng ngay từ bước detect, nên
> FEN thiếu hẳn `k`/`K` và `server.py` trả `detected=false`. Hậu xử lý trong doc này có
> sửa cũng vô ích khi quân cờ chưa từng được detect.
>
> ⚠️ **Cập nhật 2026-08-17:** giả thuyết nguyên nhân của doc đó (lỗ hổng tập train,
> `flipud=0.0` + `degrees=45`) **đã bị số liệu bác bỏ** — train v20 vốn đã có 37,5% bàn
> lật, và prod fail 3,96% (đỏ dưới) vs 5,01% (đen dưới), p≈0,13. Xem §Đính chính trong
> `ROT180_TRAINING_GAP.md`.

User-observed symptoms:
1. Phone in **portrait** photographing a **landscape** board → whole board
   recognized in the wrong direction.
2. Even when the board (palace/cung) is detected, **general (tướng) and
   advisor (sĩ) end up reversed** relative to the palace.

---

## How orientation currently works (full trace)

```
pipeline.recognize()
 ├─ item_detector.detect()            → pieces + landmarks
 ├─ board_segmenter.get_board_quad()  → 4 corners (tl,tr,bl,br)   [board_segmenter.py:26]
 │     tl=min(x+y) br=max(x+y) tr=max(x-y) bl=min(x-y)   ← ordered by IMAGE position only
 ├─ ItemDetector.build_grid_from_quad()                          [item_detector.py:915]
 │     maps col 0..8 along tl→tr,  row 0..9 along tl→bl  ← FIXED axis assignment
 ├─ fen_generator.map_pieces_to_grid()  → board_state
 └─ Step 4: fen_generator.detect_board_orientation(board_state)  [fen_generator.py:367]
        if 'flipped': flip_board()      ← 180° vertical flip only, from PIECE COLORS
```

Key facts:
- The board is **9 files × 10 ranks** (height:width ≈ 9:8 ≈ **1.125**, taller than wide).
- Corner order is purely by **image position**; `build_grid_from_quad` **always**
  assigns the tl→tr edge to the 9 columns and tl→bl edge to the 10 rows.
- `detect_orientation_from_landmarks` is a **no-op** (always returns `'standard'`).
- The only real orientation logic is the **180° vertical flip** in
  `detect_board_orientation`, decided from **General (K/k) row, then colour centroids**.
- Horizontal mirror is **intentionally not handled** (left to the consuming app).

---

## Problem 1 — 90° rotation (portrait phone / landscape board)

**Root cause:** `build_grid_from_quad` forces `tl→tr = 9 cols` and `tl→bl = 10 rows`
regardless of the board's true orientation. When the board is captured landscape,
the **longer** physical edge (the 10-rank side) lies along tl→tr, but the code
still squeezes only 9 columns onto it and stretches 10 rows onto the **shorter**
edge → the entire lattice is transposed → garbage grid → wrong everything.

**Fix:** canonicalize the quad by edge length before grid building. Compute
```
W = |tl→tr|   (currently the cols axis)
H = |tl→bl|   (currently the rows axis)
```
The true upright board has `H/W ≈ 1.125` (rows axis longer). If `W > H` by a
margin → the capture is rotated 90° → **rotate the corner labels** so the longer
edge becomes the rows axis, e.g. 90° CW relabel:
```
(tl,tr,bl,br) → (bl, tl, br, tr)
```
(Pick CW vs CCW by which keeps row 0 at the image-top; ambiguity in the remaining
180° is resolved later by the vertical-flip step.) Implement in
`board_segmenter.get_board_quad()` so all downstream stays consistent; needs no
piece info, only the quad geometry.

Edge cases: near-square captures (aspect ~1) are the only fuzzy zone; use a
margin (e.g. only rotate if `W > 1.05·H`) and let the vertical-flip vote clean up.

---

## Problem 2 — General/advisor reversed (vertical 180° flip is fragile)

**Root cause:** `detect_board_orientation` trusts the **General position** first.
But the general is exactly one of the pieces the model often **misclassifies**
(tướng↔tượng, tướng↔sĩ in the hard-core boards). A single misread general →
wrong flip → the whole board is flipped → tướng/sĩ land in the **wrong-facing
palace**. The detected **palace landmarks are ignored** for this decision.

**Fix — robust vote** combining independent signals instead of trusting one general:
1. **General rows** (current): red K row vs black k row.
2. **Colour centroid** (already the fallback): mean row of UPPERCASE (red) vs
   lowercase (black) pieces — red should be the higher-row (bottom) half.
   Uses *all* pieces → robust to a few misreads.
3. **Palace constraint:** generals/advisors must sit in palace rows (0–2 / 7–9,
   cols 3–5). A general detected outside any palace is a red flag for a bad flip.

Decision = **majority vote**; on disagreement trust the **colour centroid**
(many pieces) over a single general. This needs no new data and cannot regress
boards where general + colours already agree.

> Note: red-vs-black truly requires piece **colours** — palace landmarks carry no
> colour, so they can validate the grid (already used) but cannot by themselves
> decide which palace is red. Colour voting is the robust path.

---

## Problem 3 — Horizontal mirror (note only, do NOT auto-fix)

A left-right mirrored Xiangqi position is geometrically valid, so auto-detecting
mirror is unreliable; current code defers it to the consuming app. Keep that.
Document so it isn't confused with Problems 1–2.

---

## Implementation plan

1. `board_segmenter.get_board_quad()` — add the 90° canonicalization (edge-length
   aspect test + corner relabel). ~10 lines.
2. `fen_generator.detect_board_orientation()` — replace single-general logic with
   the 3-signal majority vote. ~20 lines.
3. Optional: have the pipeline pass palace-bottom rows into the orientation vote
   for the palace-constraint signal.

## Testing / risk

- **Regression guard:** rerun the 86-image FEN eval (`detect.py` + `eval_fen.py`)
  — must not drop below the current best (v15 58 / v14 61). The vertical-flip
  vote is designed to be a no-op when signals already agree.
- **Targeted:** need a handful of **rotated/landscape captures** (user has phone
  portrait-vs-landscape examples) added to `test/` with ground truth to prove the
  90° fix. Without them the 90° path is untested.
- Both fixes are **code-only, no retraining** — independent of the weekly
  label-fix loop.

Related: hard-core boards 4, 20, 50, 65, 73 (palace-piece reversals) are the
prime suspects to recheck after the flip fix.
