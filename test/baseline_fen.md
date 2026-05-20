# Test FEN Baseline

Model active: **v3** (`models/backups/items_v3.pt`, training mAP50 = 0.949)

Backup v4: `boarddetection/models/items_items_v4.pt` (mAP50 = 0.937, palace-bottom +25% nhưng palace-conner -23%)

## Detection Results

| # | Image | Detected FEN | Correct FEN | Match? |
|---|---|---|---|---|
| 1 | 1.png | `1rbakabnr/9/1cn3c2/pR2p1p1p/2p6/9/P1P1P1P1P/2N1C2C1/9/2BAKABNR w` | `1rbakabnr/9/1cn3c2/pR2p1p1p/2p6/9/P1P1P1P1P/2N1C2C1/9/2BAKABNR w` | ✅ |
| 2 | 2.png | `2rak1bnr/4a4/1c2b1c2/1R6p/5N3/2p6/P3P1P1P/4C2C1/9/2BAKA1NR w` | `2rak1bnr/4a4/1c2b1c2/1R6p/5N3/2p6/P3P1P1P/4B2C1/9/2BAKA1NR w` | ⚠️ 1 piece sai: row 7 col 4 (C→B) |
| 3 | 3.png | `r1bakabnr/9/1cn4c1/p1p1p1p1p/9/9/P1P1P1P1P/2N1C2C1/9/R1BAKABNR w` | `r1bakabnr/9/1cn4c1/p1p1p1p1p/9/9/P1P1P1P1P/2N1C2C1/9/R1BAKABNR w` | ✅ EXACT (rotated 90°) |
| 4 | 4.png | `2bak3r/4a4/2n1bcc2/p1p1p1N1p/9/2P6/P3P3P/2N5B/1r2A4/4KABR1 w` | `2bak3r/4a4/2n1bcc2/p1p1p1N1p/9/2P6/P3P3P/2N1B3C/1r2A4/3RKABR1 w` | ⚠️ Thiếu 2 pieces (R@9,3 và B@7,4); B@7,8 should be C |
| 5 | 5.png | `3aka3/9/2n1b1cP1/p3p3c/5n3/6p2/P3P3P/2NCB4/4A4/3A2B2 w` | `2Raka3/3r5/2n1b1cR1/p3p3c/5n3/6p2/P3P3P/2NCB4/4A4/3AK1B2 w` | ⚠️ Thiếu 3 pieces (R@0,2 + r@1,3 + K@9,4) + 1 misclass (P@2,7 nên R) |
| 6 | 6.png | `2bak4/4a4/4b1c2/p3p1C1p/2pn5/P5P1n/c1P1P3N/2C1B4/3NA4/4KAB2 w` | `2bak4/4a4/4b1c2/p3p1C1p/2pn5/P5P1n/c1P1P3N/2C1B4/3NA4/4KAB2 w` | ✅ EXACT |
| 7 | 7.png | `2bak4/4a4/4b1c2/p4C2p/9/P1p1pnP1n/cCP1P3N/4B4/3NA4/4KAB2 w` | `2bak4/4a4/4b1c2/p4C2p/9/P1p1pnP1n/cCP1P3N/4B4/3NA4/4KAB2 w` | ✅ EXACT |
| 8 | 8.png | `2bakabr1/9/1c4nc1/3rp1p1p/p1p6/5NP2/n1P1P3P/N2CC4/R8/2BAKABR1 w` | `2bakabr1/9/1c4nc1/3rp1p1p/p1p6/5NP2/n1P1P3P/N2CC4/R8/2BAKABR1 w` | ✅ EXACT |
| 9 | 10.jpg | `2baka1r1/9/n1c1b1n2/p1p1p3p/6p2/1NP6/P3P1P1P/4B1N2/CR6R/3AKAB2 w` | `2baka1r1/1r7/n1c1b1n1c/p1p1p3p/6p2/1NP6/P3P1P1P/C3B1N2/CR6R/3AKAB2 w` | ⚠️ Thiếu 3 pieces (r@1,1 + c@2,8 + C@7,0) |
| 10 | 11.jpg | `R7b/9/9/B7p/3C3n1/7p1/3CP4/7Nb/5PNn1/3P3c1 w` | `3rka1r1/4a4/1c2b1n1c/p3pNR1p/1nb3p2/6P2/P3P3P/2C1C4/9/RNBAKAB2 w` | ❌ HOÀN TOÀN SAI (rotation failure, conf 0.68 thấp) |
| 11 | 12.jpg | `4kab2/4a4/9/3n1b3/3R1pN2/1p2p3n/1c4P2/5P3/4CB3/c4A3 w` | `4kab2/4a4/2n1bc3/2R1p1N2/p2r4n/c4NP2/4P4/3CB2C1/4A4/2BAK4 w` | ❌ Row 0,1 đúng; row 2-9 lệch cells (grid scale off?) |

## Summary

| Status | Count | Tests |
|---|---|---|
| ✅ EXACT | 5 | 1, 3, 6, 7, 8 |
| ⚠️ 1-3 lệch | 3 | 2 (1 misclass), 4 (2 missing + 1 misclass), 10 (3 missing) |
| ⚠️ Lệch nhiều | 1 | 5 (3 missing + 1 misclass) |
| ❌ Hoàn toàn sai | 2 | 11 (rotation), 12 (grid scale) |

**Accuracy:** 5/11 = 45% exact match. 8/11 = 73% có ít nhất palace structure đúng.

## Common error patterns

1. **Missing red chariot (R)** at corner positions — appears in test/4, test/5, test/10
2. **Missing palace pieces** (general, advisor) — test/5 missing K, test/10 missing pieces in palace area
3. **Misclassifications** at row 7 (palace-conner area) — test/2 (C→B), test/4 (B at wrong col)
4. **Landscape rotation** — test/5, test/11, test/12 (low confidence ~0.7)

## Notes
- Per-piece detection counts and confidences in `baseline_v3_fen.txt`
- Visualizations available in `output/*_visualization.png`
