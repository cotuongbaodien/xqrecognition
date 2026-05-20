# Test Results — v6 + Convex Hull Algorithm

Model: `items_v6` (commit `90a8905`)
Algorithm: convex hull → 4-vertex polygon on (pieces + board-conner + palace-bottom + board-border)

## Summary

| Status | Count | Tests |
|---|---|---|
| ✅ **EXACT** | 5 | 1, 3, 6, 7, 8 |
| ⚠️ Lệch 1-2 pieces | 1 | 4 |
| ⚠️ Lệch một chút | 1 | 2 |
| ❌ Grid sai (priority fix) | 4 | 5, 10, 11, 12 |

**Cell accuracy:** ước ~70-75% trên 11 ảnh.

## Detail

### ✅ test/1 — EXACT
- **Detected:** `1rbakabnr/9/1cn3c2/pR2p1p1p/2p6/9/P1P1P1P1P/2N1C2C1/9/2BAKABNR w`
- **Correct:** giống detected

### ⚠️ test/2 — 1 piece misclass + 1 missing
- **Detected:** `2rak1bnr/4a4/1c4c2/1R6p/5N3/2p6/P3P1P1P/4C2C1/9/2BAKA1NR w`
- **Correct:** `2rak1bnr/4a4/1c2b1c2/1R6p/5N3/2p6/P3P1P1P/4B2C1/9/2BAKA1NR w`
- Row 2: thiếu `b` (elephant) @ col 4 → `1c4c2` vs `1c2b1c2`
- Row 7: `C` @ col 4 nên là `B` (cannon→elephant misclass)

### ✅ test/3 — EXACT
- **Detected:** `r1bakabnr/9/1cn4c1/p1p1p1p1p/9/9/P1P1P1P1P/2N1C2C1/9/R1BAKABNR w`

### ⚠️ test/4 — row 7 sai
- **Detected:** `2bak3r/4a4/2n1bcc2/p1p1p1N1p/9/2P6/P3P3P/2N5B/1r2A4/3RKABR1 w`
- **Correct:** `2bak3r/4a4/2n1bcc2/p1p1p1N1p/9/2P6/P3P3P/2N1B3C/1r2A4/3RKABR1 w`
- Row 7: `2N5B` vs `2N1B3C` → thiếu B@col4, B@col8 should be C
- Row 9: `3RKABR1` ✅ FIXED từ v3 (đã có R@col3)

### ❌ test/5 — grid lệch (priority)
- **Detected:** `3aka3/3b5/2n1b1c1R/p3p3c/5n3/6p2/P3P3P/2NCC4/4A4/3AK1B2 w`
- **Correct:** `2Raka3/3r5/2n1b1cR1/p3p3c/5n3/6p2/P3P3P/2NCB4/4A4/3AK1B2 w`
- Row 0: thiếu R@col2 → `3aka3` vs `2Raka3`
- Row 1: `3b5` vs `3r5` (b→r misclass)
- Row 2: R@col 8 nên là col 7 → `2n1b1c1R` vs `2n1b1cR1`
- Row 7: C@col4 nên là B → `2NCC4` vs `2NCB4`

### ✅ test/6 — EXACT
- `2bak4/4a4/4b1c2/p3p1C1p/2pn5/P5P1n/c1P1P3N/2C1B4/3NA4/4KAB2 w`

### ✅ test/7 — EXACT
- `2bak4/4a4/4b1c2/p4C2p/9/P1p1pnP1n/cCP1P3N/4B4/3NA4/4KAB2 w`

### ✅ test/8 — EXACT
- `2bakabr1/9/1c4nc1/3rp1p1p/p1p6/5NP2/n1P1P3P/N2CC4/R8/2BAKABR1 w`

### ❌ test/10 — grid lệch (priority)
- **Detected:** `2bak4/1r4n2/n1c6/2p1p4/p5p2/1NP6/4P1P1C/P5N2/B7R/CR1AKAB2 w`
- **Correct:** `2baka1r1/1r7/n1c1b1n1c/p1p1p3p/6p2/1NP6/P3P1P1P/C3B1N2/CR6R/3AKAB2 w`
- Row 0: thiếu a@col5, r@col7
- Row 1: r@col 1 ✓ (ko thừa n@col6)
- Row 2: thiếu b@col4, n@col6, c@col8
- Row 3-9: nhiều cells lệch

### ❌ test/11 — board tilted, grid hoàn toàn sai (priority)
- **Detected:** `2cp1P3/r8/1nRpP3B/a1N5A/k4P2K/a7A/8B/9/1c1n4N/5P2R w`
- **Correct:** `3rka1r1/4a4/1c2b1n1c/p3pNR1p/1nb3p2/6P2/P3P3P/2C1C4/9/RNBAKAB2 w`
- Board nghiêng ~30°, grid algorithm không track rotation

### ❌ test/12 — perspective lệch (priority)
- **Detected:** `4kab2/4a4/2n1bc3/4p1N2/2R6/1p1p2P1n/1c3P3/7C1/r3CA3/r3AK3 w`
- **Correct:** `4kab2/4a4/2n1bc3/2R1p1N2/p2r4n/c4NP2/4P4/3CB2C1/4A4/2BAK4 w`
- Row 0-2 đúng
- Row 3 lệch: `4p1N2` vs `2R1p1N2` (thiếu R@col2)
- Row 4-9: nhiều cells lệch perspective

## Priority cần fix grid

Các test 5, 10, 11, 12 sai do grid construction chưa tận dụng đủ board-border points
để fit edges chính xác. User insight cần implement:

> "phải tận dụng tất cả các board border board conner palace bottom để fill thành cái khung của grid, bắt buộc tất cả phải nằm ở rìa của grid"
> "ko phải lúc nào cũng có edge đâu, mình vẫn phải dự đoán dựa vào các điểm xung quanh"

Approach: line-fit per edge + edge prediction khi thiếu detection.
