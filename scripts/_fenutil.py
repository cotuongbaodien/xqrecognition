"""Helper FEN dùng chung cho các script.

Ba hàm dưới đây đang bị chép lại ở `eval_fen.py`, `eval_bench.py` và vài script
khác. Script mới import từ đây thay vì chép bản thứ tư. (Các script cũ giữ
nguyên — đổi chúng là việc dọn dẹp riêng, không gộp vào tính năng mới.)

Quy ước: mọi hàm nhận FEN **chỉ phần bàn cờ** hoặc FEN có đuôi lượt đi; phần
sau khoảng trắng luôn bị bỏ. Bàn 10 hàng x 9 cột, ô trống là ".".
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from boarddetection.settings import GRID_COLS, GRID_ROWS, STARTING_FEN  # noqa: E402


def expand_rows(fen):
    """FEN -> list 10 hàng, mỗi hàng 9 ký tự ('.' = trống). Vá cả FEN méo."""
    rows = []
    for row in fen.split()[0].split("/"):
        cells = []
        for ch in row:
            cells += ["."] * int(ch) if ch.isdigit() else [ch]
        rows.append((cells + ["."] * GRID_COLS)[:GRID_COLS])
    while len(rows) < GRID_ROWS:
        rows.append(["."] * GRID_COLS)
    return rows[:GRID_ROWS]


def compress_rows(rows):
    """Nghịch đảo expand_rows."""
    out = []
    for row in rows:
        s, empty = "", 0
        for ch in row:
            if ch == ".":
                empty += 1
            else:
                if empty:
                    s += str(empty)
                    empty = 0
                s += ch
        out.append(s + (str(empty) if empty else ""))
    return "/".join(out)


def mirror_fen(fen):
    """Soi gương trái-phải (pipeline KHÔNG chuẩn hoá chiều ngang)."""
    return compress_rows([r[::-1] for r in expand_rows(fen)])


def cell_diff(fen_a, fen_b):
    """Số ô khác nhau giữa hai thế cờ (0-90)."""
    ra, rb = expand_rows(fen_a), expand_rows(fen_b)
    return sum(
        ra[i][j] != rb[i][j]
        for i in range(GRID_ROWS)
        for j in range(GRID_COLS)
    )


def dist_to_start(fen):
    """Số ô lệch so với thế khai cuộc, đã tính cả bản soi gương.

    Trả 99 khi FEN rỗng. Đây là tín hiệu chính để nhận ra "bàn vừa xếp lại":
    thế khai cuộc cho 0, tàn cuộc thường 30-40.
    """
    if not fen:
        return 99
    return min(cell_diff(fen, STARTING_FEN), cell_diff(mirror_fen(fen), STARTING_FEN))


def dist_to_start_farside(fen):
    """Như trên nhưng so với thế khai cuộc ĐỔI MÀU (bàn đọc ngược đầu).

    Thế khai cuộc quay 180° = chính nó nhưng hoán đổi đen/đỏ. Nếu tầng chuẩn hoá
    hướng có lúc nào đó hụt, FEN sẽ lệch **32 ô** so với thế khai cuộc — rơi đúng
    vào dải 30-40 của "không phải khai cuộc" nên im lặng trôi qua. Đo riêng khoảng
    cách này thì biến ca đó thành một cảnh báo nhìn thấy được.
    """
    if not fen:
        return 99
    far = STARTING_FEN.swapcase()
    return min(cell_diff(fen, far), cell_diff(mirror_fen(fen), far))


def piece_count(fen):
    return sum(ch not in ".0123456789/" for ch in fen.split()[0]) if fen else 0
