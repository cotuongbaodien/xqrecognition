"""Generate diverse Xiangqi FENs for synthetic data (no eval leakage).

For a DETECTION model, what matters is positional diversity — every piece
class appearing at many cells, in many densities — not game legality. So we
place pieces randomly with per-side count caps, keeping generals inside their
palace (mild realism). This yields unlimited leakage-free positions (vs the
17 distinct in fendata, and bench/test which must NOT be used).

Usage: python scripts/gen_fens.py --n 2000 --out data/synth_fens.txt --seed 42
"""

import argparse
import random

# per-side max counts (lowercase = black; mirror for red uppercase)
CAPS = {"k": 1, "a": 2, "b": 2, "n": 2, "r": 2, "c": 2, "p": 5}
PALACE_COLS = [3, 4, 5]
BLACK_PALACE_ROWS = [0, 1, 2]
RED_PALACE_ROWS = [7, 8, 9]


def gen_board(rng):
    """Return 10x9 grid of fen symbols or None."""
    grid = [[None] * 9 for _ in range(10)]
    occupied = set()

    def free_cell(rows=None):
        for _ in range(60):
            r = rng.randint(0, 9) if rows is None else rng.choice(rows)
            c = rng.randint(0, 8)
            if (r, c) not in occupied:
                return r, c
        return None

    # generals in palace (1 each)
    for sym, rows in (("k", BLACK_PALACE_ROWS), ("K", RED_PALACE_ROWS)):
        for _ in range(40):
            r = rng.choice(rows); c = rng.choice(PALACE_COLS)
            if (r, c) not in occupied:
                grid[r][c] = sym; occupied.add((r, c)); break

    # other pieces: random count per type per side, random cells
    for color_upper in (False, True):
        for letter, cap in CAPS.items():
            if letter == "k":
                continue
            sym = letter.upper() if color_upper else letter
            n = rng.randint(0, cap)
            for _ in range(n):
                cell = free_cell()
                if cell:
                    r, c = cell
                    grid[r][c] = sym; occupied.add((r, c))
    return grid


def to_fen(grid):
    rows = []
    for r in grid:
        s, run = "", 0
        for cell in r:
            if cell is None:
                run += 1
            else:
                if run:
                    s += str(run); run = 0
                s += cell
        if run:
            s += str(run)
        rows.append(s or "9")
    return "/".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default="data/synth_fens.txt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    seen, out = set(), []
    while len(out) < args.n:
        fen = to_fen(gen_board(rng))
        if fen in seen:
            continue
        seen.add(fen); out.append(fen)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"Wrote {len(out)} unique FENs -> {args.out}")


if __name__ == "__main__":
    main()
