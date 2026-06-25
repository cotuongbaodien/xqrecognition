"""Update one bench ground-truth entry: python scripts/set_gt.py <board> '<fen>'

Used to correct wrong GT entries the user spots during the visual audit.
Appends ' w - - 0 1' if only the board part is given. Backs up once.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT = os.path.join(ROOT, "test", "bench", "ground_truth.txt")


def main():
    board, fen = sys.argv[1], sys.argv[2].strip()
    if " " not in fen:
        fen += " w - - 0 1"
    if not os.path.exists(GT + ".audit_bak"):
        import shutil
        shutil.copy(GT, GT + ".audit_bak")
    lines = open(GT, encoding="utf-8").read().splitlines()
    done = False
    for i, l in enumerate(lines):
        if l.strip().startswith(board) and ":" in l and \
                l.split(":", 1)[0].strip() == board:
            lines[i] = f"{board}: {fen}"
            done = True
            break
    if not done:
        lines.append(f"{board}: {fen}")
    open(GT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"GT[{board}] = {fen}")


if __name__ == "__main__":
    main()
