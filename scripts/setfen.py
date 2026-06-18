"""Set a FEN for a numbered entry in test/bench/ground_truth.txt.
Validates 10 ranks x 9 cols. Usage: python scripts/setfen.py <num> "<fen>"
"""
import sys
from pathlib import Path

num = sys.argv[1].strip()
fen = sys.argv[2].strip()
rows = fen.split()[0].split("/")
assert len(rows) == 10, f"INVALID: {len(rows)} ranks"
for i, r in enumerate(rows):
    s = sum(int(c) if c.isdigit() else 1 for c in r)
    assert s == 9, f"INVALID rank {i+1} = {s}"

p = Path(__file__).parent.parent / "test/bench/ground_truth.txt"
lines = p.read_text(encoding="utf-8").splitlines()
done = False
for i, ln in enumerate(lines):
    if ":" in ln and not ln.strip().startswith("#") and ln.split(":", 1)[0].strip() == num:
        lines[i] = f"{num}: {fen}"
        done = True
        break
p.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"{'wrote' if done else 'NOT FOUND'} {num} (valid 10x9)")
