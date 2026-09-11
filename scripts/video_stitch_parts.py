"""Khâu lại những ván bị ĐỨT ĐÔI vì file gốc được chia thành nhiều phần.

Buổi live 8-9 tiếng thường bị chia nhỏ thành `..._p1, _p2, ...` cho dễ quản lý. Chỗ
chia rơi vào giữa ván thì ván đó ra HAI NỬA ở hai file khác nhau:

    phần N:   … [ván A đầy đủ] [ván B nửa đầu]   <- ván cuối chạm hết file
    phần N+1: [ván B nửa sau] [ván C đầy đủ] …   <- ván đầu là "dau-video"

Không ván nào mất, nhưng cả hai nửa đều vô dụng nếu muốn đăng nguyên ván.

Cách làm:
  1. Gom file theo tên (part/p/phan hoặc đuôi số 01,02…).
  2. Nghi đứt khi phần N có ván cuối **chạm hết file** VÀ phần N+1 có ván đầu là
     **dau-video** (bắt đầu giữa chừng).
  3. **Kiểm bằng THẾ CỜ**, không tin mỗi tên file: đọc bàn ở frame cuối nửa đầu và
     frame đầu nửa sau; cùng một ván thì hai thế phải gần như trùng nhau (vài nước
     là cùng lắm). Lệch nhiều nghĩa là chỗ chia rơi vào lúc nghỉ, không phải giữa ván.
  4. Nối hai nửa bằng `ffmpeg concat -c copy` (cùng nguồn nên cùng thông số).

  python scripts/video_stitch_parts.py --dry-run   # xem sẽ khâu gì
  python scripts/video_stitch_parts.py            # khâu thật

Clip khâu xong nằm ở `<root>\\_van_noi\\`; HAI NỬA GỐC ĐƯỢC GIỮ NGUYÊN, không xoá.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "scripts"))
VIDEO_EXT = (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".ts")
PATS = [re.compile(p, re.I) for p in [
    r"^(.*?)[ _-]*(?:part|phan|ph)[ _-]?(\d{1,2})$",
    r"^(.*?)[ _-]p(\d{1,2})$",
    r"^(.*?)[ _-]p(\d{1,2})[ _-].*$",
    r"^(.*?)(\d{2})$",
]]


def split_name(n):
    for pat in PATS:
        m = pat.match(n)
        if m and m.group(1).strip(" _-"):
            return m.group(1).strip(" _-").lower(), int(m.group(2))
    return None, None


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def clips_of(folder):
    """Clip trong thư mục, sắp theo số ván."""
    out = []
    for f in os.listdir(folder):
        if "_van" in f and f.lower().endswith(VIDEO_EXT) and not f.endswith(".shrink.mp4"):
            m = re.search(r"_van(\d+)_", f)
            out.append((int(m.group(1)) if m else 0, os.path.join(folder, f)))
    return [p for _, p in sorted(out)]


def frame_at(video, when, path):
    """Trích 1 frame: when='dau' lấy đầu, 'cuoi' lấy trước khi hết 3 giây."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    if when == "cuoi":
        r = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                  "-of", "csv=p=0", video])
        try:
            dur = float(r.stdout.strip())
        except ValueError:
            return False
        cmd += ["-ss", f"{max(0, dur - 3):.2f}"]
    cmd += ["-i", video, "-frames:v", "1", "-pix_fmt", "yuvj420p", "-q:v", "3",
            "-y", path]
    return _run(cmd).returncode == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=r"E:\videos\GiangHo")
    ap.add_argument("--max-diff", type=int, default=8,
                    help="hai thế cờ lệch quá bao nhiêu ô thì coi là KHÁC ván")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from boarddetection.pipeline import XiangqiRecognizer
    from _fenutil import cell_diff
    import cv2
    import numpy as np

    root = args.root
    groups = {}
    for n in sorted(os.listdir(root)):
        d = os.path.join(root, n)
        if not os.path.isdir(d) or n.startswith("_"):
            continue
        if not os.path.exists(os.path.join(d, "_data", "games.json")):
            continue
        b, i = split_name(n)
        if b:
            groups.setdefault(b, []).append((i, n))

    pairs = []
    for b, items in sorted(groups.items()):
        if len(items) < 2:
            continue
        items.sort()
        meta = []
        for i, n in items:
            d = os.path.join(root, n)
            g = json.load(open(os.path.join(d, "_data", "games.json"), encoding="utf-8"))
            sj = os.path.join(d, "_data", "scan.json")
            dur = json.load(open(sj, encoding="utf-8"))["duration"] \
                if os.path.exists(sj) else 0
            meta.append({"i": i, "name": n, "dir": d, "games": g, "dur": dur,
                         "cham_cuoi": bool(g and dur and dur - g[-1]["end"] < 30),
                         "dau_giua": bool(g and g[0].get("start_kind") == "dau-video")})
        for a, c in zip(meta, meta[1:]):
            if a["cham_cuoi"] and c["dau_giua"]:
                ca, cc = clips_of(a["dir"]), clips_of(c["dir"])
                if ca and cc:
                    pairs.append((b, a, c, ca[-1], cc[0]))

    print(f"{len(pairs)} chỗ nghi đứt ván\n")
    if not pairs:
        return
    rec = XiangqiRecognizer()
    out_dir = os.path.join(root, "_van_noi")
    os.makedirs(out_dir, exist_ok=True)
    tmp = tempfile.mkdtemp()
    rows = []
    for k, (b, a, c, tail, head) in enumerate(pairs, 1):
        f1, f2 = os.path.join(tmp, "a.jpg"), os.path.join(tmp, "b.jpg")
        ok = frame_at(tail, "cuoi", f1) and frame_at(head, "dau", f2)
        d = 99
        if ok:
            fens = []
            for f in (f1, f2):
                img = cv2.imdecode(np.fromfile(f, np.uint8), cv2.IMREAD_COLOR)
                r = rec.recognize_image_2pass(img, piece_confidence=0.25)
                fens.append((r.fen or "").split()[0])
            if all(fens):
                d = cell_diff(fens[0], fens[1])
        same = d <= args.max_diff
        name = f"{b[:50]}_noi_p{a['i']}-p{c['i']}.mp4"
        dst = os.path.join(out_dir, name)
        print(f"[{k}/{len(pairs)}] {b[:38]} p{a['i']}->p{c['i']}  "
              f"lệch {d if d < 99 else '?'} ô  "
              f"{'-> KHÂU' if same else '-> BỎ (khác ván)'}")
        rows.append({"nhom": b, "phan_truoc": a["name"], "phan_sau": c["name"],
                     "lech_o": d, "khau": int(same), "file": name if same else ""})
        if same and not args.dry_run and not os.path.exists(dst):
            lst = os.path.join(tmp, "list.txt")
            with open(lst, "w", encoding="utf-8") as fh:
                for p in (tail, head):
                    fh.write("file '" + p.replace("'", "'\\''") + "'\n")
            r = _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                      "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                      "-movflags", "+faststart", "-y", dst])
            if r.returncode != 0:
                print(f"    nối lỗi: {r.stderr.strip()[:160]}")
    with open(os.path.join(out_dir, "_ket_qua.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_ok = sum(r["khau"] for r in rows)
    print(f"\n{n_ok}/{len(rows)} chỗ là CÙNG MỘT VÁN -> "
          f"{'sẽ khâu' if args.dry_run else 'đã khâu'} vào {out_dir}")
    print("hai nửa gốc vẫn giữ nguyên, không xoá gì")


if __name__ == "__main__":
    main()
