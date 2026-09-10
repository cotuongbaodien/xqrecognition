"""Soát lại các clip đã cắt: có đọc được không, độ dài có khớp index.csv không.

Tiến trình cắt bị giết giữa chừng để lại file **ghi dở** — tên đúng, dung lượng có
vẻ hợp lý, mở lên vẫn chạy, nhưng cụt. Không có cách nào biết ngoài đối chiếu độ dài
thật với độ dài đáng lẽ phải có (`index.csv` ghi sẵn lúc cắt).

  python scripts/video_verify.py "E:\\videos\\GiangHo"
  python scripts/video_verify.py "E:\\videos\\GiangHo" --fix     # xoá clip hỏng + index

`--fix` chỉ xoá clip hỏng và `index.csv` của thư mục đó, KHÔNG đụng `00_goc_*`; lần
chạy `video_split.py` sau sẽ tự cắt lại thư mục đó (vì mất index.csv).
"""
import argparse
import csv
import json
import os
import subprocess
import sys

VIDEO_EXT = (".mp4", ".mov", ".mkv", ".avi", ".m4v")


def probe_dur(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


def check_folder(d, tol):
    """Trả (số clip, danh sách lỗi)."""
    idx = os.path.join(d, "index.csv")
    if not os.path.exists(idx):
        return 0, [("thiếu index.csv (video chưa cắt xong)", "")]
    rows = list(csv.DictReader(open(idx, encoding="utf-8")))
    clips = sorted(f for f in os.listdir(d)
                   if "_van" in f and f.lower().endswith(VIDEO_EXT)
                   and not f.endswith(".shrink.mp4"))
    bad = []
    if len(clips) != len(rows):
        bad.append((f"số clip {len(clips)} khác index.csv {len(rows)}", ""))
    for row in rows:
        name = row.get("file", "")
        # Clip đã nén xong thì đổi đuôi sang .mp4 -> tìm theo phần thân tên.
        stem = os.path.splitext(name)[0]
        found = next((c for c in clips if os.path.splitext(c)[0] == stem), None)
        if not found:
            bad.append((f"mất file {name}", name))
            continue
        p = os.path.join(d, found)
        if os.path.getsize(p) < 10000:
            bad.append((f"file rỗng/quá nhỏ {found}", found))
            continue
        want = float(row.get("dai_giay") or 0)
        got = probe_dur(p)
        if got is None:
            bad.append((f"ffprobe không đọc được {found}", found))
        elif want and want - got > tol:
            # CHỈ báo khi NGẮN hơn. Dài hơn là bình thường: cắt copy stream lùi về
            # keyframe trước đó nên clip luôn dư ở đầu, không bao giờ thiếu.
            bad.append((f"CỤT: {found} dài {got:.0f}s, đáng lẽ {want:.0f}s", found))
    return len(clips), bad


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--tol", type=float, default=8.0,
                    help="lệch độ dài bao nhiêu giây thì coi là cụt (copy stream lùi "
                         "về keyframe nên lệch vài giây là bình thường)")
    ap.add_argument("--fix", action="store_true",
                    help="xoá clip hỏng + index.csv để lần cắt sau làm lại thư mục đó")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    dirs = sorted(os.path.join(root, d) for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)) and not d.startswith("_"))
    tot_clip = 0
    bad_dirs = []
    for d in dirs:
        n, bad = check_folder(d, args.tol)
        tot_clip += n
        if bad:
            bad_dirs.append((d, bad))
            print(f"\n{os.path.basename(d)[:60]}")
            for msg, _f in bad:
                print(f"   {msg}")

    print(f"\n{'=' * 70}")
    print(f"{len(dirs)} thư mục · {tot_clip} clip · {len(bad_dirs)} thư mục có vấn đề")

    if args.fix and bad_dirs:
        n_del = 0
        for d, bad in bad_dirs:
            for _msg, f in bad:
                p = os.path.join(d, f)
                if f and os.path.exists(p):
                    os.remove(p)
                    n_del += 1
            idx = os.path.join(d, "index.csv")
            if os.path.exists(idx):
                os.remove(idx)
        print(f"đã xoá {n_del} clip hỏng + index.csv của {len(bad_dirs)} thư mục "
              f"-> chạy lại video_split.py để cắt lại (bản gốc còn nguyên)")


if __name__ == "__main__":
    main()
