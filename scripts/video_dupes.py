"""Tìm file video TRÙNG TUYỆT ĐỐI trong kho (cùng dung lượng + cùng nội dung).

Kho gom từ nhiều nguồn — file cục bộ, bản tải về từ YouTube, thư mục TikTok — nên
rất dễ có cùng một file nằm ở hai ba chỗ dưới tên khác nhau.

Cách nhận trùng, ba tầng, tầng sau chỉ chạy trên thứ đã qua tầng trước:
  1. **Dung lượng byte y hệt** — lọc thô, gần như miễn phí.
  2. **Băm 4 MB đầu + 4 MB cuối** — bắt gần hết ca trùng thật mà không phải đọc cả file.
  3. **Băm toàn bộ file** (chỉ khi `--full`) — chắc chắn tuyệt đối, nhưng đọc cả TB.

Chỉ khi qua đủ tầng mới coi là trùng. KHÔNG bao giờ xoá dựa vào tên file: hai bản
cùng nội dung thường có tên khác hẳn (tên gốc vs tiêu đề YouTube).

  python scripts/video_dupes.py                       # báo cáo
  python scripts/video_dupes.py --delete              # xoá bản thừa, GIỮ bản ưu tiên

Thứ tự giữ lại khi trùng (`--keep-order`): thư mục nào đứng trước trong danh sách
--root thì bản nằm ở đó được giữ; cùng thư mục thì giữ file có tên ngắn hơn.
"""
import argparse
import hashlib
import os
import sys
from collections import defaultdict

VIDEO_EXT = (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".ts", ".webm", ".flv")
DEFAULT_ROOTS = [
    r"E:\videos\GiangHo",
    r"E:\videos\_yt_tai",
    r"D:\videos\_yt_tai",
    r"E:\Tiktoker\cotuongnghiasing",
]
EDGE = 4 * 1024 * 1024


def edge_hash(path, size):
    """Băm 4 MB đầu + 4 MB cuối. File nhỏ hơn 8 MB thì băm cả file."""
    h = hashlib.sha1()
    try:
        with open(path, "rb") as fh:
            if size <= 2 * EDGE:
                h.update(fh.read())
            else:
                h.update(fh.read(EDGE))
                fh.seek(-EDGE, os.SEEK_END)
                h.update(fh.read(EDGE))
    except OSError:
        return None
    return h.hexdigest()


def full_hash(path):
    h = hashlib.sha1()
    try:
        with open(path, "rb") as fh:
            for blk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
                h.update(blk)
    except OSError:
        return None
    return h.hexdigest()


def collect(roots, include_clips):
    files = []
    for root in roots:
        if not os.path.isdir(root):
            print(f"bỏ qua (không thấy): {root}")
            continue
        for dp, _dn, fs in os.walk(root):
            if os.path.basename(dp) == "_data":
                continue
            for f in fs:
                if not f.lower().endswith(VIDEO_EXT):
                    continue
                if not include_clips and "_van" in f:
                    continue      # clip cắt ra, không phải file nguồn
                p = os.path.join(dp, f)
                try:
                    files.append((p, os.path.getsize(p)))
                except OSError:
                    pass
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", default=None)
    ap.add_argument("--include-clips", action="store_true",
                    help="xét cả clip `*_vanNN_*` (mặc định chỉ xét file nguồn)")
    ap.add_argument("--full", action="store_true",
                    help="băm TOÀN BỘ file để chắc tuyệt đối (đọc rất nhiều, chậm)")
    ap.add_argument("--min-mb", type=float, default=1.0,
                    help="bỏ qua file nhỏ hơn (MB)")
    ap.add_argument("--delete", action="store_true",
                    help="xoá bản thừa, giữ bản ở thư mục ưu tiên cao nhất")
    args = ap.parse_args()

    roots = args.root or DEFAULT_ROOTS
    prio = {os.path.abspath(r).lower(): i for i, r in enumerate(roots)}
    files = collect(roots, args.include_clips)
    files = [(p, s) for p, s in files if s >= args.min_mb * 1e6]
    print(f"{len(files)} file video, "
          f"{sum(s for _, s in files) / 1e9:.0f} GB\n")

    by_size = defaultdict(list)
    for p, s in files:
        by_size[s].append(p)
    cand = {s: ps for s, ps in by_size.items() if len(ps) > 1}
    print(f"tầng 1 — trùng dung lượng: {sum(len(v) for v in cand.values())} file "
          f"trong {len(cand)} nhóm")

    groups = []
    for s, ps in cand.items():
        by_h = defaultdict(list)
        for p in ps:
            h = edge_hash(p, s)
            if h:
                by_h[h].append(p)
        for h, g in by_h.items():
            if len(g) > 1:
                groups.append((s, g))
    print(f"tầng 2 — trùng cả đầu+cuối: {sum(len(g) for _, g in groups)} file "
          f"trong {len(groups)} nhóm")

    if args.full and groups:
        conf = []
        for s, g in groups:
            by_h = defaultdict(list)
            for p in g:
                fh = full_hash(p)
                if fh:
                    by_h[fh].append(p)
            conf += [(s, gg) for gg in by_h.values() if len(gg) > 1]
        groups = conf
        print(f"tầng 3 — băm toàn bộ: {sum(len(g) for _, g in groups)} file "
              f"trong {len(groups)} nhóm")

    waste = sum(s * (len(g) - 1) for s, g in groups)
    print(f"\nDƯ THỪA: {waste / 1e9:.1f} GB có thể thu hồi\n")

    def rank(p):
        ap_ = os.path.abspath(p).lower()
        best = min((i for r, i in prio.items() if ap_.startswith(r)), default=99)
        return (best, len(p))

    for s, g in sorted(groups, key=lambda x: -x[0])[:40]:
        g = sorted(g, key=rank)
        print(f"{s / 1e6:8.1f} MB  x{len(g)}")
        for i, p in enumerate(g):
            print(f"   {'GIỮ ' if i == 0 else 'thừa'} {p}")
    if len(groups) > 40:
        print(f"... còn {len(groups) - 40} nhóm nữa")

    if args.delete and groups:
        n = 0
        freed = 0
        for s, g in groups:
            for p in sorted(g, key=rank)[1:]:
                try:
                    os.remove(p)
                    n += 1
                    freed += s
                except OSError as e:
                    print(f"không xoá được {p}: {e}")
        print(f"\nđã xoá {n} bản thừa, thu hồi {freed / 1e9:.1f} GB")
    elif groups:
        print("\n(báo cáo thôi — thêm --delete để xoá bản thừa)")


if __name__ == "__main__":
    main()
