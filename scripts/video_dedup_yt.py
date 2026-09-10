"""Đối chiếu video trên YouTube với kho cục bộ để biết thật sự phải tải bao nhiêu.

Kênh `@Cotuongnghiasing` là bản backup của chính đám file cục bộ, nên phần lớn video
trên đó đã có sẵn ở máy. Tải hết 1575 giờ tốn ~1 TB, mà có thể quá nửa là trùng.

Đối chiếu bằng **độ dài** (±--tol giây). Đây là dấu hiệu mạnh nhưng KHÔNG chắc chắn:
  * cùng độ dài mà khác nội dung -> rất hiếm với video dài chục phút, nhưng có thể;
  * cùng nội dung mà khác độ dài -> RẤT HAY GẶP: file cục bộ nhiều cái là "part1..5"
    cắt ra từ một buổi live dài, còn YouTube giữ nguyên buổi live.
Vì vậy kết quả nên đọc là "chắc chắn phải tải" (không khớp gì) và "nhiều khả năng đã
có" (khớp), chứ đừng coi là tuyệt đối.

  # 1. lấy danh sách kênh (kèm video ẩn — PHẢI qua uploads playlist)
  yt-dlp --flat-playlist --cookies C:\\Users\\PC\\.yt_cookies_work.txt \\
         --extractor-args "youtubetab:skip=authcheck" \\
         --print "%(id)s|%(duration)s|%(title)s" \\
         "https://www.youtube.com/playlist?list=UUs5-T_IsPmBEBwiGuzyw-vw" > yt.txt

  # 2. đối chiếu (lần đầu quét ffprobe cả kho, sau đó dùng cache)
  python scripts/video_dedup_yt.py yt.txt --out can_tai.txt
"""
import argparse
import json
import os
import subprocess
import sys
import time

VIDEO_EXT = (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".ts", ".webm", ".flv")
DEFAULT_ROOTS = [
    r"E:\videos\GiangHo",
    r"E:\Tiktoker\cotuongnghiasing\download",
    r"E:\Tiktoker\cotuongnghiasing\tiktoklive\raw",
    r"E:\Tiktoker\cotuongnghiasing\gianghorecord",
]


def probe_dur(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", path], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


def local_index(roots, cache_path, refresh=False):
    """{đường dẫn: độ dài}. Có cache vì ffprobe cả nghìn file rất lâu.

    Lấy CẢ `00_goc_*` nằm trong thư mục kết quả (đó chính là video gốc đã dời vào),
    nhưng BỎ các clip `*_vanNN_*` — chúng là mảnh cắt ra, không phải nguồn.
    """
    cache = {}
    if os.path.exists(cache_path) and not refresh:
        try:
            cache = json.load(open(cache_path, encoding="utf-8"))
        except Exception:
            cache = {}
    files = []
    for root in roots:
        if not os.path.isdir(root):
            print(f"bỏ qua (không thấy): {root}")
            continue
        for dp, _dn, fs in os.walk(root):
            if os.path.basename(dp) == "_data":
                continue
            for f in fs:
                if not f.lower().endswith(VIDEO_EXT) or "_van" in f:
                    continue
                files.append(os.path.join(dp, f))
    todo = [p for p in files if p not in cache]
    print(f"kho cục bộ: {len(files)} file ({len(todo)} cái cần đọc độ dài)")
    t0 = time.time()
    for i, p in enumerate(todo, 1):
        cache[p] = probe_dur(p)
        if i % 100 == 0:
            print(f"  {i}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
            json.dump(cache, open(cache_path, "w", encoding="utf-8"))
    json.dump(cache, open(cache_path, "w", encoding="utf-8"))
    return {p: d for p, d in cache.items() if d and p in set(files)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("yt_list", help="file `id|duration|title` xuất từ yt-dlp")
    ap.add_argument("--root", action="append", default=None,
                    help="thư mục kho cục bộ (lặp lại được)")
    ap.add_argument("--tol", type=float, default=3.0, help="lệch bao nhiêu giây thì coi là một")
    ap.add_argument("--cache", default="output/local_durations.json")
    ap.add_argument("--refresh", action="store_true", help="đọc lại độ dài, bỏ cache")
    ap.add_argument("--out", default="output/yt_can_tai.txt",
                    help="ghi danh sách id CHƯA có ở máy")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.cache)), exist_ok=True)
    loc = local_index(args.root or DEFAULT_ROOTS, args.cache, args.refresh)
    # gom theo giây tròn để tra nhanh thay vì so từng cặp (1028 x 700 phép so)
    buckets = {}
    for p, d in loc.items():
        buckets.setdefault(int(round(d)), []).append(p)

    yt = []
    for ln in open(args.yt_list, encoding="utf-8", errors="replace"):
        parts = ln.strip().split("|")
        if len(parts) < 2:
            continue
        try:
            yt.append((parts[0], float(parts[1]), parts[2] if len(parts) > 2 else ""))
        except ValueError:
            continue

    matched, missing = [], []
    for vid, dur, title in yt:
        hit = None
        for s in range(int(round(dur - args.tol)), int(round(dur + args.tol)) + 1):
            if s in buckets:
                hit = buckets[s][0]
                break
        (matched if hit else missing).append((vid, dur, title, hit))

    h_all = sum(d for _, d, _, _ in matched + missing) / 3600
    h_miss = sum(d for _, d, _, _ in missing) / 3600
    print(f"\nYouTube      : {len(yt)} video · {h_all:.0f} giờ")
    print(f"đã có ở máy  : {len(matched)} video · {h_all - h_miss:.0f} giờ")
    print(f"CHƯA có      : {len(missing)} video · {h_miss:.0f} giờ "
          f"(~{h_miss * 1.5 / 8 * 3.6:.0f} GB nếu tải 720p)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for vid, dur, title, _ in sorted(missing, key=lambda x: x[1]):
            fh.write(f"{vid}\t{dur:.0f}\t{title}\n")
    print(f"-> {args.out} ({len(missing)} id, sắp NGẮN TRƯỚC)")
    print("\ntải bằng:  yt-dlp -a <file id> --cookies ... "
          "(cột 1 là id, yt-dlp nhận thẳng)")


if __name__ == "__main__":
    main()
