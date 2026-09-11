"""Tải video YouTube theo lô, tự chuyển ổ đĩa khi ổ chính sắp đầy.

Dùng cho kho `@Cotuongnghiasing` (xem `docs/VIDEO_SOURCES.md`). Danh sách id lấy từ
`scripts/video_dedup_yt.py` — đã sắp NGẮN TRƯỚC nên tải tới đâu dùng được tới đó.

  python scripts/yt_download.py output/yt_can_tai.txt

Vì sao chia lô thay vì gọi `yt-dlp -a` một phát:
  * trước mỗi lô mới **đo lại chỗ trống** và đổi ổ nếu cần — chạy một phát thì đường
    dẫn cố định từ đầu, đầy ổ là chết giữa chừng;
  * yt-dlp chết (mạng, YouTube chặn) thì chỉ mất lô đó, lô sau vẫn chạy tiếp.

Sổ `_archive.txt` **dùng chung cho mọi ổ** nên video đã tải không bao giờ tải lại,
dù nằm ở ổ khác.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

FMT = ("bv*[height<=720][vcodec^=avc1]+ba[ext=m4a]/"
       "b[height<=720][ext=mp4]/bv*[height<=720]+ba/b")


def pick_dir(dirs, need_gb):
    """Thư mục đầu tiên còn đủ chỗ. Hết sạch thì trả None."""
    for d in dirs:
        drive = os.path.splitdrive(os.path.abspath(d))[0] + os.sep
        try:
            free = shutil.disk_usage(drive).free / 1e9
        except OSError:
            continue
        if free >= need_gb:
            return d, free
    return None, 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("list_file", help="file `id<TAB>duration<TAB>title` hoặc mỗi dòng 1 id/URL")
    ap.add_argument("--dir", action="append",
                    default=None, help="thư mục tải, lặp lại để xếp thứ tự ưu tiên")
    ap.add_argument("--archive", default=r"E:\videos\_yt_tai\_archive.txt",
                    help="sổ video đã tải, DÙNG CHUNG cho mọi ổ")
    ap.add_argument("--cookies-master", default=r"C:\Users\PC\.yt_cookies_master.txt")
    ap.add_argument("--cookies-work", default=r"C:\Users\PC\.yt_cookies_work.txt")
    ap.add_argument("--chunk", type=int, default=20, help="mỗi lô bao nhiêu video")
    ap.add_argument("--min-free", type=float, default=150,
                    help="ổ còn dưới bấy nhiêu GB thì chuyển sang ổ kế tiếp")
    ap.add_argument("--limit", type=int, default=0, help="chỉ tải N video đầu")
    args = ap.parse_args()

    dirs = args.dir or [r"E:\videos\_yt_tai", r"D:\videos\_yt_tai"]
    ids = []
    for ln in open(args.list_file, encoding="utf-8", errors="replace"):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        vid = ln.split("\t")[0].split("|")[0].strip()
        if vid:
            ids.append(vid if vid.startswith("http")
                       else f"https://www.youtube.com/watch?v={vid}")
    if args.limit:
        ids = ids[:args.limit]
    os.makedirs(os.path.dirname(args.archive), exist_ok=True)
    done = set()
    if os.path.exists(args.archive):
        done = {l.split()[-1] for l in open(args.archive, encoding="utf-8") if l.split()}
    print(f"{len(ids)} video trong danh sách, sổ đã có {len(done)} cái", flush=True)

    t0 = time.time()
    n_ok = n_err = 0
    for i in range(0, len(ids), args.chunk):
        chunk = ids[i:i + args.chunk]
        out, free = pick_dir(dirs, args.min_free)
        if out is None:
            print(f"HẾT CHỖ trên mọi ổ (cần >{args.min_free:.0f} GB) -> dừng", flush=True)
            break
        os.makedirs(out, exist_ok=True)
        # Copy cookie mỗi lô: yt-dlp GHI ĐÈ file cookie sau khi chạy.
        try:
            shutil.copy(args.cookies_master, args.cookies_work)
        except OSError as e:
            print(f"không copy được cookie: {e}", flush=True)
        cmd = ["yt-dlp", "--cookies", args.cookies_work,
               "--extractor-args", "youtubetab:skip=authcheck",
               "-f", FMT, "--merge-output-format", "mp4",
               "-o", os.path.join(out, "%(upload_date)s_%(title).60B [%(id)s].%(ext)s"),
               "--download-archive", args.archive,
               "--no-overwrites", "--continue", "--ignore-errors", "--no-warnings",
               "--retries", "10", "--fragment-retries", "10",
               "--concurrent-fragments", "4", "--no-progress"] + chunk
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            n_ok += len(chunk)
        else:
            n_err += 1
            print(f"  lô lỗi (mã {r.returncode}): {r.stderr.strip()[-200:]}", flush=True)
        print(f"[{min(i + args.chunk, len(ids))}/{len(ids)}] -> {out} "
              f"(ổ còn {free:.0f} GB) · {(time.time() - t0) / 60:.0f} phút", flush=True)

    print(f"\nXONG: {n_ok} video, {n_err} lô lỗi, {(time.time() - t0) / 60:.0f} phút")
    for d in dirs:
        if os.path.isdir(d):
            gb = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)
                     if os.path.isfile(os.path.join(d, f))) / 1e9
            print(f"  {d}: {gb:.1f} GB")


if __name__ == "__main__":
    main()
