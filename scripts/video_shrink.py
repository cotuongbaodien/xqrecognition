"""Nén các clip ván cờ cho nhẹ: HEVC/MOV bitrate cao -> H.264 mp4 720p.

Video iPhone (.MOV) trong kho là **HEVC 5-6 Mbps ở 1280x720**. Nặng không phải vì
độ phân giải mà vì bitrate. Encode lại H.264 ~1.5-2 Mbps thì:
  * nhẹ khoảng 3 lần (2,3-2,6 GB/giờ -> 0,7-0,9 GB/giờ)
  * mở được ở MỌI máy/trình duyệt, không riêng iPhone (HEVC trong .MOV hay kén)

Chạy SAU khi `video_split.py` đã cắt xong — nó chỉ đụng các file clip
`*_vanNN_*`, KHÔNG đụng `00_goc_*` (bản gốc để nguyên làm kho), trừ khi
`--include-source`.

  # xem sẽ nén những gì, tiết kiệm bao nhiêu — KHÔNG encode
  python scripts/video_shrink.py "E:\\videos\\GiangHo" --dry-run

  # nén thật (thay tại chỗ, chỉ thay khi bản mới đã kiểm tra hợp lệ)
  python scripts/video_shrink.py "E:\\videos\\GiangHo"

  # giữ lại file cũ thay vì thay tại chỗ
  python scripts/video_shrink.py "E:\\videos\\GiangHo" --keep-original

An toàn: encode ra file tạm, **so lại độ dài** với bản gốc (lệch quá --tol giây thì
bỏ, giữ nguyên file cũ), rồi mới thay. Dừng giữa chừng chạy lại được — file đã nhẹ
(bitrate dưới ngưỡng) sẽ tự bị bỏ qua.
"""
import argparse
import json
import os
import subprocess
import sys
import time

VIDEO_EXT = (".mp4", ".mov", ".mkv", ".avi", ".m4v")


def _run(cmd, timeout=None):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def probe(path):
    r = _run(["ffprobe", "-v", "error", "-select_streams", "v:0",
              "-show_entries", "stream=codec_name,width,height",
              "-show_entries", "format=duration,size,bit_rate",
              "-of", "json", path])
    if r.returncode != 0:
        return None
    j = json.loads(r.stdout or "{}")
    st = (j.get("streams") or [{}])[0]
    fm = j.get("format", {})
    dur = float(fm.get("duration", 0) or 0)
    size = int(fm.get("size", 0) or 0)
    br = float(fm.get("bit_rate", 0) or 0)
    if not br and dur:
        br = size * 8 / dur
    return {"codec": st.get("codec_name"), "w": st.get("width"), "h": st.get("height"),
            "dur": dur, "size": size, "mbps": br / 1e6}


def hhmm(sec):
    return f"{int(sec) // 3600:d}h{int(sec) % 3600 // 60:02d}"


def find_clips(root, include_source=False, min_age_sec=300):
    """Clip ván trong các thư mục con do video_split tạo ra.

    An toàn khi chạy SONG SONG với video_split đang cắt:
      * chỉ nhận thư mục đã có `index.csv` (= video đó cắt xong hẳn);
      * bỏ file vừa được ghi trong `min_age_sec` giây (đang ghi dở).
    Nén nhầm file đang ghi thì ra clip cụt mà vẫn "hợp lệ" — không có cách nào biết.
    """
    out = []
    for dirpath, _dirs, files in os.walk(root):
        if os.path.basename(dirpath) == "_data":
            continue
        if dirpath != root and "index.csv" not in files:
            continue                      # video đang cắt dở -> để yên
        now = time.time()
        for n in sorted(files):
            fp = os.path.join(dirpath, n)
            try:
                if now - os.path.getmtime(fp) < min_age_sec:
                    continue              # file còn nóng, có thể đang ghi
            except OSError:
                continue
            if not n.lower().endswith(VIDEO_EXT):
                continue
            is_source = n.startswith("00_goc_")
            if is_source and not include_source:
                continue
            if not is_source and "_van" not in n:
                continue        # không phải clip do video_split cắt -> đừng đụng
            out.append(fp)
    return out


def build_cmd(src, dst, args):
    """H.264, cao nhất 720p, CQ (chất lượng không đổi) + trần bitrate."""
    vf = f"scale='min({args.max_width},iw)':-2"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    if args.hw:
        cmd += ["-hwaccel", "auto"]
    cmd += ["-i", src, "-vf", vf, "-map", "0:v:0", "-map", "0:a?"]
    if args.encoder == "h264_nvenc":
        cmd += ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq",
                "-rc", "vbr", "-cq", str(args.cq), "-b:v", "0",
                "-maxrate", args.maxrate, "-bufsize", args.bufsize,
                "-profile:v", "high"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(args.cq),
                "-maxrate", args.maxrate, "-bufsize", args.bufsize,
                "-profile:v", "high"]
    cmd += ["-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k",
            "-movflags", "+faststart", dst]
    return cmd


def shrink_one(path, args):
    """Trả (trạng thái, byte_trước, byte_sau)."""
    info = probe(path)
    if info is None:
        return "loi-doc", 0, 0
    if info["mbps"] <= args.max_mbps and (info["codec"] or "") == "h264":
        return "da-nhe", info["size"], info["size"]
    if info["dur"] <= 0:
        return "hong", info["size"], info["size"]

    dst = os.path.splitext(path)[0] + ".shrink.mp4"
    if args.dry_run:
        est = info["dur"] * args.est_mbps * 1e6 / 8
        return "se-nen", info["size"], int(est)

    t0 = time.time()
    r = _run(build_cmd(path, dst, args))
    if r.returncode != 0 or not os.path.exists(dst):
        if os.path.exists(dst):
            os.remove(dst)
        return f"loi-encode: {r.stderr.strip()[:120]}", info["size"], 0

    # Kiểm tra bản mới trước khi dám thay: phải đọc được và ĐỦ ĐỘ DÀI.
    new = probe(dst)
    if new is None or abs(new["dur"] - info["dur"]) > args.tol:
        os.remove(dst)
        return (f"lech-do-dai ({new['dur'] if new else '?'} vs {info['dur']:.0f}s)",
                info["size"], 0)

    final = os.path.splitext(path)[0] + ".mp4"
    if args.keep_original:
        if os.path.abspath(final) == os.path.abspath(path):
            final = os.path.splitext(path)[0] + "_nhe.mp4"
        os.replace(dst, final)
    else:
        os.replace(dst, final)
        if os.path.abspath(final) != os.path.abspath(path):
            os.remove(path)
    args._sec += time.time() - t0
    args._dur += info["dur"]
    return "xong", info["size"], new["size"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="thư mục chứa kết quả của video_split (hoặc 1 file)")
    ap.add_argument("--max-mbps", type=float, default=2.5,
                    help="file H.264 nhẹ hơn mức này thì bỏ qua (mặc định 2.5)")
    ap.add_argument("--max-width", type=int, default=1280,
                    help="thu nhỏ nếu rộng hơn mức này (mặc định 1280 = 720p ngang)")
    ap.add_argument("--cq", type=int, default=30,
                    help="chất lượng: số nhỏ = nét hơn + nặng hơn (nvenc cq / x264 crf)")
    ap.add_argument("--maxrate", default="3M")
    ap.add_argument("--bufsize", default="6M")
    ap.add_argument("--encoder", default="h264_nvenc",
                    choices=["h264_nvenc", "libx264"],
                    help="nvenc = card NVIDIA (nhanh hơn nhiều); libx264 = CPU")
    ap.add_argument("--hw", action="store_true", help="thử decode bằng phần cứng")
    ap.add_argument("--tol", type=float, default=2.0,
                    help="độ dài bản mới lệch quá mức này (giây) thì bỏ, giữ file cũ")
    ap.add_argument("--include-source", action="store_true",
                    help="nén CẢ 00_goc_* (mặc định KHÔNG đụng bản gốc)")
    ap.add_argument("--keep-original", action="store_true",
                    help="giữ file cũ, ghi bản nhẹ ra tên khác")
    ap.add_argument("--est-mbps", type=float, default=1.5,
                    help="chỉ dùng cho --dry-run: ước lượng bitrate sau khi nén")
    ap.add_argument("--min-age", type=float, default=300,
                    help="bỏ qua file vừa ghi trong bấy nhiêu giây (tránh đụng clip "
                         "mà video_split đang cắt dở)")
    ap.add_argument("--limit", type=int, default=0, help="chỉ làm N file đầu (để thử)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args._sec = 0.0
    args._dur = 0.0

    root = os.path.abspath(args.root)
    # Dọn file tạm của lần chạy trước bị giết giữa chừng (encode dở, vô dụng).
    if os.path.isdir(root):
        n_tmp = 0
        for dp, _d, fs in os.walk(root):
            for f in fs:
                if f.endswith(".shrink.mp4"):
                    os.remove(os.path.join(dp, f))
                    n_tmp += 1
        if n_tmp:
            print(f"đã dọn {n_tmp} file tạm .shrink.mp4 của lần chạy trước\n")
    files = ([root] if os.path.isfile(root)
             else find_clips(root, args.include_source, args.min_age))
    if args.limit:
        files = files[:args.limit]
    if not files:
        sys.exit(f"không thấy clip nào trong {root}")
    print(f"{len(files)} file cần xem xét\n")

    tot_before = tot_after = 0
    n_done = n_skip = n_err = 0
    t0 = time.time()
    for i, p in enumerate(files, 1):
        st, before, after = shrink_one(p, args)
        tot_before += before
        tot_after += after if after else before
        tag = os.path.basename(p)[:52]
        if st in ("xong", "se-nen"):
            n_done += 1
            print(f"[{i}/{len(files)}] {tag:52s} {before / 1e6:8.1f} -> "
                  f"{after / 1e6:8.1f} MB  ({after / before:.0%})")
        elif st == "da-nhe":
            n_skip += 1
        else:
            n_err += 1
            print(f"[{i}/{len(files)}] {tag:52s} BỎ: {st}")
        if i % 25 == 0:
            print(f"  … {i}/{len(files)}, đã tiết kiệm "
                  f"{(tot_before - tot_after) / 1e9:.1f} GB", flush=True)

    print(f"\n{'DỰ TÍNH' if args.dry_run else 'XONG'}: nén {n_done} file, "
          f"bỏ qua {n_skip} (đã nhẹ), lỗi {n_err}")
    print(f"{tot_before / 1e9:.1f} GB -> {tot_after / 1e9:.1f} GB "
          f"(tiết kiệm {(tot_before - tot_after) / 1e9:.1f} GB, "
          f"còn {tot_after / max(tot_before, 1):.0%})")
    if args._sec:
        print(f"tốc độ encode: {args._dur / args._sec:.1f}x thời gian thực "
              f"({time.time() - t0:.0f}s cho lô này)")


if __name__ == "__main__":
    main()
