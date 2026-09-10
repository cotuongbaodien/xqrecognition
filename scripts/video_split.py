"""Cắt video cờ tướng dài thành từng ván + xuất chuỗi FEN theo thời gian.

Dùng chính detector đang chạy prod (items.pt + board_seg.pt) để đọc thế cờ trên
các frame lấy mẫu. Bàn vừa được xếp lại về **thế khai cuộc** = mốc bắt đầu ván
mới. Từ các mốc đó cắt video ra nhiều clip bằng ffmpeg (copy stream).

  # chạy đủ: quét -> tìm mốc -> cắt (mặc định lấy dư 5 phút mỗi đầu)
  python scripts/video_split.py "E:\\videos\\GiangHo\\ten video.mp4"

  # chỉ quét + tìm mốc, CHƯA cắt (soi starts.jpg trước)
  python scripts/video_split.py <video> --stage scan --stage segment

  # sửa ngưỡng rồi tính lại mốc — KHÔNG chạy lại model, tốn ~1 giây
  python scripts/video_split.py <video> --stage segment --start-dist 3 --min-gap 90

  # duyệt ảnh xong thì cắt
  python scripts/video_split.py <video> --stage cut

Kết quả nằm NGAY CẠNH video gốc, mỗi video một thư mục tự chứa đủ:

  E:\\videos\\GiangHo\\<tên video>\\
      00_goc_<tên video>.mp4        <- video gốc, ĐƯỢC DỜI VÀO (--no-move để tắt)
      <tên video>_van01_00-00-09.mp4
      <tên video>_van02_00-11-23.mp4  …
      starts.jpg                    <- ảnh mốc từng ván, soi trước khi tin
      index.csv                     <- mốc, độ dài, dung lượng từng clip
      _data/                        <- cache + timeline.csv + games.json + fens/

Ba stage tách rời, chung cache `_data/scan.json` nên chạy lại rất rẻ:

  scan     decode + detect  (phần đắt duy nhất; có cache, --repredict để ép lại)
  segment  logic thuần      -> games.json, timeline.csv, starts.jpg, fens/*.jsonl
  cut      ffmpeg -c copy   -> các clip ván + index.csv, rồi dời video gốc vào

Cách nhận mốc ván (số liệu đo trên video thật, xem docs/VIDEO_SPLIT.md):
  * `d` = số ô lệch so với thế khai cuộc. Tàn cuộc ~30-40, khai cuộc = 0.
  * Mẫu khai cuộc = `d <= --start-dist` VÀ `>= --start-pieces` quân.
  * Các mẫu khai cuộc cách nhau <= --cluster-gap giây gộp thành MỘT cụm (pha xếp
    bàn tay che làm d nhảy loạn), mốc ván = mẫu CUỐI của cụm.
  * Phải có dấu hiệu ván trước kết thúc trong --confirm-window giây trước đó.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
for _p in (ROOT, SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from boarddetection.pipeline import XiangqiRecognizer  # noqa: E402
from _fenutil import cell_diff, dist_to_start, dist_to_start_farside  # noqa: E402


# --------------------------------------------------------------------------- #
# tiện ích chung
# --------------------------------------------------------------------------- #
_REC = None


def recognizer():
    """Nạp model MỘT lần cho cả tiến trình (nạp lại tốn vài giây mỗi lần)."""
    global _REC
    if _REC is None:
        _REC = XiangqiRecognizer()
    return _REC


def short(path):
    """Đường dẫn ngắn để in ra. KHÔNG dùng os.path.relpath: kết quả nằm ở ổ E:
    còn repo ở ổ C:, relpath khác ổ đĩa là ném ValueError và giết cả lô."""
    return os.path.join(os.path.basename(os.path.dirname(path)),
                        os.path.basename(path))


def work_dir(out):
    """Thư mục phụ `_data/` — cache, csv, chuỗi FEN, frame tạm.

    Thư mục gốc chỉ để lại thứ người dùng thực sự mở: video gốc, các clip ván,
    `starts.jpg` và `index.csv`.
    """
    d = os.path.join(out, "_data")
    os.makedirs(d, exist_ok=True)
    return d


def imread_u(path):
    """cv2.imread không đọc được đường dẫn có dấu tiếng Việt trên Windows."""
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def imwrite_u(path, img):
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


def hhmmss(t):
    t = int(round(t))
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def _run(cmd, timeout=None):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def video_info(video):
    """Độ dài + kích thước + fps. Thiếu ffprobe là hỏng hẳn nên báo lỗi luôn."""
    r = _run(["ffprobe", "-v", "error", "-select_streams", "v:0",
              "-show_entries", "stream=width,height,r_frame_rate",
              "-show_entries", "format=duration", "-of", "json", video])
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe lỗi: {r.stderr.strip()[:200]}")
    j = json.loads(r.stdout)
    st = (j.get("streams") or [{}])[0]
    num, _, den = (st.get("r_frame_rate") or "0/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    return {
        "duration": float(j.get("format", {}).get("duration", 0.0)),
        "width": st.get("width"), "height": st.get("height"), "fps": round(fps, 3),
    }


def keyframe_before(video, t):
    """Keyframe gần nhất TRƯỚC mốc t (giây) — chính là chỗ `-c copy` sẽ bắt đầu."""
    lo = max(0.0, t - 15.0)
    r = _run(["ffprobe", "-v", "error", "-select_streams", "v:0",
              "-read_intervals", f"{lo:.2f}%+20",
              "-show_entries", "packet=pts_time,flags", "-of", "csv=p=0", video])
    best = None
    for line in r.stdout.splitlines():
        parts = line.split(",")
        if len(parts) < 2 or "K" not in parts[1]:
            continue
        try:
            pts = float(parts[0])
        except ValueError:
            continue
        if pts <= t + 0.001 and (best is None or pts > best):
            best = pts
    return best


def keyframe_gap(video, at=600.0, window=60.0):
    """Khoảng cách trung vị giữa các keyframe (giây), đo trên một cửa sổ ngắn.

    Dùng để quyết định có dám decode CHỈ keyframe hay không: keyframe dày hơn nhiều
    so với bước lấy mẫu thì bỏ qua khung hình giữa không mất gì, mà nhanh gấp đôi.
    """
    r = _run(["ffprobe", "-v", "error", "-select_streams", "v:0",
              "-read_intervals", f"{at:.0f}%+{window:.0f}",
              "-show_entries", "packet=pts_time,flags", "-of", "csv=p=0", video])
    ks = []
    for line in r.stdout.splitlines():
        parts = line.split(",")
        if len(parts) >= 2 and "K" in parts[1]:
            try:
                ks.append(float(parts[0]))
            except ValueError:
                pass
    if len(ks) < 3:
        return None
    gaps = [b - a for a, b in zip(ks, ks[1:]) if b > a]
    return float(np.median(gaps)) if gaps else None


def decode(video, out_dir, step, start=0.0, dur=None, width=1280, tag="f",
           keyframe_only=False):
    """Trích frame bằng MỘT tiến trình ffmpeg (decode tuần tự).

    Đừng thay bằng `-ss` từng mốc: seek một lần ~0,5-1 s, còn decode cả video 1h
    ở fps=1/20 chỉ mất ~28 s (đo thật).

    Trả list (timestamp_giây, đường_dẫn_jpg).
    """
    # Mỗi lượt decode một thư mục riêng: nếu dùng chung thư mục thì frame của lượt
    # trước còn nằm đó và chỉ số -> timestamp bị lệch (bug đã dính một lần).
    out_dir = os.path.join(out_dir, tag)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    pat = os.path.join(out_dir, "%06d.jpg")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if keyframe_only:
        # Bỏ hẳn việc dựng lại khung hình giữa hai keyframe. Đo trên HEVC iPhone:
        # 10 phút nội dung 23,8 s -> 11,2 s (nhanh 2,1x). Chỉ bật khi keyframe dày
        # hơn bước lấy mẫu nhiều lần, nếu không lưới mẫu sẽ thưa và lệch.
        cmd += ["-skip_frame", "nokey"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", video]
    if dur:
        cmd += ["-t", f"{dur:.3f}"]
    fps = f"1/{step}" if step >= 1 else f"{1 / step:g}"
    # -pix_fmt yuvj420p: video iPhone hay gắn cờ color-range lạ, encoder mjpeg của
    # ffmpeg 8 từ chối mở ("Non full-range YUV is non-standard") và cả video chết.
    cmd += ["-vf", f"fps={fps},scale={width}:-2", "-pix_fmt", "yuvj420p",
            "-q:v", "3", "-y", pat]
    r = _run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg decode lỗi: {r.stderr.strip()[:300]}")
    names = sorted(n for n in os.listdir(out_dir) if n.endswith(".jpg"))
    return [(round(start + i * step, 2), os.path.join(out_dir, n))
            for i, n in enumerate(names)]


# --------------------------------------------------------------------------- #
# stage 1: scan (decode + detect)
# --------------------------------------------------------------------------- #
def board_roi(rec, paths, pad=0.10, sample=20):
    """ROI bàn cờ = MEDIAN bbox của quad segmentation trên nhiều frame.

    Median chứ không phải một frame: đo trên video dọc 720x1280 mép trên/dưới rất
    ổn nhưng mép trái nhảy 36->248 px vì mask seg lem. Trả (roi, tỉ_lệ_chiều_cao)
    hoặc (None, 0) khi không thấy bàn.
    """
    if not paths:
        return None, 0.0
    idx = np.linspace(0, len(paths) - 1, min(sample, len(paths))).astype(int)
    boxes, fracs = [], []
    h_img = w_img = None
    for i in idx:
        img = imread_u(paths[i])
        if img is None:
            continue
        h_img, w_img = img.shape[:2]
        seg = rec.board_segmenter.detect(img)
        if seg is None:
            continue
        xs = [p[0] for p in seg.quad]
        ys = [p[1] for p in seg.quad]
        boxes.append([min(xs), min(ys), max(xs), max(ys)])
        fracs.append((max(ys) - min(ys)) / h_img)
    if len(boxes) < 3 or h_img is None:
        return None, 0.0
    b = np.median(np.array(boxes), axis=0)
    bw, bh = b[2] - b[0], b[3] - b[1]
    roi = (
        int(max(0, b[0] - pad * bw)), int(max(0, b[1] - pad * bh)),
        int(min(w_img, b[2] + pad * bw)), int(min(h_img, b[3] + pad * bh)),
    )
    return roi, float(np.median(fracs))


def detect_frames(rec, frames, conf, roi=None, label=""):
    """Chạy detector trên list (t, path). Trả list bản ghi cho scan.json."""
    out = []
    t0 = time.time()
    for i, (t, path) in enumerate(frames, 1):
        img = imread_u(path)
        if img is None:
            continue
        if roi:
            x1, y1, x2, y2 = roi
            crop = img[y1:y2, x1:x2]
            if crop.size:
                img = crop
        r = rec.recognize_image_2pass(img, piece_confidence=conf)
        fen = (r.fen or "").split()[0] if r.fen else ""
        out.append({
            "t": t, "fen": fen, "n": len(r.pieces),
            "conf": round(float(r.confidence), 3),
            "d": dist_to_start(fen),
            "d_far": dist_to_start_farside(fen),
            "gate": bool(XiangqiRecognizer.passes_gate(r)),
            "rot180": bool(r.used_rot180),
        })
        if i % 100 == 0:
            print(f"  detect{label} {i}/{len(frames)}  {time.time() - t0:.0f}s",
                  flush=True)
    return out


def merge_frames(cache, new):
    """Gộp bản ghi mới vào cache, khử trùng theo timestamp (làm tròn 0,1 s)."""
    by_t = {round(f["t"], 1): f for f in cache.get("frames", [])}
    for f in new:
        by_t[round(f["t"], 1)] = f
    cache["frames"] = [by_t[k] for k in sorted(by_t)]
    return cache


def stage_scan(video, out, args):
    """Quét thô cả video + quét tinh quanh từng ứng viên mốc ván."""
    cache_path = os.path.join(work_dir(out), "scan.json")
    info = video_info(video)
    print(f"video: {info['duration'] / 60:.1f} phút, {info['width']}x{info['height']}, "
          f"{info['fps']} fps")

    cache = None
    if os.path.exists(cache_path) and not args.repredict:
        cache = json.load(open(cache_path, encoding="utf-8"))
        same = (cache.get("params", {}).get("step") == args.step
                and cache.get("params", {}).get("conf") == args.conf)
        if same:
            print(f"dùng lại {short(cache_path)} "
                  f"({len(cache['frames'])} frame) — --repredict để chạy lại")
        else:
            print("tham số đổi so với cache -> quét lại")
            cache = None
    if cache is None:
        cache = {"video": os.path.abspath(video), **info,
                 "params": {"step": args.step, "conf": args.conf,
                            "fine_step": args.fine_step, "roi": None},
                 "frames": []}

    rec = recognizer()
    frames_dir = os.path.join(work_dir(out), "frames")

    # --- quét thô, CHIA TỪNG KHÚC ---
    # Chia khúc để: (a) video 9 tiếng chết giữa chừng thì chạy lại không mất phần đã
    # làm — cache ghi sau MỖI khúc; (b) thấy tiến độ thay vì im lặng 20 phút.
    dur_total = info["duration"]
    chunk = max(60.0, args.chunk * 60.0)
    chunks = [(a, min(chunk, dur_total - a))
              for a in np.arange(0.0, dur_total, chunk)]
    done_chunks = {round(c, 1) for c in cache.get("chunks_done", [])}
    todo = [(a, d) for a, d in chunks if round(a, 1) not in done_chunks and d > 1]

    if todo:
        kf = keyframe_gap(video, at=min(600.0, dur_total / 3))
        # Chỉ dám bỏ khung hình giữa khi keyframe dày hơn hẳn bước lấy mẫu, nếu không
        # lưới mẫu bị thưa và lệch so với mốc mong muốn.
        # Ngưỡng PHẢI chặt. Thử nới lên step/2 (keyframe 6s, bước 20s) thì lưới mẫu
        # thô xê dịch vài giây và **mất hẳn một ván** trên video mẫu (6 -> 5 ván):
        # cửa sổ "bàn còn giống khai cuộc" chỉ kéo dài vài chục giây, lệch một chút
        # là không frame nào rơi vào. Chỉ bật khi keyframe dày hơn bước lấy mẫu ÍT
        # NHẤT 8 lần (video iPhone: 0,93s so với bước 20s) -> xê dịch dưới 1 giây.
        kf_max = args.kf_max if args.kf_max > 0 else args.step / 8
        key_only = kf is not None and kf <= kf_max
        print(f"keyframe mỗi ~{kf:.1f}s -> "
              f"{'CHỈ decode keyframe (nhanh ~2x)' if key_only else 'decode đầy đủ'}"
              if kf else "không đo được keyframe -> decode đầy đủ")
        cache["params"]["keyframe_only"] = bool(key_only)

        roi = tuple(cache["params"]["roi"]) if cache["params"].get("roi") else None
        for ci, (a, d) in enumerate(todo, 1):
            t0 = time.time()
            fr = decode(video, frames_dir, args.step, start=a, dur=d,
                        width=args.width, tag=f"c{int(a):07d}", keyframe_only=key_only)
            # ROI chốt một lần từ khúc đầu tiên có bàn cờ.
            if not args.no_roi and cache["params"].get("roi_done") is not True:
                roi, frac = board_roi(rec, [p for _, p in fr], sample=args.roi_sample)
                if roi and frac >= args.roi_min_frac:
                    print(f"bàn chiếm {frac:.0%} chiều cao -> KHÔNG crop (đủ to)")
                    roi = None
                elif roi:
                    print(f"bàn chỉ chiếm {frac:.0%} chiều cao -> crop ROI {roi}")
                else:
                    print("không chốt được ROI (seg trả None) -> dùng full-frame")
                cache["params"]["roi"] = list(roi) if roi else None
                cache["params"]["board_frac"] = round(frac, 3)
                cache["params"]["roi_done"] = True
            merge_frames(cache, detect_frames(rec, fr, args.conf, roi, ""))
            done_chunks.add(round(a, 1))
            cache["chunks_done"] = sorted(done_chunks)
            json.dump(cache, open(cache_path, "w", encoding="utf-8"), indent=0)
            shutil.rmtree(os.path.join(frames_dir, f"c{int(a):07d}"),
                          ignore_errors=True)
            print(f"  khúc {ci}/{len(todo)} ({hhmmss(a)}-{hhmmss(a + d)}): "
                  f"{len(fr)} frame / {time.time() - t0:.0f}s", flush=True)

    roi = tuple(cache["params"]["roi"]) if cache["params"].get("roi") else None

    # --- quét tinh quanh ứng viên ---
    done = {round(w, 1) for w in cache.get("fine_done", [])}
    cands = [f["t"] for f in cache["frames"]
             if f["n"] >= args.cand_pieces and f["d"] <= args.cand_dist]
    windows = []
    for t in cands:
        lo = max(0.0, t - args.fine_before)
        if any(abs(lo - w) < 1.0 for w in done):
            continue
        if windows and lo <= windows[-1][1]:          # gộp cửa sổ chồng nhau
            windows[-1] = (windows[-1][0], lo + args.fine_before + args.fine_after)
        else:
            windows.append((lo, lo + args.fine_before + args.fine_after))
    if windows:
        print(f"{len(cands)} ứng viên -> {len(windows)} cửa sổ quét tinh "
              f"(fps=1/{args.fine_step})")
    for k, (lo, hi) in enumerate(windows, 1):
        hi = min(hi, info["duration"])
        fine = decode(video, frames_dir, args.fine_step, start=lo, dur=hi - lo,
                      width=args.width, tag=f"w{k:02d}")
        merge_frames(cache, detect_frames(rec, fine, args.conf, roi,
                                          f" tinh {k}/{len(windows)}"))
        done.add(round(lo, 1))
    cache["fine_done"] = sorted(done)
    json.dump(cache, open(cache_path, "w", encoding="utf-8"), indent=0)
    print(f"scan xong: {len(cache['frames'])} mẫu -> "
          f"{short(cache_path)}")
    return cache


def scan_spans(video, out, cache, spans, step, conf, label="dày"):
    """Quét bổ sung các khoảng [a,b] ở bước `step` (dùng cho chuỗi FEN)."""
    if step <= 0 or not spans:
        return cache
    rec = recognizer()
    roi = tuple(cache["params"]["roi"]) if cache["params"].get("roi") else None
    key_only = bool(cache["params"].get("keyframe_only"))
    frames_dir = os.path.join(work_dir(out), "frames")
    for k, (a, b) in enumerate(spans, 1):
        have = [f["t"] for f in cache["frames"] if a <= f["t"] <= b]
        if len(have) >= (b - a) / step * 0.9:      # đã có sẵn mẫu đủ dày
            continue
        fr = decode(video, frames_dir, step, start=a, dur=b - a, tag=f"d{k:02d}",
                    keyframe_only=key_only and step >= 4)
        merge_frames(cache, detect_frames(rec, fr, conf, roi,
                                          f" {label} {k}/{len(spans)}"))
    json.dump(cache, open(os.path.join(work_dir(out), "scan.json"), "w", encoding="utf-8"),
              indent=0)
    return cache


# --------------------------------------------------------------------------- #
# stage 2: segment (mốc ván) — logic thuần, không nạp model
# --------------------------------------------------------------------------- #
def find_starts(frames, args):
    """Mốc bắt đầu từng ván, theo luật cụm mô tả trong docstring."""
    frames = sorted(frames, key=lambda f: f["t"])
    hits = [f for f in frames
            if f["n"] >= args.start_pieces and f["d"] <= args.start_dist]
    if not hits:
        return [], []          # phải trả ĐỦ 2 giá trị, video ngắn hay bị rơi vào đây

    clusters = [[hits[0]]]
    for f in hits[1:]:
        if f["t"] - clusters[-1][-1]["t"] <= args.cluster_gap:
            clusters[-1].append(f)
        else:
            clusters.append([f])

    starts, rejects = [], []
    for cl in clusters:
        t = cl[-1]["t"]                       # mẫu CUỐI: sau đó bàn chỉ đi tiếp
        item = {"t": t, "hms": hhmmss(t), "d": cl[-1]["d"], "n": cl[-1]["n"],
                "cluster": len(cl), "cluster_t0": cl[0]["t"]}
        if t > args.confirm_window:
            before = [f for f in frames
                      if t - args.confirm_window <= f["t"] < cl[0]["t"]]
            ended = sum(1 for f in before if f["d"] >= args.confirm_dist
                        or f["n"] <= args.confirm_pieces)
            if ended < args.confirm_min:
                # Không thấy ván trước tàn -> nhiều khả năng là thế giữa ván tình
                # cờ giống khai cuộc, hoặc bàn phân tích 2D lặp lại thế khai cuộc.
                rejects.append({**item, "ly_do": "khong-thay-van-truoc-tan "
                                                 f"({ended}/{args.confirm_min})"})
                continue
        if starts and t - starts[-1]["t"] < args.min_gap:
            rejects.append({**item, "ly_do": "qua-gan-moc-truoc "
                                             f"({t - starts[-1]['t']:.0f}s < "
                                             f"{args.min_gap:.0f}s)"})
            continue
        starts.append(item)
    return starts, rejects


def build_games(starts, duration, args):
    games = []
    ts = [s["t"] for s in starts]
    if not ts or ts[0] > args.min_gap:
        # Video mở đầu bằng một ván đang đá dở -> vẫn tính là ván 1.
        games.append({"i": 1, "start": 0.0, "start_kind": "dau-video",
                      "end": (ts[0] - args.reset_lead) if ts else duration})
    for k, s in enumerate(starts):
        end = (ts[k + 1] - args.reset_lead) if k + 1 < len(ts) else duration
        games.append({"i": len(games) + 1, "start": s["t"], "start_kind": "khai-cuoc",
                      "end": max(end, s["t"] + 1), "d": s["d"], "n": s["n"],
                      "cluster": s["cluster"]})
    for g in games:
        g["end"] = min(g["end"], duration)
        g["dur"] = round(g["end"] - g["start"], 1)
        g["start_hms"], g["end_hms"] = hhmmss(g["start"]), hhmmss(g["end"])
    return games


def fen_series(frames, a, b, min_seen):
    """Chuỗi quan sát FEN trong [a,b]: chỉ ghi lúc ĐỔI thế, có lọc nhiễu.

    KHÔNG phải biên bản nước đi — xem docs/VIDEO_SPLIT.md §Giới hạn.
    """
    sel = [f for f in sorted(frames, key=lambda f: f["t"])
           if a <= f["t"] <= b and f["gate"] and f["fen"]]
    out, run = [], []
    for f in sel + [None]:
        if run and (f is None or f["fen"] != run[-1]["fen"]):
            if len(run) >= min_seen:
                best = max(run, key=lambda x: x["conf"])
                row = {"t": run[0]["t"], "t_last": run[-1]["t"], "fen": run[0]["fen"],
                       "n": best["n"], "conf": best["conf"], "seen": len(run)}
                # d_prev = số ô đổi so với thế ĐÃ GHI trước. Một nước cờ hợp lệ đổi
                # ĐÚNG 2 ô (ô đi khỏi thành trống, ô đến đổi quân) — ăn quân cũng vậy.
                # Tỉ lệ d_prev==2 chính là thước đo chuỗi này sạch tới đâu.
                row["d_prev"] = cell_diff(out[-1]["fen"], row["fen"]) if out else None
                out.append(row)
            run = []
        if f is not None:
            run.append(f)
    stat = {"n_obs": len(out),
            "mot_nuoc": sum(1 for r in out if r["d_prev"] == 2),
            "hai_nuoc": sum(1 for r in out if r["d_prev"] == 4),
            "nhieu_hon": sum(1 for r in out if r["d_prev"] not in (None, 2, 4))}
    return out, stat


def contact_sheet(video, games, out_path, width=480, cols=3):
    """Ảnh dán frame lúc bắt đầu mỗi ván để soi mắt trước khi cắt."""
    tiles = []
    tmp = out_path + ".tmp.jpg"
    for g in games:
        r = _run(["ffmpeg", "-hide_banner", "-loglevel", "error",
                  "-ss", f"{g['start']:.2f}", "-i", video, "-frames:v", "1",
                  "-vf", f"scale={width}:-2", "-y", tmp])
        img = imread_u(tmp) if r.returncode == 0 and os.path.exists(tmp) else None
        if img is None:
            img = np.full((width * 9 // 16, width, 3), 60, np.uint8)
        cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(img, f"Van {g['i']}  {g['start_hms']}  ({g['dur'] / 60:.1f}p)",
                    (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        tiles.append(img)
    if os.path.exists(tmp):
        os.remove(tmp)
    if not tiles:
        return None
    h = min(t.shape[0] for t in tiles)
    tiles = [cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h)) for t in tiles]
    w = min(t.shape[1] for t in tiles)
    tiles = [t[:, :w] for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    imwrite_u(out_path, np.vstack(rows))
    return out_path


def stage_segment(video, out, cache, args):
    frames = cache["frames"]
    starts, rejects = find_starts(frames, args)
    games = build_games(starts, cache["duration"], args)

    # Không thấy mốc ván NÀO và hầu như chẳng frame nào có bàn cờ -> video này
    # không phải video cờ (hoặc bàn không đọc được). Đừng đẻ ra một bản copy cả
    # video gắn mác "ván 1".
    gate_frac = (sum(1 for f in frames if f["gate"]) / len(frames)) if frames else 0.0
    if not starts and gate_frac < args.min_board_frac:
        print(f"KHÔNG thấy ván nào và chỉ {gate_frac:.0%} frame có bàn cờ "
              f"-> bỏ qua video này (không cắt)")
        games = []

    # "Một ván cờ tối đa ~25 phút": khoảng trống dài hơn thế gần như chắc chắn là
    # BỎ SÓT một mốc chứ không phải một ván dài thật. Quét riêng khoảng đó dày hơn
    # rồi tính lại — rẻ hơn nhiều so với quét dày cả video.
    if args.max_game > 0 and args.gap_step > 0:
        bounds = [0.0] + [s["t"] for s in starts] + [cache["duration"]]
        gaps = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
                if bounds[i + 1] - bounds[i] > args.max_game * 60]
        if gaps:
            tot = sum(b - a for a, b in gaps) / 60
            print(f"{len(gaps)} khoảng dài hơn {args.max_game:.0f} phút "
                  f"(tổng {tot:.0f} phút) -> quét lại dày {args.gap_step:.0f}s/frame "
                  f"để tìm mốc bỏ sót", flush=True)
            cache = scan_spans(video, out, cache, gaps, args.gap_step, args.conf,
                               "kẽ hở")
            frames = cache["frames"]
            n_before = len(starts)
            starts, rejects = find_starts(frames, args)
            print(f"  -> {len(starts) - n_before:+d} mốc so với trước")
            games = build_games(starts, cache["duration"], args)

    # Bàn đọc ngược đầu (đổi màu đen<->đỏ) lệch ĐÚNG 32 ô so với thế khai cuộc —
    # rơi trọn vào dải 30-40 của "không phải khai cuộc" nên sẽ im lặng trôi qua.
    # Soi riêng để biến ca đó thành cảnh báo nhìn thấy được.
    far = [f for f in frames if f.get("d_far", 99) <= 2 and f["n"] >= 30]
    if far:
        print(f"CANH BAO: {len(far)} mẫu giống thế khai cuộc NHƯNG ĐỔI MÀU "
              f"(đầu tiên lúc {hhmmss(far[0]['t'])}) — bàn đang bị đọc ngược đầu, "
              f"chuỗi FEN đoạn đó sẽ sai màu.")

    # Chuỗi FEN cần mẫu dày hơn quét thô -> quét bổ sung trong từng ván.
    if args.fen_step > 0:
        spans = [(g["start"], g["end"]) for g in games]
        cache = scan_spans(video, out, cache, spans, args.fen_step, args.conf)
        frames = cache["frames"]

    fen_dir = os.path.join(work_dir(out), "fens")
    os.makedirs(fen_dir, exist_ok=True)
    min_seen = args.min_seen if args.fen_step and args.fen_step <= 5 else 1
    for g in games:
        series, stat = fen_series(frames, g["start"], g["end"], min_seen)
        g["n_obs"] = stat["n_obs"]
        g["fen_stat"] = stat
        path = os.path.join(fen_dir, f"g{g['i']:02d}_fens.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"_meta": {
                "game": g["i"], "start": g["start"], "end": g["end"],
                "fen_step": args.fen_step, "min_seen": min_seen,
                "thong_ke": stat,
                "note": "quan sat tho tu detector, KHONG phai bien ban nuoc di"
            }}, ensure_ascii=False) + "\n")
            for s in series:
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    json.dump(games, open(os.path.join(work_dir(out), "games.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    with open(os.path.join(work_dir(out), "timeline.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "hms", "n", "d", "gate", "rot180", "conf", "fen"])
        for f in sorted(frames, key=lambda f: f["t"]):
            w.writerow([f["t"], hhmmss(f["t"]), f["n"], f["d"], int(f["gate"]),
                        int(f.get("rot180", False)), f["conf"], f["fen"]])
    sheet = contact_sheet(video, games, os.path.join(out, "starts.jpg"))

    json.dump(rejects, open(os.path.join(work_dir(out), "rejected.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"\n=== {len(games)} ván ===")
    for g in games:
        st = g.get("fen_stat", {})
        note = (f"{st['n_obs']} thế, {st['mot_nuoc']} hợp một nước"
                if st.get("n_obs") else "chưa dựng chuỗi FEN")
        print(f"  ván {g['i']:2d}  {g['start_hms']} -> {g['end_hms']}  "
              f"({g['dur'] / 60:5.1f} phút)  {g['start_kind']}  {note}")
    if rejects:
        print(f"\n{len(rejects)} ứng viên bị loại (xem rejected.json):")
        for r in rejects:
            print(f"  {r['hms']}  d={r['d']:2d} n={r['n']:2d}  {r['ly_do']}")
    print(f"\nsoi ảnh trước khi cắt: {sheet}")
    return games


# --------------------------------------------------------------------------- #
# stage 3: cut
# --------------------------------------------------------------------------- #
def clip_name(stem, i, start_hms, ext):
    """`<tên video>_van01_00-19-17.mp4` — mang theo tên video để clip tách khỏi
    thư mục vẫn biết là của video nào. Cắt bớt tên dài cho khỏi vượt giới hạn
    260 ký tự của Windows."""
    return f"{stem[:60]}_van{i:02d}_{start_hms.replace(':', '-')}{ext}"


def stage_cut(video, out, games, duration, args):
    """Cắt bằng `-c copy`. Mỗi clip lấy dư --lead trước và --tail sau.

    Clip nằm THẲNG trong thư mục của video (cạnh chính file gốc), không nhét vào
    thư mục con — mở thư mục ra là thấy ngay các ván.
    """
    clips = out
    stem = os.path.splitext(os.path.basename(video))[0]
    if stem.startswith("00_goc_"):          # video gốc đã được dời vào từ lần trước
        stem = stem[len("00_goc_"):]
    rows = []
    for g in games:
        a = max(0.0, g["start"] - args.lead)
        b = min(duration, g["end"] + args.tail)
        ext = os.path.splitext(video)[1] or ".mp4"
        name = clip_name(stem, g["i"], g["start_hms"], ext)
        path = os.path.join(clips, name)
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
               # -ss TRƯỚC -i: seek theo index, không decode; với -c copy ffmpeg
               # lùi về keyframe gần nhất nên chỉ dư ở đầu, không bao giờ cụt.
               "-ss", f"{a:.3f}", "-i", video, "-t", f"{b - a:.3f}",
               # -map 0 kéo theo cả stream dữ liệu mp4 không copy được -> chỉ lấy
               # video + audio (dấu ? = không có audio cũng không sao).
               "-map", "0:v:0", "-map", "0:a?", "-map_metadata", "0",
               "-c", "copy", "-avoid_negative_ts", "make_zero",
               "-movflags", "+faststart", "-y", path]
        if args.dry_run:
            print("  " + subprocess.list2cmdline(cmd))
            continue
        if os.path.exists(path) and not args.overwrite:
            print(f"  ván {g['i']:2d}  đã có {name} — bỏ qua (--overwrite để ghi đè)")
            continue
        t0 = time.time()
        r = _run(cmd)
        if r.returncode != 0:
            print(f"  ván {g['i']}: ffmpeg LỖI — {r.stderr.strip()[:200]}")
            continue
        snap = keyframe_before(video, a)
        size = os.path.getsize(path) / 1e6
        print(f"  ván {g['i']:2d}  {hhmmss(a)} -> {hhmmss(b)}  "
              f"({(b - a) / 60:5.1f} phút, {size:6.1f} MB, {time.time() - t0:.1f}s)"
              f"  snap={hhmmss(snap) if snap is not None else '?'}  {name}")
        rows.append({"van": g["i"], "file": name,
                     "mo_dau_yeu_cau": round(a, 2),
                     "mo_dau_thuc_te": round(snap, 2) if snap is not None else "",
                     "ket_thuc": round(b, 2), "dai_giay": round(b - a, 1),
                     "van_bat_dau": round(g["start"], 2),
                     "van_ket_thuc": round(g["end"], 2), "mb": round(size, 1)})
    # LUÔN ghi index.csv, kể cả không có clip nào — nó là dấu "video này đã xử lý"
    # để lần chạy lô sau bỏ qua, khỏi quét lại từ đầu.
    cols = ["van", "file", "mo_dau_yeu_cau", "mo_dau_thuc_te", "ket_thuc",
            "dai_giay", "van_bat_dau", "van_ket_thuc", "mb"]
    with open(os.path.join(out, "index.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} clip -> {clips}")
    return rows


def move_source(video, out):
    """Dời video gốc vào chính thư mục của nó, đổi tên `00_goc_<tên>.<ext>`.

    Tiền tố `00_goc_` để (a) phân biệt hẳn với các clip ván, (b) luôn nằm đầu khi
    sắp xếp theo tên. Cùng ổ đĩa thì đây chỉ là đổi tên -> tức thì; khác ổ thì phải
    copy cả GB nên báo trước.
    """
    stem = os.path.splitext(os.path.basename(video))[0]
    ext = os.path.splitext(video)[1]
    if stem.startswith("00_goc_"):
        return video                                   # đã dời từ lần chạy trước
    dest = os.path.join(out, f"00_goc_{stem}{ext}")
    if os.path.abspath(video) == os.path.abspath(dest):
        return video
    if os.path.exists(dest):
        print(f"đã có {os.path.basename(dest)} — không dời nữa")
        return dest
    same_drive = os.path.splitdrive(os.path.abspath(video))[0].lower() == \
        os.path.splitdrive(os.path.abspath(dest))[0].lower()
    size_gb = os.path.getsize(video) / 1e9
    if not same_drive:
        print(f"video gốc khác ổ đĩa với thư mục ra -> phải COPY {size_gb:.1f} GB…")
    shutil.move(video, dest)
    print(f"đã dời video gốc vào -> {os.path.basename(dest)}")
    return dest


# --------------------------------------------------------------------------- #
def run_one(video, out, args):
    """Chạy đủ các stage cho MỘT video. Ném exception nếu hỏng (để chạy lô bắt được)."""
    if not os.path.exists(video):
        # Chạy lại lần hai: video gốc đã được dời vào trong thư mục rồi.
        moved = os.path.join(out, f"00_goc_{os.path.basename(video)}")
        if not os.path.exists(moved):
            raise FileNotFoundError(f"không thấy video: {video}")
        video = moved
        print(f"video gốc đã nằm trong thư mục kết quả -> dùng {moved}")
    os.makedirs(out, exist_ok=True)
    stages = args.stage or ["scan", "segment", "cut"]
    print(f"video : {video}\nout   : {out}\nstage : {', '.join(stages)}\n")

    cache_path = os.path.join(work_dir(out), "scan.json")
    cache = None
    if "scan" in stages:
        cache = stage_scan(video, out, args)
    elif os.path.exists(cache_path):
        cache = json.load(open(cache_path, encoding="utf-8"))
    elif stages != ["cut"]:
        raise RuntimeError(f"chưa có {cache_path} — chạy --stage scan trước")

    games = None
    games_path = os.path.join(work_dir(out), "games.json")
    if "segment" in stages:
        games = stage_segment(video, out, cache, args)
    elif os.path.exists(games_path):
        games = json.load(open(games_path, encoding="utf-8"))

    if "cut" in stages:
        if games is None:
            raise RuntimeError("chưa có games.json — chạy --stage segment trước")
        dur = (cache or {}).get("duration") or video_info(video)["duration"]
        print(f"\n=== cắt (lấy dư {args.lead:.0f}s trước / {args.tail:.0f}s sau) ===")
        rows = stage_cut(video, out, games, dur, args)
        if rows and not args.no_move and not args.dry_run:
            move_source(video, out)

    frames_dir = os.path.join(work_dir(out), "frames")
    if not args.keep_frames and os.path.isdir(frames_dir) and "cut" in stages:
        shutil.rmtree(frames_dir, ignore_errors=True)
        print("đã dọn _data/frames (--keep-frames để giữ)")


VIDEO_EXT = (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".ts", ".flv", ".webm")


def list_videos(folder):
    """Video ở TẦNG ĐẦU của thư mục, sắp xếp NGẮN TRƯỚC DÀI SAU.

    Ngắn trước vì: xong sớm là có cái để soi, và nếu ngưỡng sai thì phát hiện sau vài
    phút chứ không phải sau vài giờ. Không đệ quy xuống thư mục con — video đã xử lý
    nằm trong thư mục con của chính nó, đệ quy vào là cắt lại lần nữa.
    """
    files = [os.path.join(folder, n) for n in sorted(os.listdir(folder))
             if n.lower().endswith(VIDEO_EXT)
             and os.path.isfile(os.path.join(folder, n))]
    out = []
    print(f"đọc độ dài {len(files)} video…", flush=True)
    for i, p in enumerate(files, 1):
        try:
            info = video_info(p)
            out.append((info["duration"], p, info))
        except Exception as e:                       # video hỏng thì bỏ qua, đừng chết
            print(f"  bỏ qua {os.path.basename(p)}: {e}")
        if i % 50 == 0:
            print(f"  {i}/{len(files)}", flush=True)
    out.sort(key=lambda x: x[0])
    return out


def run_folder(folder, args):
    """Chạy lần lượt mọi video trong thư mục, ngắn trước dài sau."""
    items = list_videos(folder)
    if not items:
        sys.exit(f"không có video nào ở tầng đầu của {folder}")
    total_h = sum(d for d, _, _ in items) / 3600
    total_gb = sum(os.path.getsize(p) for _, p, _ in items) / 1e9
    print(f"\n{len(items)} video, tổng {total_h:.1f} giờ, {total_gb:.0f} GB")
    print(f"lấy dư {args.lead:.0f}s+{args.tail:.0f}s mỗi ván -> clip sẽ CHỒNG LẤN, "
          f"dung lượng ra ước chừng {total_gb * 1.5:.0f}-{total_gb * 2:.0f} GB")
    free_gb = shutil.disk_usage(folder).free / 1e9
    print(f"ổ đĩa còn trống {free_gb:.0f} GB")
    if free_gb < total_gb * 1.5:
        print("CẢNH BÁO: có thể không đủ chỗ. Cân nhắc hạ --lead/--tail hoặc chạy từng phần.")

    done, failed, skipped = [], [], []
    t_all = time.time()
    for k, (dur, path, _info) in enumerate(items, 1):
        stem = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(os.path.dirname(path), stem)
        if os.path.exists(os.path.join(out, "index.csv")) and not args.overwrite:
            print(f"\n[{k}/{len(items)}] BỎ QUA (đã có kết quả): {stem[:70]}")
            skipped.append(stem)
            continue
        print(f"\n{'=' * 78}\n[{k}/{len(items)}] {dur / 60:.0f} phút — {stem[:70]}\n"
              f"{'=' * 78}", flush=True)
        t0 = time.time()
        try:
            run_one(path, out, args)
            done.append(stem)
            print(f"[{k}/{len(items)}] xong sau {(time.time() - t0) / 60:.1f} phút "
                  f"(tổng {(time.time() - t_all) / 60:.0f} phút)", flush=True)
        except Exception as e:                       # một video hỏng không được chặn cả lô
            failed.append((stem, str(e)[:200]))
            print(f"[{k}/{len(items)}] LỖI: {e}", flush=True)

    print(f"\n{'=' * 78}\nXONG {len(done)} video, bỏ qua {len(skipped)}, "
          f"lỗi {len(failed)} — tổng {(time.time() - t_all) / 60:.0f} phút")
    for stem, err in failed:
        print(f"  LỖI  {stem[:60]}: {err}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="file video, HOẶC thư mục -> chạy mọi video trong đó "
                                  "(ngắn trước dài sau)")
    ap.add_argument("--out", default=None,
                    help="mặc định là thư mục CÙNG CHỖ với video, tên = tên video")
    ap.add_argument("--stage", action="append", choices=["scan", "segment", "cut"],
                    help="chạy lẻ từng stage (lặp lại được). Bỏ trống = chạy cả 3")

    g = ap.add_argument_group("quét")
    g.add_argument("--step", type=float, default=20, help="giây/frame, quét thô")
    g.add_argument("--fine-step", type=float, default=1, help="giây/frame, quét tinh")
    g.add_argument("--fen-step", type=float, default=5,
                   help="giây/frame khi dựng chuỗi FEN trong ván (0 = không quét thêm)")
    g.add_argument("--conf", type=float, default=0.25, help="ngưỡng conf quân cờ")
    g.add_argument("--width", type=int, default=1280, help="bề ngang frame khi decode")
    g.add_argument("--repredict", action="store_true", help="bỏ cache, detect lại")
    g.add_argument("--no-roi", action="store_true", help="không dò ROI, dùng full-frame")
    g.add_argument("--roi-sample", type=int, default=20)
    g.add_argument("--roi-min-frac", type=float, default=0.35,
                   help="bàn cao hơn tỉ lệ này so với khung thì KHÔNG crop")

    g = ap.add_argument_group("tìm mốc ván")
    g.add_argument("--cand-dist", type=int, default=14,
                   help="quét thô: d <= mức này thì mở cửa sổ quét tinh. Đo thật: "
                        "mẫu thô tại mốc ván cho d = 0,3,4,6,9 còn giữa ván là "
                        "31-40 -> khoảng trống 22 ô, đặt 14 ở giữa. Để 8 là hụt "
                        "đúng ván có d=9")
    g.add_argument("--cand-pieces", type=int, default=28)
    g.add_argument("--fine-before", type=float, default=60)
    g.add_argument("--fine-after", type=float, default=30)
    g.add_argument("--start-dist", type=int, default=2, help="d tối đa của mẫu khai cuộc")
    g.add_argument("--start-pieces", type=int, default=30)
    g.add_argument("--cluster-gap", type=float, default=30,
                   help="mẫu khai cuộc cách nhau dưới mức này là cùng một ván")
    g.add_argument("--min-gap", type=float, default=120, help="hai mốc gần hơn -> gộp")
    g.add_argument("--confirm-window", type=float, default=60)
    g.add_argument("--confirm-dist", type=int, default=25)
    g.add_argument("--confirm-pieces", type=int, default=25)
    g.add_argument("--confirm-min", type=int, default=3,
                   help="số mẫu 'ván trước đã tàn' cần thấy trước mốc")
    g.add_argument("--reset-lead", type=float, default=25,
                   help="ván N kết thúc = mốc ván N+1 trừ đi mức này")
    g.add_argument("--kf-max", type=float, default=0,
                   help="chỉ decode keyframe khi khoảng cách keyframe <= mức này "
                        "(giây). 0 = tự tính = --step/8")
    g.add_argument("--chunk", type=float, default=30,
                   help="quét thô theo từng khúc bao nhiêu PHÚT (ghi cache sau mỗi "
                        "khúc -> video dài chết giữa chừng vẫn chạy tiếp được)")
    g.add_argument("--max-game", type=float, default=25,
                   help="một ván tối đa bao nhiêu phút; khoảng trống dài hơn thế bị "
                        "coi là BỎ SÓT mốc và được quét lại dày hơn (0 = tắt)")
    g.add_argument("--gap-step", type=float, default=8,
                   help="giây/frame khi quét lại các khoảng nghi bỏ sót mốc")
    g.add_argument("--min-board-frac", type=float, default=0.15,
                   help="không thấy ván nào VÀ tỉ lệ frame có bàn cờ dưới mức này "
                        "-> coi như không phải video cờ, không cắt")
    g.add_argument("--min-seen", type=int, default=2,
                   help="một FEN phải lặp lại bấy nhiêu mẫu mới được ghi")

    g = ap.add_argument_group("cắt")
    # 120s: đủ dư để không cụt (pha xếp bàn đo được chỉ ~15-20 giây), mà ghi đĩa ít
    # hơn hẳn mức 300s — trên ổ cứng cơ chính việc ghi mới là chỗ nghẽn.
    g.add_argument("--lead", type=float, default=120,
                   help="lấy dư bao nhiêu giây TRƯỚC ván (mặc định 2 phút)")
    g.add_argument("--tail", type=float, default=120,
                   help="lấy dư bao nhiêu giây SAU ván (mặc định 2 phút)")
    g.add_argument("--dry-run", action="store_true",
                   help="in lệnh ffmpeg ra chứ không cắt thật")
    g.add_argument("--overwrite", action="store_true",
                   help="ghi đè clip đã có (mặc định bỏ qua -> cắt lại được)")
    g.add_argument("--no-move", action="store_true",
                   help="KHÔNG dời video gốc vào thư mục kết quả (mặc định là dời, "
                        "đổi tên thành 00_goc_<tên>.<ext>)")
    g.add_argument("--keep-frames", action="store_true",
                   help="giữ lại thư mục _data/frames/ sau khi chạy")
    args = ap.parse_args()

    target = os.path.abspath(args.video)
    if os.path.isdir(target):
        run_folder(target, args)
        return
    stem = os.path.splitext(os.path.basename(target))[0]
    # Mặc định: thư mục CÙNG CHỖ với video, đặt tên theo video. Video gốc sẽ được
    # dời vào đây luôn nên mỗi video là một thư mục tự chứa đủ mọi thứ.
    run_one(target, args.out or os.path.join(os.path.dirname(target), stem), args)


if __name__ == "__main__":
    main()
