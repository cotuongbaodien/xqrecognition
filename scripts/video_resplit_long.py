"""Soi lại những "ván" dài bất thường — gần như chắc chắn là NHIỀU VÁN DÍNH LÀM MỘT.

Ván cờ chớp không thể dài 30 phút. Ván dài như vậy nghĩa là lượt cắt đã **bỏ sót mốc
bắt đầu** ở giữa, thường vì lúc xếp lại bàn tay che kín hoặc người chơi đi ngay khi
vừa xếp xong nên không mẫu nào rơi trúng thế khai cuộc.

Cách chữa KHÔNG phải cắt lại từ đầu (đắt), mà là:
  1. Nới ngưỡng nhận thế khai cuộc (`--start-dist` 4 thay vì 2, `--start-pieces` 29
     thay vì 30, `--min-gap` 90 thay vì 120, `--confirm-min` 2 thay vì 3).
  2. Quét lại DÀY HƠN đúng khoảng nghi ngờ (`--gap-step`, 4 s/frame thay vì 20).
  3. Tính lại mốc từ dữ liệu đó.

Bước 1 gần như MIỄN PHÍ — chỉ tính lại trên `scan.json` sẵn có, không nạp model, không
decode gì. Đo trên 321 thư mục: **132 thư mục ra thêm 252 ván** chỉ nhờ nới ngưỡng.
Bước 2 mới đắt (~6 phút/thư mục), để dành cho chỗ nào nới ngưỡng vẫn còn ván dài.
Chạy `--no-rescan` để chỉ làm bước 1.

CỔNG CHẶN MẤT VÁN: ngưỡng lỏng hơn KHÔNG bảo đảm ra nhiều mốc hơn — thêm mẫu vào một
cụm làm cụm dài ra, mốc chốt (mẫu cuối cụm) trôi về sau rồi dính luật gộp `--min-gap`
và biến mất. Đo thật: 24/321 thư mục mất tổng 28 mốc đang có. Nên mặc định thư mục nào
có mốc cũ KHÔNG tìm lại được (lệch quá `--giu-tol` giây) thì **bỏ qua nguyên thư mục**;
`--cho-mat-moc` để tắt cổng này.

Model nạp MỘT lần cho cả lô (chạy `video_split.py` từng thư mục thì mỗi lần nạp lại
mất 5-8 giây, nhân với hàng trăm thư mục là vô lý).

  python scripts/video_resplit_long.py                 # chỉ BÁO CÁO, không đụng file
  python scripts/video_resplit_long.py --apply         # cắt lại thư mục nào tách được thêm

`--apply` xoá clip cũ + index.csv của ĐÚNG thư mục có thay đổi rồi cắt lại; bản gốc
`00_goc_*` không bao giờ bị đụng.
"""
import argparse
import csv
import json
import os
import sys
import time
from argparse import Namespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import video_split as vs  # noqa: E402

VIDEO_EXT = (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".ts", ".webm")


def make_args(a):
    """Namespace đầy đủ cho các hàm của video_split (chúng đọc thẳng args.*)."""
    return Namespace(
        # ngưỡng nhận mốc — chỗ được NỚI so với lượt cắt đầu
        start_dist=a.start_dist, gap_step=a.gap_step, max_game=a.max_game,
        cand_dist=14, cand_pieces=28, start_pieces=a.start_pieces,
        cluster_gap=30, min_gap=a.min_gap, confirm_window=60, confirm_dist=25,
        confirm_pieces=25, confirm_min=a.confirm_min, reset_lead=25,
        min_board_frac=0.15, min_seen=2, fen_step=0,
        # quét
        step=20, fine_step=1, fine_before=60, fine_after=30, conf=0.25,
        width=1280, repredict=False, no_roi=False, roi_sample=20,
        roi_min_frac=0.35, kf_max=0, chunk=30,
        # cắt
        lead=120, tail=120, reencode="never", reencode_above=2.5,
        encoder="h264_nvenc", cq=30, max_width=1280,
        dry_run=False, overwrite=True, no_move=True, keep_frames=False,
        stage=None, out=None, video=None,
    )


def src_of(folder):
    g = [f for f in os.listdir(folder) if f.startswith("00_goc_")]
    if g:
        return os.path.join(folder, g[0])
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=r"E:\videos\GiangHo")
    ap.add_argument("--min-long", type=float, default=30,
                    help="ván dài hơn bao nhiêu PHÚT thì đem soi lại")
    ap.add_argument("--start-dist", type=int, default=4,
                    help="nới ngưỡng nhận thế khai cuộc (lượt cắt đầu dùng 2)")
    ap.add_argument("--start-pieces", type=int, default=29)
    ap.add_argument("--gap-step", type=float, default=4,
                    help="giây/frame khi quét lại khoảng nghi ngờ")
    ap.add_argument("--max-game", type=float, default=20,
                    help="khoảng dài hơn mức này bị quét lại (lượt đầu dùng 25)")
    ap.add_argument("--min-gap", type=float, default=90,
                    help="hai mốc gần hơn mức này thì gộp (lượt đầu dùng 120)")
    ap.add_argument("--confirm-min", type=int, default=2,
                    help="số mẫu 'ván trước đã tàn' cần thấy (lượt đầu dùng 3)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-rescan", action="store_true",
                    help="chỉ tính lại trên cache sẵn có, không quét dày thêm (rất nhanh)")
    ap.add_argument("--giu-tol", type=float, default=60,
                    help="mốc cũ coi là còn nếu mốc mới nằm trong ngần này giây")
    ap.add_argument("--cho-mat-moc", action="store_true",
                    help="nhận cả thư mục làm mất mốc cũ (mặc định: bỏ qua)")
    ap.add_argument("--apply", action="store_true",
                    help="cắt lại thư mục nào tách được thêm ván")
    a = ap.parse_args()

    root = a.root
    todo = []
    for n in sorted(os.listdir(root)):
        d = os.path.join(root, n)
        idx = os.path.join(d, "index.csv")
        if not os.path.isdir(d) or n.startswith("_") or not os.path.exists(idx):
            continue
        longest = 0.0
        rows = list(csv.DictReader(open(idx, encoding="utf-8")))
        moc_cu = []
        for r in rows:
            try:
                longest = max(longest, float(r.get("van_ket_thuc", 0))
                              - float(r.get("van_bat_dau", 0)))
                moc_cu.append(float(r["van_bat_dau"]))
            except (KeyError, ValueError):
                pass
        if longest > a.min_long * 60:
            todo.append((d, len(rows), longest, sorted(moc_cu)))
    todo.sort(key=lambda x: -x[2])
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(todo)} thư mục có ván dài hơn {a.min_long:.0f} phút\n")

    args = make_args(a)
    t0 = time.time()
    them = 0
    doi = []
    bo_vi_mat = 0
    for k, (d, n_old, longest, moc_cu) in enumerate(todo, 1):
        video = src_of(d)
        cache = vs.load_json(os.path.join(d, "_data", "scan.json"))
        if not video or not cache or not cache.get("frames"):
            print(f"[{k}/{len(todo)}] {os.path.basename(d)[:44]:44s} bỏ (thiếu nguồn/cache)")
            continue
        frames = cache["frames"]
        starts, _ = vs.find_starts(frames, args)
        # quét dày lại đúng những khoảng còn dài bất thường
        bounds = [0.0] + [s["t"] for s in starts] + [cache["duration"]]
        gaps = [] if a.no_rescan else             [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
             if bounds[i + 1] - bounds[i] > a.max_game * 60]
        if gaps:
            cache = vs.scan_spans(video, d, cache, gaps, a.gap_step, args.conf, "kẽ hở")
            frames = cache["frames"]
            starts, _ = vs.find_starts(frames, args)
        games = vs.build_games(starts, cache["duration"], args)
        n_new = len(games)
        # cổng chặn: mốc cũ nào không tìm lại được thì thư mục này KHÔNG an toàn
        moc_moi = [g["start"] for g in games]
        mat = [o for o in moc_cu
               if not any(abs(o - x) <= a.giu_tol for x in moc_moi)]
        mark = ""
        if mat and not a.cho_mat_moc:
            bo_vi_mat += 1
            print(f"[{k}/{len(todo)}] {os.path.basename(d)[:44]:44s} "
                  f"{n_old:3d} -> {n_new:3d} ván  BỎ QUA: mất {len(mat)} mốc cũ "
                  f"({', '.join(f'{m/60:.0f}p' for m in mat[:4])})", flush=True)
            continue
        if n_new > n_old:
            them += n_new - n_old
            doi.append((d, n_old, n_new, games, cache, video))
            mark = f"  ==> +{n_new - n_old} ván"
        print(f"[{k}/{len(todo)}] {os.path.basename(d)[:44]:44s} "
              f"{n_old:3d} -> {n_new:3d} ván (dài nhất {longest/60:.0f}p){mark}",
              flush=True)

    print(f"\n{len(doi)}/{len(todo)} thư mục tách được thêm — TỔNG +{them} ván "
          f"({(time.time()-t0)/60:.0f} phút)")

    if not a.apply:
        print("\n(chỉ báo cáo — thêm --apply để cắt lại những thư mục đó)")
        return
    for i, (d, n_old, n_new, games, cache, video) in enumerate(doi, 1):
        for f in os.listdir(d):
            if "_van" in f and f.lower().endswith(VIDEO_EXT):
                os.remove(os.path.join(d, f))
        idx = os.path.join(d, "index.csv")
        if os.path.exists(idx):
            os.remove(idx)
        vs.save_json(games, os.path.join(vs.work_dir(d), "games.json"), indent=1)
        print(f"[{i}/{len(doi)}] cắt lại {os.path.basename(d)[:44]} ({n_new} ván)",
              flush=True)
        vs.stage_cut(video, d, games, cache["duration"], args)
    print(f"\nđã cắt lại {len(doi)} thư mục, +{them} ván")


if __name__ == "__main__":
    main()
