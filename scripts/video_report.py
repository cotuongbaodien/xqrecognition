"""Báo cáo cắt video: mỗi video cắt được mấy ván, và toàn hệ thống có bao nhiêu.

Dựng LẠI TỪ ĐẦU theo trạng thái thật trên đĩa (`index.csv` + `_data/scan.json` của
từng thư mục), nên chạy lúc nào cũng ra số đúng — không phụ thuộc log hay bộ nhớ.

  python scripts/video_report.py                       # in ra + ghi file báo cáo
  python scripts/video_report.py --watch 300           # tự dựng lại mỗi 5 phút

Ghi ra hai file cạnh kho video:
  _BAO_CAO.md    người đọc: bảng từng video + tổng
  _BAO_CAO.csv   máy đọc: mỗi video một dòng, mở bằng Excel

Một video coi là XONG khi có `index.csv`. Video chưa có file đó là chưa cắt (hoặc
đang cắt dở) và được liệt kê riêng ở mục "chờ".
"""
import argparse
import csv
import json
import os
import time

VIDEO_EXT = (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".ts", ".webm", ".flv")
DEFAULT_ROOTS = [r"E:\videos\GiangHo"]


def hm(sec):
    sec = int(sec or 0)
    return f"{sec // 3600}h{sec % 3600 // 60:02d}"


def scan_root(root):
    """(danh sách video đã cắt, danh sách video chờ)."""
    done, waiting = [], []
    if not os.path.isdir(root):
        return done, waiting
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if os.path.isfile(p) and name.lower().endswith(VIDEO_EXT):
            waiting.append({"ten": name, "root": root,
                            "gb": os.path.getsize(p) / 1e9})
            continue
        if not os.path.isdir(p) or name.startswith("_"):
            continue
        idx = os.path.join(p, "index.csv")
        if not os.path.exists(idx):
            waiting.append({"ten": name + "  (đang cắt dở)", "root": root, "gb": 0})
            continue
        rows = []
        try:
            rows = [r for r in csv.DictReader(open(idx, encoding="utf-8"))]
        except OSError:
            pass
        clip_gb = 0.0
        for f in os.listdir(p):
            if "_van" in f and f.lower().endswith(VIDEO_EXT) \
                    and not f.endswith(".shrink.mp4"):
                clip_gb += os.path.getsize(os.path.join(p, f)) / 1e9
        src_sec = 0.0
        sj = os.path.join(p, "_data", "scan.json")
        if os.path.exists(sj):
            try:
                src_sec = float(json.load(open(sj, encoding="utf-8")).get("duration", 0))
            except Exception:
                pass
        van_sec = [float(r.get("van_ket_thuc", 0) or 0) - float(r.get("van_bat_dau", 0) or 0)
                   for r in rows]
        done.append({
            "ten": name, "root": root, "so_van": len(rows),
            "dai_nguon_s": src_sec, "clip_gb": clip_gb,
            "van_dai_nhat_s": max(van_sec) if van_sec else 0,
            "van_dai_qua_30p": sum(1 for v in van_sec if v > 1800),
        })
    return done, waiting


def build(roots, out_dir):
    done, waiting = [], []
    for r in roots:
        d, w = scan_root(r)
        done += d
        waiting += w

    n_van = sum(x["so_van"] for x in done)
    h_src = sum(x["dai_nguon_s"] for x in done) / 3600
    gb = sum(x["clip_gb"] for x in done)
    n_long = sum(x["van_dai_qua_30p"] for x in done)
    gb_wait = sum(x["gb"] for x in waiting)

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "_BAO_CAO.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["video", "so_van", "dai_nguon_phut", "clip_GB",
                    "van_dai_nhat_phut", "van_dai_qua_30p"])
        for x in sorted(done, key=lambda x: -x["so_van"]):
            w.writerow([x["ten"], x["so_van"], round(x["dai_nguon_s"] / 60),
                        round(x["clip_gb"], 2), round(x["van_dai_nhat_s"] / 60),
                        x["van_dai_qua_30p"]])

    md = [f"# Báo cáo cắt video — {time.strftime('%H:%M %d/%m/%Y')}", "",
          "| | |", "|---|---|",
          f"| **Video đã cắt xong** | **{len(done)}** |",
          f"| **TỔNG SỐ VÁN** | **{n_van}** |",
          f"| Giờ nội dung đã xử lý | {h_src:.1f} giờ |",
          f"| Dung lượng clip | {gb:.0f} GB |",
          f"| Trung bình | {n_van / max(len(done), 1):.1f} ván/video |",
          f"| Ván dài hơn 30 phút (nghi dính 2 ván) | {n_long} |",
          f"| Còn chờ cắt | {len(waiting)} video · {gb_wait:.0f} GB |", "",
          "## Từng video (nhiều ván nhất trước)", "",
          "| Video | Ván | Dài nguồn | Clip | Ván dài nhất |", "|---|---|---|---|---|"]
    for x in sorted(done, key=lambda x: -x["so_van"]):
        flag = f" ⚠{x['van_dai_qua_30p']}" if x["van_dai_qua_30p"] else ""
        md.append(f"| {x['ten'][:58]} | **{x['so_van']}**{flag} | "
                  f"{hm(x['dai_nguon_s'])} | {x['clip_gb']:.1f} GB | "
                  f"{x['van_dai_nhat_s'] / 60:.0f} phút |")
    if waiting:
        md += ["", f"## Chờ cắt ({len(waiting)})", ""]
        md += [f"- {x['ten'][:70]}" + (f" · {x['gb']:.1f} GB" if x["gb"] else "")
               for x in waiting[:60]]
        if len(waiting) > 60:
            md.append(f"- … và {len(waiting) - 60} video nữa")
    md += ["", "⚠ = số ván dài hơn 30 phút, nhiều khả năng hai ván bị dính làm một; "
           "xem cách tách lại ở `docs/VIDEO_BATCH_RUNBOOK.md`."]
    # Ghi NGUYÊN TỬ: pipeline và vòng --watch có thể cùng dựng báo cáo một lúc; ghi
    # đè thẳng thì một bên đọc/ghi phải file cụt (đã làm chết vòng watch một lần).
    md_path = os.path.join(out_dir, "_BAO_CAO.md")
    tmp = md_path + ".tmp"
    open(tmp, "w", encoding="utf-8").write("\n".join(md) + "\n")
    os.replace(tmp, md_path)

    print(f"{len(done)} video đã cắt · {n_van} VÁN · {h_src:.1f} giờ nguồn · "
          f"{gb:.0f} GB clip · còn chờ {len(waiting)}")
    return md_path, n_van


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", default=None)
    ap.add_argument("--out", default=None, help="mặc định = thư mục root đầu tiên")
    ap.add_argument("--watch", type=float, default=0,
                    help="dựng lại mỗi N giây (0 = chạy một lần)")
    args = ap.parse_args()
    roots = args.root or DEFAULT_ROOTS
    out = args.out or roots[0]
    while True:
        path, _ = build(roots, out)
        if not args.watch:
            print(f"-> {path}")
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
