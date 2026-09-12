"""Thêm tiền tố `YYYYMMDD_` vào tên thư mục kết quả để sắp theo ngày quay.

Thư mục tải từ YouTube đã có sẵn dạng `20260423_tên [id]`. Thư mục từ file quay tay
thì không, nên trộn lẫn vào là hết sắp được. Script này chuẩn hoá phần còn lại.

Ngày lấy theo thứ tự tin cậy GIẢM DẦN:
  1. `com.apple.quicktime.creationdate` — iPhone ghi lúc bấm quay, đáng tin nhất.
  2. `creation_time` trong container.
  3. **Cụm 6 số `YYMMDD` trong tên thư mục** — quy ước đặt tên của kho này. Đã đối
     chiếu: `minhla26042302` nằm trong thư mục YouTube `20260423_...` → đúng YYMMDD.
  4. `mtime` của file gốc — CHỐT CUỐI, hay sai vì đó là lúc copy/chia file chứ không
     phải lúc quay (5 thư mục "Chú Việt vs Minh La - Phần 1..5" đều ra cùng một ngày).

Thư mục chia phần (`... - P1`, `... Phần 3`) mà chỉ còn `mtime` thì **thừa hưởng ngày
của thành viên cùng nhóm có nguồn tốt hơn** — `MinhLaSonHang Manh - P1` lấy ngày của
`MinhLaSonHang Manh` thay vì lúc file được chia ra. Tên đã mở đầu bằng đúng cụm 6 số
ngày đó thì cắt bỏ cho khỏi lặp (`260731Thần+thâu` -> `20260731_Thần+thâu`).

Chỉ đổi tên THƯ MỤC. Tên clip bên trong giữ nguyên — `index.csv` chỉ lưu tên file nên
không vỡ, `_data/scan.json` có đường dẫn tuyệt đối nhưng đã cũ từ trước (bản gốc đã
được dời vào trong thư mục thành `00_goc_*`, mọi công cụ tìm qua `src_of()`).

  python scripts/video_rename_date.py --dry-run     # xem sẽ đổi thành gì
  python scripts/video_rename_date.py               # đổi thật

BỎ QUA thư mục nào còn file nguồn cùng tên ở tầng ngoài — đổi tên thư mục đó sẽ làm
`video_split.py` tưởng chưa cắt và cắt lại từ đầu.
"""
import argparse
import csv
import datetime as dt
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

VIDEO_EXT = (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".ts", ".webm", ".flv")
DA_CO_NGAY = re.compile(r"^\d{8}[_ ]")
# tach ten nhom / so phan: "X - P2", "X Phan 3", "X_part02"
PHAN = [re.compile(pat, re.I) for pat in (
    r"^(.*?)[ _-]*(?:part|phan|phần|ph)[ _-]?(\d{1,2})$",
    r"^(.*?)[ _-]p(\d{1,2})$",
)]
TIN_CAY = {"apple": 3, "creation_time": 2, "tên thư mục": 1, "mtime": 0}
# YYMMDD: nam 15-26, thang 01-12, ngay 01-31. Doi so it nhat 6 chu so lien tiep.
YYMMDD = re.compile(r"(?<!\d)(1[5-9]|2[0-6])(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)")


def ten_nhom(name):
    for pat in PHAN:
        m = pat.match(name)
        if m and m.group(1).strip(" _-"):
            return m.group(1).strip(" _-").lower()
    return None


def probe_tags(video):
    r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json",
                        "-show_format", video],
                       capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    try:
        return json.loads(r.stdout).get("format", {}).get("tags", {}) or {}
    except Exception:
        return {}


def hop_le(s):
    """'20260423' -> True nếu là ngày thật và không ở tương lai xa."""
    try:
        d = dt.datetime.strptime(s, "%Y%m%d").date()
    except ValueError:
        return False
    return dt.date(2015, 1, 1) <= d <= dt.date.today() + dt.timedelta(days=2)


def ngay_tu_ten(name):
    m = YYMMDD.search(name)
    if not m:
        return None
    s = f"20{m.group(1)}{m.group(2)}{m.group(3)}"
    return s if hop_le(s) else None


def lay_ngay(folder, video):
    """(ngay 'YYYYMMDD', nguon, ngay_theo_ten) — ngay_theo_ten để báo lệch."""
    theo_ten = ngay_tu_ten(os.path.basename(folder))
    t = probe_tags(video) if video else {}
    for key, ten_nguon in (("com.apple.quicktime.creationdate", "apple"),
                           ("creation_time", "creation_time")):
        v = (t.get(key) or "")[:10].replace("-", "")
        if hop_le(v):
            return v, ten_nguon, theo_ten
    if theo_ten:
        return theo_ten, "tên thư mục", theo_ten
    if video:
        v = dt.datetime.fromtimestamp(os.path.getmtime(video)).strftime("%Y%m%d")
        if hop_le(v):
            return v, "mtime", theo_ten
    return None, "không rõ", theo_ten


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=r"E:\videos\GiangHo")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sep", default="_", help="ký tự nối ngày với tên (mặc định '_')")
    a = ap.parse_args()

    root = a.root
    # file nguon con o tang ngoai -> thu muc trung ten voi no khong duoc doi
    treo = {os.path.splitext(f)[0] for f in os.listdir(root)
            if os.path.isfile(os.path.join(root, f))
            and f.lower().endswith(VIDEO_EXT)}

    can, da_co, bo = [], 0, []
    for n in sorted(os.listdir(root)):
        d = os.path.join(root, n)
        if not os.path.isdir(d) or n.startswith("_"):
            continue
        if not os.path.exists(os.path.join(d, "index.csv")):
            continue
        if DA_CO_NGAY.match(n):
            da_co += 1
            continue
        if n in treo:
            bo.append(n)
            continue
        can.append((n, d))

    print(f"{da_co} thư mục đã có tiền tố ngày")
    print(f"{len(bo)} thư mục BỎ QUA (còn file nguồn cùng tên ở tầng ngoài)")
    print(f"{len(can)} thư mục cần thêm ngày\n")

    doi, lech, hong, dem = [], [], [], {}
    for n, d in can:
        g = [f for f in os.listdir(d) if f.startswith("00_goc_")]
        video = os.path.join(d, g[0]) if g else None
        ngay, nguon, theo_ten = lay_ngay(d, video)
        if not ngay:
            hong.append(n)
            continue
        dem[nguon] = dem.get(nguon, 0) + 1
        if theo_ten and theo_ten != ngay:
            lech.append((n, ngay, nguon, theo_ten))
        doi.append((n, d, ngay, nguon))

    # nhóm chia phần: thành viên chỉ có mtime thừa hưởng ngày của thành viên tốt hơn
    tot_nhat = {}
    for n, d, ngay, nguon in doi:
        g = ten_nhom(n) or n.lower()
        cu = tot_nhat.get(g)
        if cu is None or TIN_CAY[nguon] > TIN_CAY[cu[1]]:
            tot_nhat[g] = (ngay, nguon)
    ke = 0
    for i, (n, d, ngay, nguon) in enumerate(doi):
        if nguon != "mtime":
            continue
        g = ten_nhom(n)
        if not g or g not in tot_nhat:
            continue
        ngay2, nguon2 = tot_nhat[g]
        if TIN_CAY[nguon2] > 0 and ngay2 != ngay:
            doi[i] = (n, d, ngay2, f"{nguon2} (cùng nhóm)")
            dem["mtime"] -= 1
            dem["thừa hưởng"] = dem.get("thừa hưởng", 0) + 1
            ke += 1
    if ke:
        print(f"{ke} thư mục chia phần đã thừa hưởng ngày của bản cùng nhóm" + chr(10))

    print("nguồn ngày:", ", ".join(f"{k}={v}" for k, v in sorted(dem.items())))
    if hong:
        print(f"\n{len(hong)} thư mục KHÔNG lấy được ngày, giữ nguyên:")
        for n in hong[:10]:
            print(f"   {n[:64]}")
    if lech:
        print(f"\n{len(lech)} thư mục có ngày trong TÊN lệch với metadata "
              f"(theo metadata, tên chỉ để tham khảo):")
        for n, ngay, nguon, tt in lech[:15]:
            print(f"   {n[:44]:44s} metadata {ngay} ({nguon})  vs tên {tt}")

    print(f"\n{len(doi)} thư mục sẽ đổi tên:")
    xong = 0
    for n, d, ngay, nguon in doi:
        goc = n
        if n.startswith(ngay[2:]):
            goc = n[6:].lstrip(" _-") or n
        moi = f"{ngay}{a.sep}{goc}"
        dst = os.path.join(root, moi)
        k = 2
        while os.path.exists(dst):
            moi = f"{ngay}{a.sep}{goc}_{k}"
            dst = os.path.join(root, moi)
            k += 1
        if xong < 12 or a.dry_run:
            print(f"   {n[:50]:50s} -> {moi[:60]}   [{nguon}]")
        if not a.dry_run:
            os.rename(d, dst)
        xong += 1
    if a.dry_run:
        print("\n(--dry-run: chưa đổi gì. Bỏ cờ đó để đổi thật)")
    else:
        print(f"\nđã đổi tên {xong} thư mục")


if __name__ == "__main__":
    main()
