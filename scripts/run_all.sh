#!/bin/bash
# Chay mot mach: cat -> soat -> nen -> gop ban tai ve -> cat tiep -> soat -> nen -> bao cao.
# Moi buoc cho buoc truoc xong han; KHONG bao gio chay hai viec gianh GPU/o dia cung luc.
cd /c/Resources/xqrecognition
LOG=/e/videos/GiangHo/_pipeline.log
export PYTHONIOENCODING=utf-8
CUT_ARGS="--fen-step 0 --lead 120 --tail 120 --reencode never"

say(){ echo "" >> $LOG; echo "### $(date '+%H:%M %d/%m') $*" >> $LOG; }
# PHAI loc theo TEN tien trinh truoc. Neu chi loc theo CommandLine thi chinh cau lenh
# powershell nay cung chua chuoi can tim -> no TU KHOP VOI MINH -> cho mai mai (da dinh).
nproc_match(){ powershell.exe -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe' or Name='yt-dlp.exe'\" | Where-Object { \$_.CommandLine -like '*$1*' } | Measure-Object).Count" 2>/dev/null | tr -d '\r' | tail -1; }
wait_gone(){ # $1 = chuoi trong command line, $2 = ten de log
  say "cho $2 xong"
  while true; do
    n=$(nproc_match "$1"); [ -z "$n" ] && n=0
    [ "$n" -le 0 ] && break
    sleep 60
  done
  say "$2 da xong"
}

# --- 1. cho lo cat hien tai ket thuc ---
wait_gone "video_split.py" "lo cat dang chay"

# --- 2. soat + nen dot 1 ---
say "soat clip dot 1"
python -u scripts/video_verify.py "E:\videos\GiangHo" >> $LOG 2>&1
say "nen clip dot 1 (thu hoi dung luong)"
python -u scripts/video_shrink.py "E:\videos\GiangHo" >> $LOG 2>&1
python -u scripts/video_report.py >> $LOG 2>&1

# --- 3. cho tai xong roi gop vao kho ---
wait_gone "yt_download.py" "tai YouTube"
wait_gone "yt-dlp" "yt-dlp con sot"
say "gop file tai ve vao E:\\videos\\GiangHo"
python - <<'PY' >> $LOG 2>&1
import os, shutil
dst = r"E:\videos\GiangHo"
EXT = (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".ts", ".webm")
moved = 0
for src in [r"E:\videos\_yt_tai", r"D:\videos\_yt_tai"]:
    if not os.path.isdir(src):
        continue
    for f in sorted(os.listdir(src)):
        p = os.path.join(src, f)
        if not os.path.isfile(p) or not f.lower().endswith(EXT):
            continue
        stem, ext = os.path.splitext(f)
        target, k = os.path.join(dst, f), 2
        # Trung ten file HOAC trung ten thu muc ket qua deu phai doi ten,
        # neu khong se ghi de ban cu hoac lan vao thu muc da cat.
        while os.path.exists(target) or os.path.isdir(os.path.join(dst, stem)):
            stem = f"{os.path.splitext(f)[0]}_{k}"
            target = os.path.join(dst, stem + ext)
            k += 1
        shutil.move(p, target)
        moved += 1
print(f"da gop {moved} file tai ve vao kho")
PY

# --- 4. cat dot 2 (phan vua gop) ---
say "cat dot 2"
for i in $(seq 1 40); do
  python -u scripts/video_split.py "E:\videos\GiangHo" $CUT_ARGS >> /e/videos/GiangHo/_video_split.log 2>&1 && break
  say "tien trinh cat chet -> khoi dong lai lan $i"
done

# --- 5. soat + nen dot 2 ---
say "soat clip dot 2"
python -u scripts/video_verify.py "E:\videos\GiangHo" >> $LOG 2>&1
say "nen clip dot 2"
python -u scripts/video_shrink.py "E:\videos\GiangHo" >> $LOG 2>&1

# --- 6. bao cao cuoi ---
say "bao cao cuoi"
python -u scripts/video_report.py >> $LOG 2>&1
python -u scripts/video_dupes.py >> $LOG 2>&1
say "HET PIPELINE"
