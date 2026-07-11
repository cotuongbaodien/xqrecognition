# OCR flow — bức tranh mobile ↔ portal ↔ OCR engine (để debug "báo bận")

> Mục đích: tài liệu cho **cả team Mobile và team OCR/engine** soi cùng một chỗ.
> Bối cảnh: mobile gọi OCR **"cứ báo bận"**. Theo mobile spec, **HTTP 500/502/503 → "Server đang bận, thử lại sau"** (`docs/ocr-phase1-mobile-spec.md`). Tức "bận" = client nhận **5xx** ở đâu đó trong chuỗi.
> Cập nhật: **2026-06-25** (đã tìm ra & xử lý root cause — xem mục 0). Prod portal: `103.175.146.124` (PM2 `ctns-portal` :3100).

---

## 0. TL;DR — ✅ ĐÃ TÌM RA & XỬ LÝ ROOT CAUSE (2026-06-25)

> **Root cause của "báo bận" Flow B = CÓ 2 ENGINE OCR cùng tồn tại, `ocr.abcxq.app` trỏ vào engine CŨ.**
> Trên máy nhà từng chạy **2 engine**: (a) MỚI `ocr-gpu` (standalone `ocr-gpu-service`, có verify token) phục vụ **`ocr-gpu.abcxq.app`**; (b) CŨ `abcengine-ocr` (service `ocr` trong stack `pikafishcloudengine`, **CHỈ biết `X-OCR-Secret`, không hiểu token**) phục vụ **`ocr.abcxq.app`**. Hai hostname là **2 tunnel khác nhau** → 2 engine khác nhau (KHÔNG phải "cùng 1 service" như tài liệu cũ ghi nhầm).
> → Token Flow B gọi vào `ocr.abcxq.app` thì **luôn 401** (engine cũ không verify token). Đo thực tế 2026-06-25: `ocr.abcxq.app/detect` + token = **10/10 → 401**; `ocr-gpu.abcxq.app/detect` + token = **10/10 → 200**.

**Đã xử lý (2026-06-25):**
1. ✅ Engine MỚI verify token đầy đủ: chữ ký HMAC + `scope` + `exp` + `jti` single-use (mục 3 & 5).
2. ✅ **Tắt + xoá hẳn engine OCR CŨ** (`abcengine-ocr`) khỏi stack `pikafishcloudengine` (container + image + thư mục `ocr/`). Pikafish (cờ tướng) không bị ảnh hưởng.
3. ✅ **Portal prod đã set `OCR_PUBLIC_URL=https://ocr-gpu.abcxq.app/detect`** → mobile nhận `ocr_url` trỏ đúng engine mới (xác nhận bằng `.env.production` + log call thật `auth=token` → 200).
4. ✅ Engine log thêm `detect auth=secret|token` để phân biệt Flow A/B.

**Còn lại (không gấp):** `ocr.abcxq.app` giờ là **domain chết (502)** — route cũ trên tunnel `pikafish-engine` vẫn trỏ vào engine đã tắt. App KHÔNG dùng nó nên vô hại. Khi tiện (hoặc khi move pikafish lên VPS): **xoá public hostname `ocr.abcxq.app` khỏi tunnel `pikafish-engine`** trên Cloudflare dashboard.

> Nghi can phụ còn theo dõi: 5xx do **GPU máy nhà sleep-restart / Cloudflare Tunnel chập chờn** (mục 6) — và có hiện tượng **docker log capture đóng băng sau sleep-resume** (recreate container để gắn lại log).

---

## 1. Các thành phần

| Thành phần | Địa chỉ | Vai trò |
|---|---|---|
| Mobile app | (iOS/Android) | Chụp ảnh bàn cờ → cần FEN |
| Portal (Next.js) | `https://nghiasingapi.abcxq.app` (origin VPS `103.175.146.124:3100`) | Auth/quota/relay/cấp token |
| OCR engine (DUY NHẤT, từ 2026-06-25) | `https://ocr-gpu.abcxq.app` → container `ocr-gpu` (repo `ocr-gpu-service`, GPU máy nhà qua tunnel `ocr-gpu`) | Nhận ảnh → detect → trả FEN. Có verify cả `X-OCR-Secret` lẫn `X-OCR-Token` |
| ~~OCR engine cũ~~ | ~~`ocr.abcxq.app` → `abcengine-ocr`~~ | **ĐÃ XOÁ 2026-06-25**. `ocr.abcxq.app` giờ 502 (domain mồ côi trên tunnel `pikafish-engine`) |
| Secret chung | `OCR_SHARED_SECRET` (xem `.env.production`) | Dùng cho cả header tĩnh lẫn ký token HMAC |

---

## 2. FLOW A — Relay (CŨ, đang chạy OK)

App KHÔNG gọi thẳng OCR; ảnh đi qua portal.

```
 ┌────────┐   1. POST /api/ocr/board-to-fen            ┌──────────────┐   2. POST /detect            ┌─────────────┐
 │ Mobile │ ──────────────────────────────────────────▶│  Portal VPS  │ ─────────────────────────────▶│ OCR engine  │
 │  app   │   Authorization: Bearer <JWT access>        │ (Next.js)    │   Header: X-OCR-Secret=<SECRET>│ ocr-gpu.    │
 │        │   x-integrity-token: <...>                  │              │   multipart: image=@file      │ abcxq.app   │
 │        │   multipart: image=@photo.jpg               │  gates:      │                               │             │
 │        │                                             │  - JWT       │   3. 200 {detected, fen,...}  │  YOLO       │
 │        │◀────────────────────────────────────────── │  - integrity │◀───────────────────────────── │  detect     │
 └────────┘   4. 200 {detected, fen, error_code, quota} │  - quota     │                               └─────────────┘
                                                         └──────────────┘
```

- **Auth app→portal**: JWT access (HS256, ký `JWT_SECRET`, claims `sub`=accountId, `device_id`).
- **Auth portal→engine**: header tĩnh **`X-OCR-Secret` == `OCR_SHARED_SECRET`**.
- **Đã verify 2026-06-18**: gọi thật `POST /api/ocr/board-to-fen` (JWT + ảnh) → **HTTP 200**, 0.56s. ✅ Flow A hoạt động.

---

## 3. FLOW B — Token, app upload thẳng (MỚI) — ✅ engine đã hỗ trợ

Mục tiêu: ảnh KHÔNG qua portal nữa (nhanh hơn). App xin token rẻ rồi tự upload ảnh lên engine.

```
 ┌────────┐  1. POST /api/ocr/token            ┌──────────────┐
 │ Mobile │ ───────────────────────────────────▶│  Portal VPS  │  (gates: JWT + integrity + rate-limit)
 │  app   │  Authorization: Bearer <JWT access> │              │
 │        │◀─────────────────────────────────── │              │  2. 200 { token, ocr_url, expires_at, tier, quota }
 │        │     token = "<payload>.<hmac_sig>"   └──────────────┘     (token TTL 120s, ký bằng OCR_SHARED_SECRET)
 │        │
 │        │  3. POST <ocr_url> (= ocr-gpu.abcxq.app/detect)    ┌─────────────┐
 │        │ ──────────────────────────────────────────────────▶│ OCR engine  │
 │        │     Header: X-OCR-Token: <token>                  │  ✅ engine  │
 │        │     multipart: image=@photo.jpg                   │  verify     │
 │        │                                                    │  token HMAC │
 │        │◀────────────────────────────────────────────────  │  (_verify_  │
 └────────┘  4. ✅ 200 {detected, fen, ...} (token hợp lệ)     │  token)     │
                                                                └─────────────┘
```

### ✅ Trạng thái Flow B (đã FIX phía engine 2026-06-18)
- Portal cấp **token HMAC**: `base64url(JSON payload) + "." + base64url(HMAC_SHA256(payload, OCR_SHARED_SECRET))`.
  - payload = `{ exp:<unix+120s>, jti:<uuid>, uid:<accountId>, dev:<deviceId>, scope:"detect" }`.
- **Engine ĐÃ verify token** trong `boarddetection/server.py`:
  - `/detect` nhận **2 header**: `X-OCR-Secret` (Flow A, secret tĩnh) **HOẶC** `X-OCR-Token` (Flow B, token HMAC). Một trong hai đúng là pass → **Flow A & B chạy song song, không gãy nhau.**
  - `_verify_token()` kiểm tra theo thứ tự: **(1) chữ ký HMAC → (2) `scope=="detect"` → (3) `exp` chưa hết hạn → (4) bắt buộc có `jti` → (5) `jti` single-use (chống replay)**.
- **Header chốt: `X-OCR-Token`** (đây là điểm mobile cần làm đúng — đừng nhét token vào `X-OCR-Secret` hay `Authorization`).
- **Chống replay**: `jti` được nhớ in-process tới khi token hết hạn. **Lưu ý deployment**: hiện chỉ 1 worker uvicorn (docker-compose không set `--workers`) nên cache RAM là đủ. Nếu sau này scale nhiều worker/replica → **phải chuyển sang Redis `EX=ttl`** để dedup dùng chung.

---

## 4. Kết quả test (2026-06-18) — cái gì ĐANG OK

| Test | Kết quả |
|---|---|
| `GET ocr-gpu.abcxq.app/health` | 200 `{"status":"ok","model":"loaded"}` (~0.4s local) |
| `GET ocr.abcxq.app/health` | 200 (cùng service) |
| `POST ocr-gpu.abcxq.app/detect` + `X-OCR-Secret` + ảnh | **200**, 39–326ms (nhanh, GPU-class) |
| `POST /detect` KHÔNG secret | 401 (auth tĩnh hoạt động) |
| Portal `POST /api/ocr/token` (JWT hợp lệ) | 200, token HMAC verify chữ ký OK, TTL 120s |
| Portal `POST /api/ocr/board-to-fen` (JWT + ảnh) | **200**, 0.56s (Flow A OK) |
| ⚠️ Từ **VPS** gọi `ocr-gpu.abcxq.app/health` | **5.5s** (so với 0.4s từ local) — độ trễ VPS→engine cao bất thường, cần điều tra |

→ Không reproduce được "bận" lúc test rời rạc ⇒ "bận" 5xx là **chập chờn/theo tải**. (Khả năng "Flow B bị engine từ chối" giờ đã loại bỏ vì engine đã verify token — xem mục 3.)

### 4b. Test 2026-06-25 — chứng minh root cause 2-engine (trước khi tắt engine cũ)

| Test | Kết quả |
|---|---|
| `ocr.abcxq.app/detect` + **token** × 10 | **10/10 → 401** (engine CŨ, không hiểu token) |
| `ocr-gpu.abcxq.app/detect` + **token** × 10 | **10/10 → 200** (engine MỚI) |
| Log call THẬT của app (qua tunnel) | `auth=token` → **200** vào engine MỚI → app dùng `ocr-gpu.abcxq.app` |

→ Khẳng định: `ocr.abcxq.app` và `ocr-gpu.abcxq.app` là **2 engine khác nhau**, app đi vào engine mới. Sau khi **tắt engine cũ**: `ocr.abcxq.app` → **502**, `ocr-gpu.abcxq.app` → **200** (app không ảnh hưởng).

---

## 5. Token contract — ✅ ĐÃ implement trong engine

> **Trạng thái:** đã có trong `boarddetection/server.py` (`_verify_token`, header `X-OCR-Token`).
> Engine nhận token qua header **`X-OCR-Token`**; nếu hợp lệ thì pass, song song với
> `X-OCR-Secret` tĩnh (Flow A). Pseudo dưới đây mô tả đúng những gì code đang làm.

```python
# pseudo (FastAPI). secret = OCR_SHARED_SECRET (bytes)
import hmac, hashlib, base64, json, time

def b64url_decode(s): return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))

def verify_ocr_token(token: str, secret: bytes, seen_jti: set) -> bool:
    try:
        p, sig = token.split(".")
    except ValueError:
        return False
    expect = base64.urlsafe_b64encode(
        hmac.new(secret, p.encode(), hashlib.sha256).digest()
    ).rstrip(b"=").decode()
    if not hmac.compare_digest(sig, expect):       # 1) chữ ký
        return False
    payload = json.loads(b64url_decode(p))
    if payload.get("scope") != "detect":           # 2) scope
        return False
    if payload["exp"] < int(time.time()):          # 3) hết hạn (TTL 120s)
        return False
    if payload["jti"] in seen_jti:                 # 4) single-use (chống replay)
        return False
    seen_jti.add(payload["jti"])                   # nên lưu Redis EX 120s thay vì set RAM
    return True
```

- ✅ Giữ **tương thích ngược**: code nhận `X-OCR-Secret == OCR_SHARED_SECRET` (Flow A/relay) **HOẶC** `X-OCR-Token` hợp lệ (Flow B). Đường cũ của app không bị gãy.
- ⚠️ `jti` single-use hiện lưu **in-process** (đủ cho 1 worker uvicorn hiện tại). Khi scale nhiều worker/replica → đổi sang **Redis EX=ttl** để dedup dùng chung (chống replay across worker/restart).

---

## 6. Checklist team OCR / engine (ocr-gpu-service) — debug "bận" (5xx)

- [ ] **Log service** lúc mobile báo bận: có **5xx** không? mã gì (500/502/503)? message gì?
- [x] **Log phân biệt flow**: mỗi `/detect` giờ log `detect auth=secret|token ...` → biết ngay request đi **Flow A (secret)** hay **Flow B (token)**. ✅ (`docker logs ocr-gpu | grep 'detect auth='`)
- [ ] **Concurrency**: service xử lý ảnh **tuần tự** (1 model GPU)? Khi nhiều request đồng thời có **queue/timeout/OOM** → trả 503 không? Nếu có giới hạn, **trả 503 + header `Retry-After`** có chủ đích (đừng để crash).
- [ ] **GPU machine ở nhà**: uptime? có **sleep / mất điện / mất mạng / reboot**? card GPU có bị chiếm/OOM?
- [ ] **Cloudflare Tunnel**: tunnel có **disconnect/restart**? Khi tunnel down, Cloudflare trả **502/530** → app thấy "bận". Check `cloudflared` logs.
- [ ] **Độ trễ**: vì sao **VPS→engine /health 5.5s** mà local 0.4s? Tunnel routing? Region? Ảnh lớn + chậm → vượt timeout relay (`OCR_BACKEND_TIMEOUT_MS=25000`).
- [x] **Auth**: engine nhận **`X-OCR-Secret`** (Flow A) **và** **`X-OCR-Token`** (Flow B) — đã implement (mục 5). ✅
- [ ] Đo **thời gian detect** với ảnh thật (không phải ảnh test) — có request nào > vài giây?

## 6b. Checklist team MOBILE — debug "bận"

- [ ] App đang dùng **Flow A (relay `/api/ocr/board-to-fen`)** hay **Flow B (`/api/ocr/token` → upload thẳng)**? (Quyết định debug ở đâu.)
- [ ] Lúc báo "bận": **HTTP code thực tế + response body** là gì? (chụp log/charles/proxy) — 401? 403? 500? 502? 503? timeout?
- [ ] Request lỗi đi tới **URL nào**? Host OCR sống DUY NHẤT là **`ocr-gpu.abcxq.app`** (token `ocr_url` phải là host này). ⚠️ `ocr.abcxq.app` đã chết (502) — nếu thấy app gọi host này nghĩa là `ocr_url`/config sai.
- [ ] Nếu Flow B: app **phải** gắn token vào header **`X-OCR-Token`** (KHÔNG phải `X-OCR-Secret` / `Authorization`) — đây là tên engine verify. ✅ engine đã sẵn sàng.
- [ ] **Kích thước ảnh** gửi lên (MB)? Engine giới hạn `too_large` → 400. Timeout client app = bao nhiêu giây?
- [ ] Đã có **retry backoff** (1s→3s→10s) theo spline chưa? "bận" thường là tạm thời.
- [ ] Token còn hạn? **TTL chỉ 120s** — nếu xin token rồi chụp/sửa ảnh lâu > 2 phút mới upload → token hết hạn → engine từ chối.

---

## 7. Vấn đề cấu hình phát hiện (nên fix)

| # | Vấn đề | Hiện trạng | Đề xuất |
|---|---|---|---|
| 0 | ~~**2 engine OCR, `ocr.abcxq.app` trỏ engine cũ không-token**~~ ✅ ĐÃ XỬ LÝ | Đã xoá engine cũ `abcengine-ocr`; chỉ còn `ocr-gpu`. Đây là root cause "báo bận" Flow B | Done. Việc dọn còn lại: xoá hostname `ocr.abcxq.app` khỏi tunnel `pikafish-engine` (CF dashboard) để hết 502 mồ côi |
| 1 | ~~Host backend không nhất quán~~ ✅ ĐÃ FIX | Prod đã set `OCR_PUBLIC_URL=https://ocr-gpu.abcxq.app/detect`; relay `OCR_BACKEND_URL` cũng `ocr-gpu.abcxq.app` → **đồng nhất 1 host = engine mới** | Done. (Lưu ý: code app default ghi `ocr.abcxq.app` nhưng portal env override đúng) |
| 2 | ~~Engine chưa verify token HMAC~~ ✅ ĐÃ FIX | Nhận `X-OCR-Token` (HMAC + scope + exp + jti single-use) song song `X-OCR-Secret` | Done. Mobile chỉ cần gửi đúng header `X-OCR-Token` |
| 3 | **Độ trễ VPS→engine 5.5s** | Quan sát 1 lần | Điều tra tunnel/route; ảnh hưởng cả Flow A (relay) |
| 4 | **Docker log đóng băng sau sleep-resume** | Sau khi máy nhà sleep ~14h, `docker logs ocr-gpu` đứng yên dù engine vẫn serve (call vẫn 200) | Recreate container (`docker compose up -d --force-recreate ocr`) để gắn lại luồng log |

---

## 8. Lệnh test nhanh (cả 2 team tự chạy)

```bash
SECRET="<OCR_SHARED_SECRET, xem .env.production>"

# Engine sống?
curl -s -m 15 https://ocr-gpu.abcxq.app/health        # mong: {"status":"ok","model":"loaded"}

# Detect bằng secret tĩnh (Flow A path) — phải 200
curl -s -m 35 -X POST -H "X-OCR-Secret: $SECRET" -F "image=@board.jpg" \
  https://ocr-gpu.abcxq.app/detect

# Detect KHÔNG secret — phải 401 (auth hoạt động)
curl -s -o /dev/null -w "%{http_code}\n" -X POST -F "image=@board.jpg" \
  https://ocr-gpu.abcxq.app/detect
```

> Lưu ý: gọi từ **IP datacenter** có thể bị Cloudflare bot-challenge (trả HTML) — test từ máy thường hoặc thêm header trình duyệt.
