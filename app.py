# app.py — 8K Natural Upscaler (Clean, ≤12 MB, per-image download)
import io
from pathlib import Path
import streamlit as st
from PIL import Image, ImageFilter, ImageFile, ImageEnhance

# --- Optional: OpenCV (akan dipakai hanya jika tersedia) ---
try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False
    np = None  # placeholder agar referensi aman

# Keamanan Pillow
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

SUPPORTED = ("png","jpg","jpeg","webp","bmp")
TARGET = (7680, 4320)   # 8K fit (tanpa distorsi)
MAX_MB = 12             # batas ukuran keras
Q_MIN, Q_MAX = 60, 100  # range kualitas JPEG

st.set_page_config(page_title="8K Natural Upscaler", page_icon="🖼️", layout="wide")
st.title("🖼️ 8K Natural Upscaler")
st.caption("Upscale proporsional ke 8K dengan tampilan alami (tanpa over-sharpen). Ukuran file otomatis ≤ 12 MB. Unduh tiap gambar langsung.")

with st.sidebar:
    st.header("Pengaturan kualitas")
    resampler = st.radio("Resampler (upscale)", ["Bicubic (disarankan)", "Lanczos"], index=0)
    sharpen_amt = st.slider("Penajaman halus", 0.0, 2.0, 0.8, 0.1, help="0.6–1.0 biasanya pas; >1.2 berisiko halo.")
    micro_contrast = st.slider("Mikro-kontras", 1.0, 1.6, 1.1, 0.05, help="Sedikit saja agar tidak 'kasar'.")
    use_cv2 = st.toggle("Aktifkan OpenCV detail (opsional)", value=False and HAS_CV2,
                        help="Default OFF agar natural. Nyala hanya jika perlu, kecilkan efek.")
    fmt = st.selectbox("Format keluaran", ["JPEG (4:4:4)", "WebP"], index=0)
    suffix = st.text_input("Akhiran nama file", value="8K")

uploaded = st.file_uploader("Pilih hingga 10 gambar", type=list(SUPPORTED), accept_multiple_files=True)
if uploaded and len(uploaded) > 10:
    st.warning("Maksimal 10 gambar per proses. Hanya 10 pertama diproses.")
    uploaded = uploaded[:10]

# ---------- util ----------

def fit_size(w, h, tw, th):
    ar_s, ar_t = w / h, tw / th
    if ar_s > ar_t:
        return tw, int(tw / ar_s)
    else:
        return int(th * ar_s), th

def upscale_staged(pil_img: Image.Image, target_wh, method="bicubic", max_step=1.8):
    filt = Image.BICUBIC if method == "bicubic" else Image.LANCZOS
    w, h = pil_img.size
    tw, th = target_wh
    if tw <= w and th <= h:
        # Downscale besar: Lanczos oke
        return pil_img.resize((tw, th), Image.LANCZOS)
    out = pil_img
    while max(out.width, out.height) < max(tw, th):
        scale = min(max_step, max(tw / out.width, th / out.height))
        nw = min(tw, int(out.width * scale))
        nh = min(th, int(out.height * scale))
        if (nw, nh) == (out.width, out.height):
            break
        out = out.resize((nw, nh), filt)
    if (out.width, out.height) != (tw, th):
        out = out.resize((tw, th), filt)
    return out

def cv2_light_detail(pil_img: Image.Image):
    if not HAS_CV2:
        return pil_img
    img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    img = cv2.bilateralFilter(img, d=5, sigmaColor=20, sigmaSpace=20)  # sangat ringan
    blur = cv2.GaussianBlur(img, (0, 0), 0.7)
    img = cv2.addWeighted(img, 1.15, blur, -0.15, 0)  # unsharp halus
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def gentle_sharpen(pil_img: Image.Image, sharp=0.8, micro_c=1.1):
    out = pil_img.filter(ImageFilter.UnsharpMask(radius=0.9, percent=120, threshold=2))
    if abs(sharp - 1.0) > 1e-3:
        out = ImageEnhance.Sharpness(out).enhance(sharp)
    if abs(micro_c - 1.0) > 1e-3:
        out = ImageEnhance.Contrast(out).enhance(micro_c)
    return out

def encode_jpeg_444(im: Image.Image, q: int) -> bytes:
    if im.mode != "RGB":
        im = im.convert("RGB")
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q, optimize=True, progressive=True, subsampling=0)  # 4:4:4
    return buf.getvalue()

def encode_webp(im: Image.Image, q: int) -> bytes:
    if im.mode != "RGB":
        im = im.convert("RGB")
    buf = io.BytesIO()
    im.save(buf, "WEBP", quality=q, method=6)
    return buf.getvalue()

def maximize_under_cap(im: Image.Image, target_bytes: int, webp=False):
    encode = (lambda I, q: encode_webp(I, q)) if webp else (lambda I, q: encode_jpeg_444(I, q))
    # Jika q=100 masih <= cap → pakai q=100
    hi = encode(im, 100)
    if len(hi) <= target_bytes:
        return hi, 100, "q=100"
    # Binary search kualitas untuk tight fit (≤ cap)
    lo_q, hi_q = Q_MIN, Q_MAX
    best = (encode(im, lo_q), lo_q)
    if len(best[0]) > target_bytes:
        # turunkan kualitas (jarang untuk JPEG 8K), tetap binary
        while lo_q <= hi_q:
            mid = (lo_q + hi_q) // 2
            data = encode(im, mid)
            if len(data) <= target_bytes:
                best = (data, mid)
                lo_q = mid + 1
            else:
                hi_q = mid - 1
        return best[0], best[1], "tight fit (down)"
    else:
        # naikkan kualitas mendekati cap
        while lo_q <= hi_q:
            mid = (lo_q + hi_q) // 2
            data = encode(im, mid)
            if len(data) <= target_bytes:
                best = (data, mid)
                lo_q = mid + 1
            else:
                hi_q = mid - 1
        return best[0], best[1], "tight fit (up)"

# ---------- proses ----------
if st.button("🚀 Proses ke 8K (tanpa ZIP)"):
    if not uploaded:
        st.warning("Unggah minimal satu gambar.")
    else:
        cap_bytes = int(MAX_MB * 1024 * 1024)
        results = []
        progress = st.progress(0)
        for i, f in enumerate(uploaded, start=1):
            try:
                img = Image.open(f)
                if img.mode == "RGBA":
                    bg = Image.new("RGB", img.size, (255, 255, 255))
                    bg.paste(img, mask=img.split()[-1])
                    img = bg
                elif img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")
                else:
                    img = img.convert("RGB")

                # 1) Fit proporsional ke 8K
                tw, th = TARGET
                new_w, new_h = fit_size(img.width, img.height, tw, th)

                # 2) Upscale bertahap (Bicubic default → minim halo)
                method = "bicubic" if "Bicubic" in resampler else "lanczos"
                out = upscale_staged(img, (new_w, new_h), method=method, max_step=1.8)

                # 3) Detail (opsional) + sharpen halus
                if use_cv2 and HAS_CV2:
                    out = cv2_light_detail(out)
                out = gentle_sharpen(out, sharpen_amt, micro_contrast)

                # 4) Encode maksimal sampai mendekati 12 MB
                use_webp = fmt.startswith("WebP")
                data, used_q, note = maximize_under_cap(out, cap_bytes, webp=use_webp)
                ext = "webp" if use_webp else "jpg"
                mime = "image/webp" if use_webp else "image/jpeg"
                name = f"{Path(f.name).stem}_{suffix}.{ext}"
                size_mb = len(data) / (1024 * 1024)

                st.image(out, caption=f"{name} — {out.width}×{out.height}px | q={used_q} | {size_mb:.2f} MB ({note})",
                         use_column_width=True)
                st.download_button(f"⬇️ Unduh {name}", data, file_name=name, mime=mime)
                results.append((name, data))
            except Exception as e:
                st.error(f"Gagal memproses {f.name}: {e}")
            progress.progress(i / len(uploaded))                        
