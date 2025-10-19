# app.py — 8K Upscaler (Detail-Preserving + Direct Downloads)
import io, math
from pathlib import Path
import streamlit as st
from PIL import Image, ImageFilter, ImageEnhance, ImageFile
import numpy as np

# Coba pakai OpenCV headless untuk peningkatan detail
try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

# Keamanan & kompatibilitas Pillow
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

SUPPORTED_TYPES = ("png", "jpg", "jpeg", "webp", "bmp")
TARGET_SIZE = (7680, 4320)   # 8K fit (proporsional)
MAX_MB = 12                  # batas ukuran file

st.set_page_config(page_title="8K Image Upscaler", page_icon="🖼️", layout="wide")
st.title("🖼️ 8K Image Upscaler — Detail Preserving")
st.caption("Semua gambar di-upscale proporsional ke **8K (fit)**, ditajamkan cerdas, dan **maks ≤ 12 MB**. Unduh tiap gambar langsung (tanpa ZIP).")

with st.sidebar:
    st.header("Pengaturan")
    # Penajaman adaptif (post-resize)
    sharp_boost = st.slider("Tingkat ketajaman", 1.0, 3.0, 1.6, 0.1)
    contrast_boost = st.slider("Tingkat kontras", 1.0, 2.0, 1.15, 0.05)
    # Detail enhancer OpenCV
    if HAS_CV2:
        detail_strength = st.slider("Detail enhance (OpenCV)", 0, 100, 35, 5, help="Semakin besar, semakin muncul micro-detail; jangan berlebihan.")
    else:
        st.info("OpenCV tidak terpasang — fallback Pillow akan dipakai (tetap tajam).")
        detail_strength = 0
    suffix = st.text_input("Akhiran nama file", value="8K")
    show_zip_toggle = st.toggle("Tampilkan tombol ZIP juga (opsional)", value=False)

uploaded = st.file_uploader(
    "Pilih hingga 10 gambar",
    type=list(SUPPORTED_TYPES),
    accept_multiple_files=True
)
if uploaded and len(uploaded) > 10:
    st.warning("Maksimal 10 gambar per proses. Hanya 10 pertama diproses.")
    uploaded = uploaded[:10]

# ---------- Utilitas ----------

def fit_to_box_wh(w, h, tw, th):
    """Hitung ukuran proporsional agar muat di (tw, th) tanpa distorsi."""
    aspect_src = w / h
    aspect_tgt = tw / th
    if aspect_src > aspect_tgt:
        new_w = tw
        new_h = int(tw / aspect_src)
    else:
        new_h = th
        new_w = int(th * aspect_src)
    return new_w, new_h

def staged_resize_lanczos(pil_img: Image.Image, target_wh, max_step=1.8):
    """
    Upscale bertahap (≤ max_step per tahap) untuk mengurangi artefak.
    """
    w, h = pil_img.size
    tw, th = target_wh
    # Jika turun ukuran besar → langsung sekali
    if tw <= w and th <= h:
        return pil_img.resize((tw, th), Image.LANCZOS)

    out = pil_img
    while max(out.width, out.height) < max(tw, th):
        scale = min(max_step, max(tw / out.width, th / out.height))
        nw = min(tw, int(out.width * scale))
        nh = min(th, int(out.height * scale))
        if nw == out.width and nh == out.height:
            break
        out = out.resize((nw, nh), Image.LANCZOS)
    if (out.width, out.height) != (tw, th):
        out = out.resize((tw, th), Image.LANCZOS)
    return out

def cv2_detail_pipeline(pil_img: Image.Image, strength: int):
    """
    Pipeline detail-preserving berbasis OpenCV:
    - Bilateral smoothing ringan untuk kurangi noise
    - DetailEnhance (edge-aware)
    - CLAHE pada channel L di LAB (kontras lokal)
    - Unsharp mask halus
    """
    if not HAS_CV2 or strength <= 0:
        return pil_img

    img = np.array(pil_img.convert("RGB"))[:, :, ::-1]  # PIL RGB -> BGR
    # Noise reduction ringan (bilateral kecil agar edge terjaga)
    img = cv2.bilateralFilter(img, d=5, sigmaColor=30, sigmaSpace=30)

    # Detail enhance (sigma_r ~ 0.1..0.25 tergantung strength)
    sigma_r = max(0.05, min(0.25, strength / 200.0))
    sigma_s = 10 + int(strength / 4)   # spatial
    img = cv2.detailEnhance(img, sigma_s=sigma_s, sigma_r=sigma_r)

    # CLAHE pada L channel
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    img = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    # Unsharp mask halus (radius 0.8 ~ 1.2)
    blur = cv2.GaussianBlur(img, (0,0), sigmaX=0.8)
    img = cv2.addWeighted(img, 1.25, blur, -0.25, 0)

    out = Image.fromarray(img[:, :, ::-1])  # BGR -> RGB back to PIL
    return out

def pillow_sharpen_pipeline(pil_img: Image.Image, sharp_boost: float, contrast_boost: float):
    """
    Fallback jika tanpa OpenCV: kombinasi Unsharp + Sharpness + Contrast.
    """
    out = pil_img.filter(ImageFilter.UnsharpMask(radius=1.6, percent=170, threshold=2))
    out = ImageEnhance.Sharpness(out).enhance(sharp_boost)
    out = ImageEnhance.Contrast(out).enhance(contrast_boost)
    out = out.filter(ImageFilter.UnsharpMask(radius=0.8, percent=120, threshold=2))
    return out

def encode_jpeg_444(im: Image.Image, q: int):
    """
    JPEG 4:4:4 (subsampling=0) untuk menjaga detail warna saat zoom.
    """
    if im.mode != "RGB":
        im = im.convert("RGB")
    buf = io.BytesIO()
    im.save(
        buf, format="JPEG",
        quality=q, optimize=True, progressive=True,
        subsampling=0  # 4:4:4 = tidak buang chroma detail
    )
    return buf.getvalue()

def save_jpeg_maximize(im: Image.Image, target_bytes: int, q_min=40, q_max=100):
    """
    Naikkan kualitas setinggi mungkin hingga mendekati target_bytes (≤ MAX_MB).
    - Jika q=100 masih < target → pakai q=100 (maksimal).
    - Binary search untuk tight fit.
    """
    data_hi = encode_jpeg_444(im, q_max)
    if len(data_hi) <= target_bytes:
        return data_hi, q_max, "q=100 (maksimum)"

    data_lo = encode_jpeg_444(im, q_min)
    if len(data_lo) > target_bytes:
        # Binary search turun
        lo, hi = q_min, q_max
        best = (data_lo, q_min)
        while lo <= hi:
            mid = (lo + hi) // 2
            data = encode_jpeg_444(im, mid)
            if len(data) <= target_bytes:
                best = (data, mid)
                lo = mid + 1
            else:
                hi = mid - 1
        return best[0], best[1], "binary down (tight fit)"

    # Binary search naik (kasus umum)
    lo, hi = q_min, q_max
    best = (data_lo, q_min)
    while lo <= hi:
        mid = (lo + hi) // 2
        data = encode_jpeg_444(im, mid)
        if len(data) <= target_bytes:
            best = (data, mid)
            lo = mid + 1
        else:
            hi = mid - 1
    return best[0], best[1], "binary up (tight fit)"

# ---------- Proses ----------
if st.button("🚀 Proses ke 8K (tanpa ZIP)"):
    if not uploaded:
        st.warning("Unggah minimal satu gambar.")
    else:
        results = []
        progress = st.progress(0)
        target_bytes = int(MAX_MB * 1024 * 1024)

        for i, f in enumerate(uploaded, start=1):
            try:
                # Load & konversi aman
                img = Image.open(f)
                if img.mode not in ("RGB", "RGBA", "L"):
                    img = img.convert("RGB")
                elif img.mode == "RGBA":
                    # Hilangkan alpha ke putih (agar hasil JPEG clean)
                    bg = Image.new("RGB", img.size, (255, 255, 255))
                    bg.paste(img, mask=img.split()[-1])
                    img = bg
                else:
                    img = img.convert("RGB")

                # 1) Tentukan ukuran target fit 8K (proporsional)
                tw, th = TARGET_SIZE
                new_w, new_h = fit_to_box_wh(img.width, img.height, tw, th)

                # 2) Upscale bertahap untuk mengurangi artefak
                up = staged_resize_lanczos(img, (new_w, new_h), max_step=1.8)

                # 3) Detail enhancement (OpenCV jika ada, else Pillow)
                if HAS_CV2 and detail_strength > 0:
                    up = cv2_detail_pipeline(up, detail_strength)
                else:
                    up = pillow_sharpen_pipeline(up, sharp_boost, contrast_boost)

                # 4) Maksimalkan kualitas sampai ~12MB (JPEG 4:4:4)
                data, used_q, note = save_jpeg_maximize(up, target_bytes)

                # 5) Tampilkan & tombol unduh (langsung per-gambar)
                name = f"{Path(f.name).stem}_{suffix}.jpg"
                size_mb = len(data) / (1024 * 1024)
                st.image(up, caption=f"{name} — {up.width}×{up.height}px | q={used_q} | {size_mb:.2f} MB ({note})", use_column_width=True)
                st.download_button(f"⬇️ Unduh {name}", data, file_name=name, mime="image/jpeg")

                results.append((name, data))
            except Exception as e:
                st.error(f"Gagal memproses {f.name}: {e}")
            progress.progress(i / len(uploaded))

        # ZIP opsional — hanya kalau kamu menyalakan toggle di sidebar
        if results and show_zip_toggle:
            import zipfile
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for n, d in results:
                    zf.writestr(n, d)
            st.download_button("📦 (Opsional) Unduh Semua (ZIP)", buf.getvalue(), file_name="upscaled_8k.zip", mime="application/zip")

st.markdown("---")
st.markdown("""
**Tips kejernihan saat zoom**
- Gunakan sumber foto berkualitas tinggi (noise rendah).
- Hindari kompresi berlapis (JPEG → edit → JPEG → edit); usahakan dari PNG/RAW bila ada.
- Pipeline ini: **staged Lanczos → detail enhance (edge-aware) → CLAHE (OpenCV) / Unsharp** → **JPEG 4:4:4**.
""")
