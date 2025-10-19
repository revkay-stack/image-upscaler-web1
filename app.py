# app.py — 8K Auto Sharpen Image Upscaler (Max 12 MB)
import io, zipfile
from datetime import datetime
from pathlib import Path
import streamlit as st
from PIL import Image, ImageFilter, ImageEnhance, ImageFile

# Keamanan & kompatibilitas
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

SUPPORTED_TYPES = ("png", "jpg", "jpeg", "webp", "bmp")
TARGET_SIZE = (7680, 4320)  # 8K resolusi final
MAX_MB = 12                 # batas ukuran 12 MB

st.set_page_config(page_title="8K Image Upscaler", page_icon="🖼️", layout="wide")
st.title("🖼️ 8K Image Upscaler (Auto Sharpen)")
st.caption("Semua gambar otomatis ditingkatkan ketajamannya dan diubah ke **8K (7680×4320)** — ukuran dijaga ≤ 12 MB.")

with st.sidebar:
    st.header("Pengaturan")
    quality = st.slider("Kualitas JPEG", 70, 100, 92)
    sharp_boost = st.slider("Tingkat ketajaman tambahan", 1.0, 3.0, 1.5, 0.1)
    contrast_boost = st.slider("Tingkat kontras", 1.0, 2.0, 1.2, 0.1)
    suffix = st.text_input("Akhiran nama file", value="8K")

uploaded = st.file_uploader("Pilih hingga 10 gambar", type=list(SUPPORTED_TYPES), accept_multiple_files=True)
if uploaded and len(uploaded) > 10:
    st.warning("Maksimal 10 gambar per proses. Hanya 10 pertama diproses.")
    uploaded = uploaded[:10]

# ---------------- fungsi ----------------
def resize_to_8k(img: Image.Image) -> Image.Image:
    w, h = img.size
    tw, th = TARGET_SIZE
    aspect_src = w / h
    aspect_tgt = tw / th
    if aspect_src > aspect_tgt:
        new_w = tw
        new_h = int(tw / aspect_src)
    else:
        new_h = th
        new_w = int(th * aspect_src)
    return img.resize((new_w, new_h), Image.LANCZOS)

def enhance_image(img: Image.Image, sharp_boost: float, contrast_boost: float) -> Image.Image:
    # Penajaman multi-step + peningkatan kontras & detail
    img = img.filter(ImageFilter.UnsharpMask(radius=1.8, percent=180, threshold=2))
    img = ImageEnhance.Sharpness(img).enhance(sharp_boost)
    img = ImageEnhance.Contrast(img).enhance(contrast_boost)
    img = img.filter(ImageFilter.UnsharpMask(radius=0.8, percent=130, threshold=2))
    return img

def save_jpeg_limit(img: Image.Image, quality: int, max_bytes: int):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
    if len(buf.getvalue()) <= max_bytes:
        return buf.getvalue(), quality
    q = quality
    while len(buf.getvalue()) > max_bytes and q > 40:
        q -= 5
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q, optimize=True, progressive=True)
    return buf.getvalue(), q

# ---------------- proses utama ----------------
if st.button("🚀 Proses ke 8K"):
    if not uploaded:
        st.warning("Unggah minimal satu gambar.")
    else:
        progress = st.progress(0)
        results = []
        for i, f in enumerate(uploaded, start=1):
            try:
                img = Image.open(f).convert("RGB")
                out = resize_to_8k(img)
                out = enhance_image(out, sharp_boost, contrast_boost)
                data, used_q = save_jpeg_limit(out, quality, int(MAX_MB * 1024 * 1024))
                name = f"{Path(f.name).stem}_{suffix}.jpg"
                st.image(out, caption=f"{name} — {out.width}×{out.height}px (q={used_q}, ≤12 MB)", use_column_width=True)
                st.download_button(f"⬇️ Unduh {name}", data, file_name=name, mime="image/jpeg")
                results.append((name, data))
            except Exception as e:
                st.error(f"Gagal memproses {f.name}: {e}")
            progress.progress(i / len(uploaded))
        if results:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for n, d in results:
                    zf.writestr(n, d)
            st.download_button("📦 Unduh Semua (ZIP)", buf.getvalue(), file_name="upscaled_8k.zip", mime="application/zip")

st.markdown("---")
st.markdown("""
### ℹ️ Catatan
- Semua output otomatis diubah ke **8K (7680×4320)** proporsional.
- Ketajaman & kontras ditingkatkan otomatis (Unsharp Mask + Sharpness + Contrast).
- File akhir dioptimasi agar ≤ 12 MB, kualitas maksimal tanpa error.
""")
