# app.py — 8K Auto Sharpen, Maximize up to 12 MB
import io, zipfile
from datetime import datetime
from pathlib import Path
import streamlit as st
from PIL import Image, ImageFilter, ImageEnhance, ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

SUPPORTED_TYPES = ("png", "jpg", "jpeg", "webp", "bmp")
TARGET_SIZE = (7680, 4320)   # 8K
MAX_MB = 12                  # target size cap
Q_MIN, Q_MAX = 40, 100       # search range

st.set_page_config(page_title="8K Image Upscaler", page_icon="🖼️", layout="wide")
st.title("🖼️ 8K Image Upscaler (Maximized up to 12 MB)")

with st.sidebar:
    st.header("Pengaturan")
    # Nilai default dibuat agresif tapi aman
    start_quality = st.slider("Kualitas awal (petunjuk)", 80, 100, 92)
    sharp_boost = st.slider("Tingkat ketajaman", 1.0, 3.0, 1.5, 0.1)
    contrast_boost = st.slider("Tingkat kontras", 1.0, 2.0, 1.2, 0.1)
    suffix = st.text_input("Akhiran nama file", value="8K")

uploaded = st.file_uploader("Pilih hingga 10 gambar", type=list(SUPPORTED_TYPES), accept_multiple_files=True)
if uploaded and len(uploaded) > 10:
    st.warning("Maksimal 10 gambar per proses. Hanya 10 pertama diproses.")
    uploaded = uploaded[:10]

def resize_to_8k(img: Image.Image) -> Image.Image:
    w, h = img.size
    tw, th = TARGET_SIZE
    # fit proporsional di dalam 8K (tanpa distorsi)
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
    # Unsharp mask + sharpness + contrast untuk ketajaman alami
    img = img.filter(ImageFilter.UnsharpMask(radius=1.8, percent=180, threshold=2))
    img = ImageEnhance.Sharpness(img).enhance(sharp_boost)
    img = ImageEnhance.Contrast(img).enhance(contrast_boost)
    img = img.filter(ImageFilter.UnsharpMask(radius=0.8, percent=130, threshold=2))
    return img

def encode_jpeg(im: Image.Image, q: int, subsampling: int = 0) -> bytes:
    """Encode ke JPEG dengan opsi kualitas dan subsampling (0=4:4:4 full chroma)."""
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    buf = io.BytesIO()
    im.save(
        buf, format="JPEG",
        quality=q,
        optimize=True,
        progressive=True,
        subsampling=subsampling  # 0=4:4:4 (detail warna maksimal)
    )
    return buf.getvalue()

def save_jpeg_maximize(im: Image.Image, target_bytes: int, q_hint: int = 92):
    """
    Maksimalkan kualitas hingga mendekati target_bytes dengan binary search.
    - Jika ukuran di q=100 masih < target → kita pakai q=100 (maks detail).
    - Jika di q=40 pun > target → tetap diturunkan dengan binary search sampai pas (jarang terjadi di 8K).
    """
    # 1) Coba batas atas dulu: kalau q=100 <= target → selesai (tak perlu kecil)
    data_hi = encode_jpeg(im, Q_MAX, subsampling=0)
    if len(data_hi) <= target_bytes:
        return data_hi, Q_MAX, "q=100 (cap belum tercapai)"

    # 2) Coba batas bawah: kalau q=40 masih > target → lakukan pencarian turun (tetap binary)
    data_lo = encode_jpeg(im, Q_MIN, subsampling=0)
    if len(data_lo) > target_bytes:
        # Binary search pada [Q_MIN, Q_MAX] cari kualitas tertinggi yang <= target
        lo, hi = Q_MIN, Q_MAX
        best = (data_lo, Q_MIN)
        while lo <= hi:
            mid = (lo + hi) // 2
            data = encode_jpeg(im, mid, subsampling=0)
            if len(data) <= target_bytes:
                best = (data, mid)
                lo = mid + 1
            else:
                hi = mid - 1
        return best[0], best[1], "binary down (tight fit)"

    # 3) Kasus umum: ukuran awal (q_hint) mungkin < target → naikkan kualitas hingga mendekati batas (binary up)
    #    Cari kualitas tertinggi yang masih <= target
    lo = max(q_hint, Q_MIN)
    hi = Q_MAX
    # Pastikan ada baseline yang <= target: pakai q_hint, kalau > target turunkan jadi Q_MIN
    data_base = encode_jpeg(im, lo, subsampling=0)
    if len(data_base) > target_bytes:
        lo = Q_MIN
        data_base = data_lo
    best = (data_base, lo)
    while lo <= hi:
        mid = (lo + hi) // 2
        data = encode_jpeg(im, mid, subsampling=0)
        if len(data) <= target_bytes:
            best = (data, mid)
            lo = mid + 1   # coba lebih tinggi lagi
        else:
            hi = mid - 1   # turunkan kualitas
    return best[0], best[1], "binary up (tight fit)"

if st.button("🚀 Proses ke 8K (maks 12 MB)"):
    if not uploaded:
        st.warning("Unggah minimal satu gambar.")
    else:
        results = []
        progress = st.progress(0)
        target_bytes = int(MAX_MB * 1024 * 1024)

        for i, f in enumerate(uploaded, start=1):
            try:
                img = Image.open(f).convert("RGB")
                out = resize_to_8k(img)
                out = enhance_image(out, sharp_boost, contrast_boost)

                data, used_q, note = save_jpeg_maximize(out, target_bytes, q_hint=start_quality)

                name = f"{Path(f.name).stem}_{suffix}.jpg"
                size_mb = len(data) / (1024 * 1024)
                st.image(out, caption=f"{name} — {out.width}×{out.height}px | q={used_q} | {size_mb:.2f} MB ({note})", use_column_width=True)
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
**Catatan kualitas**
- Encoder memakai **subsampling=0 (4:4:4)** → detail warna maksimal.
- Algoritma **binary search** menaikkan kualitas setinggi mungkin hingga mendekati **12 MB**.
- Jika pada **q=100** ukuran masih < 12 MB, itu normal untuk gambar yang mudah dikompresi — kamu sudah di kualitas maksimum.
""")
