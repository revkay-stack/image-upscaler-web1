# app.py — 4K / 8K Image Upscaler (Auto Size Limit)
import io, zipfile
from datetime import datetime
from pathlib import Path
import streamlit as st
from PIL import Image, ImageFilter, ImageFile

# Keamanan & kompatibilitas
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

SUPPORTED_TYPES = ("png", "jpg", "jpeg", "webp", "bmp")

st.set_page_config(page_title="4K / 8K Image Upscaler", page_icon="🖼️", layout="wide")
st.title("🖼️ 4K / 8K Image Upscaler")
st.caption("Upscale gambar ke resolusi **4K (3840×2160)** atau **8K (7680×4320)** secara proporsional dan otomatis membatasi ukuran file.")

# Sidebar pengaturan
with st.sidebar:
    st.header("Pengaturan")
    target_res = st.radio(
        "Pilih resolusi output:",
        ["4K (3840×2160)", "8K (7680×4320)"],
        index=1,
    )
    sharpen = st.slider("Penajaman", 0, 3, 1)
    quality = st.slider("Kualitas JPEG", 70, 100, 90)
    suffix = st.text_input("Akhiran nama file", value="upscaled")

# Tentukan resolusi dan batas MB otomatis
if "4K" in target_res:
    TARGET_SIZE = (3840, 2160)
    MAX_MB = 12
else:
    TARGET_SIZE = (7680, 4320)
    MAX_MB = 16

uploaded = st.file_uploader("Pilih hingga 10 gambar", type=list(SUPPORTED_TYPES), accept_multiple_files=True)
if uploaded and len(uploaded) > 10:
    st.warning("Maksimal 10 gambar per proses. Hanya 10 pertama diproses.")
    uploaded = uploaded[:10]

def resize_to_target(img: Image.Image, target_size=(3840, 2160)) -> Image.Image:
    w, h = img.size
    tw, th = target_size
    aspect_src = w / h
    aspect_tgt = tw / th
    if aspect_src > aspect_tgt:
        new_w = tw
        new_h = int(tw / aspect_src)
    else:
        new_h = th
        new_w = int(th * aspect_src)
    return img.resize((new_w, new_h), Image.LANCZOS)

def sharpen_img(img: Image.Image, steps: int) -> Image.Image:
    for _ in range(steps):
        img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=150, threshold=3))
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

if st.button("🚀 Proses Upscale"):
    if not uploaded:
        st.warning("Unggah minimal satu gambar.")
    else:
        progress = st.progress(0)
        results = []
        for i, f in enumerate(uploaded, start=1):
            try:
                img = Image.open(f).convert("RGB")
                out = resize_to_target(img, TARGET_SIZE)
                out = sharpen_img(out, sharpen)
                max_bytes = int(MAX_MB * 1024 * 1024)
                data, used_q = save_jpeg_limit(out, quality, max_bytes)
                name = f"{Path(f.name).stem}_{suffix}.jpg"
                st.image(out, caption=f"{name} — {out.width}×{out.height}px (q={used_q}, ≤{MAX_MB} MB)", use_column_width=True)
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
            st.download_button("📦 Unduh Semua (ZIP)", buf.getvalue(), file_name="upscaled_images.zip", mime="application/zip")

st.markdown("---")
st.markdown(f"""
### ℹ️ Catatan
- Semua gambar otomatis diubah ke **{target_res.split()[0]}** resolusi proporsional.
- Batas ukuran file otomatis:  
  • **4K → 12 MB**  
  • **8K → 16 MB**
- Output JPEG dioptimasi otomatis agar tidak melebihi batas tersebut.
""")
