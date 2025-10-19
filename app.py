# app.py
import io, zipfile
from datetime import datetime
from pathlib import Path
import streamlit as st
from PIL import Image, ImageFilter

SUPPORTED_TYPES = ("png", "jpg", "jpeg", "webp", "bmp")

st.set_page_config(page_title="Batch Image Upscaler", page_icon="🖼️", layout="wide")
st.title("🖼️ Batch Image Upscaler (Fixed Version)")
st.caption("Upscale hingga **10 gambar sekaligus** dengan metode Lanczos berkualitas tinggi.")

with st.sidebar:
    scale = st.select_slider("Skala", options=[2,3,4], value=4)
    sharpen = st.slider("Penajaman", 0, 3, 1)
    suffix = st.text_input("Akhiran nama file output", value=f"x{scale}")

uploaded = st.file_uploader("Pilih hingga 10 gambar", type=list(SUPPORTED_TYPES), accept_multiple_files=True)
if uploaded and len(uploaded) > 10:
    st.warning("Maksimal 10 gambar per proses.")
    uploaded = uploaded[:10]

def upscale_lanczos(img: Image.Image, factor: int, sharpen_steps: int=1) -> Image.Image:
    new_size = (img.width * factor, img.height * factor)
    out = img.resize(new_size, Image.LANCZOS)
    for _ in range(sharpen_steps):
        out = out.filter(ImageFilter.UnsharpMask(radius=1.2, percent=150, threshold=3))
    return out

def img_bytes(im: Image.Image, fmt:str) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format=fmt)
    return buf.getvalue()

if st.button("🚀 Proses Upscale"):
    if not uploaded:
        st.warning("Unggah minimal 1 gambar.")
    else:
        results = []
        progress = st.progress(0)
        for i, f in enumerate(uploaded, start=1):
            try:
                img = Image.open(f)
                out = upscale_lanczos(img, scale, sharpen)
                ext = f.name.split('.')[-1].lower()
                fmt = 'PNG' if ext == 'png' else 'JPEG'
                name = f"{Path(f.name).stem}_{suffix}.{ 'png' if fmt=='PNG' else 'jpg'}"
                st.image(out, caption=f"{name} ({out.width}×{out.height})", use_column_width=True)
                data = img_bytes(out, fmt)
                st.download_button(f"⬇️ Unduh {name}", data, file_name=name, mime=f"image/{fmt.lower()}")
                results.append((name, data))
            except Exception as e:
                st.error(f"Gagal memproses {f.name}: {e}")
            progress.progress(i/len(uploaded))
        if results:
            zipbuf = io.BytesIO()
            with zipfile.ZipFile(zipbuf, "w", zipfile.ZIP_DEFLATED) as zf:
                for n, d in results:
                    zf.writestr(n, d)
            st.download_button("📦 Unduh semua (ZIP)", zipbuf.getvalue(), file_name="upscaled_images.zip", mime="application/zip")

st.markdown("---")
st.markdown("""
### ℹ️ Catatan
- Versi ini sudah diperbaiki (tanpa error `use_container_width`).
- Menggunakan metode Lanczos bawaan Pillow.
- Bisa langsung dijalankan di Streamlit Cloud tanpa error instalasi.
""")
