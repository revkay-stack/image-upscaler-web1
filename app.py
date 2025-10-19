# app.py — 8K Upscaler (No Crop, No Blur/Solid) via Mirror Padding, ≤12 MB, per-image download
import io
from pathlib import Path
import streamlit as st
from PIL import Image, ImageFilter, ImageFile, ImageEnhance
import numpy as np
import cv2

# Safety
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

SUPPORTED = ("png","jpg","jpeg","webp","bmp")
TARGET_W, TARGET_H = 7680, 4320   # 8K exact
MAX_MB = 12
Q_MIN, Q_MAX = 60, 100            # JPEG 4:4:4 quality range

st.set_page_config(page_title="8K Upscaler — Mirror Padding", page_icon="🖼️", layout="wide")
st.title("🖼️ 8K Upscaler — No Crop, No Blur/Solid")
st.caption("Output selalu **8K (7680×4320)** tanpa cropping dan tanpa blur/solid: padding berupa **refleksi tepi**. Jernih & ≤ 12 MB. Unduh per gambar.")

with st.sidebar:
    st.header("Pengaturan kualitas")
    resampler = st.radio("Resampler (upscale)", ["Bicubic (disarankan)", "Lanczos"], index=0)
    sharpen_amt = st.slider("Penajaman halus", 0.0, 2.0, 0.8, 0.1, help="0.6–1.0 biasanya pas; >1.2 berisiko halo.")
    micro_contrast = st.slider("Mikro-kontras", 1.0, 1.6, 1.1, 0.05)
    suffix = st.text_input("Akhiran nama file", value="8K")

uploaded = st.file_uploader("Pilih hingga 10 gambar", type=list(SUPPORTED), accept_multiple_files=True)
if uploaded and len(uploaded) > 10:
    st.warning("Maksimal 10 gambar per proses. Hanya 10 pertama diproses.")
    uploaded = uploaded[:10]

# ---------- utils ----------

def fit_inside(img: Image.Image, tw: int, th: int, method: str = "bicubic") -> Image.Image:
    """Resize proporsional agar MUAT di dalam (tw, th) tanpa distorsi."""
    filt = Image.BICUBIC if method == "bicubic" else Image.LANCZOS
    w, h = img.size
    ar_s, ar_t = w/h, tw/th
    if ar_s > ar_t:
        new_w = tw
        new_h = max(1, int(tw / ar_s))
    else:
        new_h = th
        new_w = max(1, int(th * ar_s))
    return img.resize((new_w, new_h), filt)

def mirror_pad_to_8k(pil_img: Image.Image, tw: int, th: int) -> Image.Image:
    """
    Pad ke ukuran target dengan REFLEKSI (tanpa crop, tanpa blur/solid).
    Menggunakan cv2.copyMakeBorder(BORDER_REFLECT_101) agar natural.
    """
    # ke BGR numpy
    arr = np.array(pil_img.convert("RGB"))[:, :, ::-1]  # RGB->BGR
    h, w = arr.shape[:2]
    pad_l = (tw - w) // 2
    pad_r = tw - w - pad_l
    pad_t = (th - h) // 2
    pad_b = th - h - pad_t
    # Jika tidak pas, lakukan refleksi
    if pad_l < 0 or pad_r < 0 or pad_t < 0 or pad_b < 0:
        raise ValueError("Ukuran sumber lebih besar dari target—pastikan sudah di-resize fit terlebih dahulu.")
    out = cv2.copyMakeBorder(arr, pad_t, pad_b, pad_l, pad_r, borderType=cv2.BORDER_REFLECT_101)
    # back to PIL
    return Image.fromarray(out[:, :, ::-1])

def gentle_sharpen(pil_img: Image.Image, sharp=0.8, micro_c=1.1) -> Image.Image:
    # Unsharp ringan + micro contrast kecil → natural, minim halo
    out = pil_img.filter(ImageFilter.UnsharpMask(radius=0.9, percent=120, threshold=2))
    if abs(sharp - 1.0) > 1e-3:
        out = ImageEnhance.Sharpness(out).enhance(sharp)
    if abs(micro_c - 1.0) > 1e-3:
        out = ImageEnhance.Contrast(out).enhance(micro_c)
    return out

def encode_jpeg_444(im: Image.Image, q: int) -> bytes:
    if im.mode != "RGB": im = im.convert("RGB")
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q, optimize=True, progressive=True, subsampling=0)  # 4:4:4
    return buf.getvalue()

def maximize_under_cap(im: Image.Image, cap_bytes: int):
    """
    Binary search kualitas agar ukuran mendekati cap (≤ 12 MB) dengan JPEG 4:4:4.
    Jika q=100 masih ≤ cap → pakai q=100 (maksimal).
    """
    hi = encode_jpeg_444(im, 100)
    if len(hi) <= cap_bytes:
        return hi, 100, "q=100 (maks)"

    lo_q, hi_q = Q_MIN, Q_MAX
    best = (encode_jpeg_444(im, lo_q), lo_q)
    while lo_q <= hi_q:
        mid = (lo_q + hi_q) // 2
        data = encode_jpeg_444(im, mid)
        if len(data) <= cap_bytes:
            best = (data, mid)
            lo_q = mid + 1
        else:
            hi_q = mid - 1
    return best[0], best[1], "tight fit"

# ---------- main ----------
if st.button("🚀 Proses ke 8K (tanpa crop & tanpa blur/solid)"):
    if not uploaded:
        st.warning("Unggah minimal satu gambar.")
    else:
        cap_bytes = int(MAX_MB * 1024 * 1024)
        progress = st.progress(0)
        for i, f in enumerate(uploaded, start=1):
            try:
                img = Image.open(f)
                # Normalisasi mode
                if img.mode == "RGBA":
                    bg = Image.new("RGB", img.size, (255,255,255))
                    bg.paste(img, mask=img.split()[-1])
                    img = bg
                elif img.mode not in ("RGB","L"):
                    img = img.convert("RGB")
                else:
                    img = img.convert("RGB")

                # 1) Resize MUAT di dalam 8K (tanpa distorsi)
                method = "bicubic" if "Bicubic" in resampler else "lanczos"
                fitted = fit_inside(img, TARGET_W, TARGET_H, method=method)

                # 2) Mirror padding agar jadi 8K tepat (tanpa crop, tanpa blur/solid)
                out = mirror_pad_to_8k(fitted, TARGET_W, TARGET_H)

                # 3) Penajaman natural
                out = gentle_sharpen(out, sharpen_amt, micro_contrast)

                # 4) Encode maksimal hingga ≤ 12 MB (JPEG 4:4:4)
                data, used_q, note = maximize_under_cap(out, cap_bytes)

                name = f"{Path(f.name).stem}_{suffix}.jpg"
                size_mb = len(data) / (1024 * 1024)
                st.image(out, caption=f"{name} — {out.width}×{out.height}px | q={used_q} | {size_mb:.2f} MB ({note})",
                         use_column_width=True)
                st.download_button(f"⬇️ Unduh {name}", data, file_name=name, mime="image/jpeg")

            except Exception as e:
                st.error(f"Gagal memproses {f.name}: {e}")
            progress.progress(i / len(uploaded))                
