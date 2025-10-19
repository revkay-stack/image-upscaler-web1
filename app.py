# app.py — 8K Proportional Upscaler (No Crop/Pad) + Indicators, ≤12 MB
import io
from pathlib import Path
import streamlit as st
from PIL import Image, ImageFilter, ImageFile, ImageEnhance

# Safety
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

SUPPORTED = ("png","jpg","jpeg","webp","bmp")
LONG_SIDE_8K = 7680   # target sisi terpanjang
TALL_SIDE_8K = 4320   # target tinggi untuk portrait
MAX_MB = 12           # batas ukuran file
Q_MIN, Q_MAX = 60, 100

st.set_page_config(page_title="8K Proportional Upscaler", page_icon="🖼️", layout="wide")
st.title("🖼️ 8K Proportional Upscaler")
st.caption("Upscale proporsional mengikuti rasio asli (tanpa crop/padding). Sisi terpanjang → 7680 px (atau tinggi → 4320 px untuk portrait). Hasil natural tajam, file ≤ 12 MB. Unduh per gambar.")

with st.sidebar:
    st.header("Pengaturan kualitas")
    resampler = st.radio("Resampler (upscale)", ["Bicubic (disarankan)", "Lanczos"], index=0)
    sharpen_amt = st.slider("Penajaman halus", 0.0, 2.0, 0.8, 0.1, help="0.6–1.0 biasanya pas; >1.2 berisiko halo.")
    micro_contrast = st.slider("Mikro-kontras", 1.0, 1.6, 1.1, 0.05)
    keep_if_bigger = st.toggle("Jika sumber > 8K, biarkan (jangan downscale)", value=True)
    suffix = st.text_input("Akhiran nama file", value="8K-prop")

uploaded = st.file_uploader("Pilih hingga 10 gambar", type=list(SUPPORTED), accept_multiple_files=True)
if uploaded and len(uploaded) > 10:
    st.warning("Maksimal 10 gambar per proses. Hanya 10 pertama diproses.")
    uploaded = uploaded[:10]

# ---------- utils ----------

def orientation_of(w: int, h: int) -> str:
    if w > h:
        return "Landscape"
    elif h > w:
        return "Portrait"
    else:
        return "Square"

def resize_longest_to_8k(img: Image.Image, method: str = "bicubic", keep_if_bigger: bool = True):
    """
    Return (resized_img, scale_factor, target_side_used)
    - Landscape/Square: width -> 7680
    - Portrait: height -> 4320
    - scale_factor = faktor pembesaran (>=1 untuk upscale, 1 jika tidak diubah)
    - target_side_used = "width" atau "height" (informasi indikator)
    """
    filt = Image.BICUBIC if method == "bicubic" else Image.LANCZOS
    w, h = img.size
    if w >= h:
        # landscape/square → target lebar 7680
        if keep_if_bigger and w >= LONG_SIDE_8K:
            return img, 1.0, "width"
        scale = LONG_SIDE_8K / w
        new_w = LONG_SIDE_8K
        new_h = max(1, int(h * scale))
        return img.resize((new_w, new_h), filt), scale, "width"
    else:
        # portrait → target tinggi 4320
        if keep_if_bigger and h >= TALL_SIDE_8K:
            return img, 1.0, "height"
        scale = TALL_SIDE_8K / h
        new_h = TALL_SIDE_8K
        new_w = max(1, int(w * scale))
        return img.resize((new_w, new_h), filt), scale, "height"

def gentle_sharpen(pil_img: Image.Image, sharp=0.8, micro_c=1.1) -> Image.Image:
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
    Binary search kualitas agar ukuran mendekati cap (≤ MAX_MB) dengan 4:4:4.
    Jika q=100 masih ≤ cap → pakai q=100 (maks).
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

# ---------- proses ----------
if st.button("🚀 Proses (8K proportional, tanpa crop/padding)"):
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

                # Info sumber
                src_w, src_h = img.size
                src_orient = orientation_of(src_w, src_h)

                # 1) Resize proporsional (tanpa crop/padding)
                method = "bicubic" if "Bicubic" in resampler else "lanczos"
                out, scale_factor, target_side_used = resize_longest_to_8k(img, method=method, keep_if_bigger=keep_if_bigger)

                # 2) Penajaman natural
                out = gentle_sharpen(out, sharpen_amt, micro_contrast)

                # 3) Encode maksimal hingga ≤ 12 MB (JPEG 4:4:4)
                data, used_q, note = maximize_under_cap(out, cap_bytes)

                # Indikator & caption
                new_w, new_h = out.size
                # Skala total relatif sisi terpanjang sumber → sisi terpanjang hasil
                if src_w >= src_h:
                    factor = new_w / src_w
                else:
                    factor = new_h / src_h
                factor_str = f"{factor:.2f}×" if factor >= 0 else "1.00×"
                aspect_str = f"{new_w}:{new_h}"

                name = f"{Path(f.name).stem}_{suffix}.jpg"
                size_mb = len(data) / (1024 * 1024)
                caption = (
                    f"{name} — {new_w}×{new_h}px | {src_orient} | upscale {factor_str} | q={used_q} "
                    f"| {size_mb:.2f} MB ({note}) | ratio {aspect_str}"
                )

                st.image(out, caption=caption, use_column_width=True)
                st.download_button(f"⬇️ Unduh {name}", data, file_name=name, mime="image/jpeg")

            except Exception as e:
                st.error(f"Gagal memproses {f.name}: {e}")
            progress.progress(i / len(uploaded))
