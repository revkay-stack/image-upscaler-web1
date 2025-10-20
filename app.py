# app.py — Proportional Upscaler 4×/8× + Presets (Natural / Anti-Halo / Tekstur Kain)
import io, math
from pathlib import Path
import streamlit as st
from PIL import Image, ImageFilter, ImageFile, ImageEnhance

# Safety
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

SUPPORTED = ("png","jpg","jpeg","webp","bmp")
MAX_MB = 12                      # batas ukuran file
Q_MIN, Q_MAX = 60, 100           # range kualitas JPEG (4:4:4)
MAX_OUT_PIXELS = 120_000_000     # pagar aman Cloud (~120 MP)

st.set_page_config(page_title="Proportional Upscaler 4× / 8× — Presets", page_icon="🖼️", layout="wide")
st.title("🖼️ Proportional Upscaler — 4× / 8× (dengan Preset)")
st.caption("Upscale proporsional 4×/8× tanpa crop/padding. Preset cepat: Natural, Anti-Halo, Tekstur Kain. File dimaksimalkan hingga ≤ 12 MB per gambar (JPEG 4:4:4).")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Pengaturan Utama")
    factor_choice = st.radio("Faktor upscale", ["4×", "8×"], index=0)
    preset = st.selectbox("Preset kualitas", ["Default Natural", "Anti-Halo", "Tekstur Kain", "Kustom"], index=0)
    keep_if_bigger = st.toggle("Jika sumber > ukuran target, jangan perkecil (keep)", value=True)
    suffix = st.text_input("Akhiran nama file", value="upscaled")

    # Parameter kustom / pengganti preset
    show_advanced = (preset == "Kustom")
    st.markdown("---")
    st.subheader("Tweak lanjutan" + (" (aktif karena Kustom)" if show_advanced else " (opsional)"))
    resampler_ui = st.radio("Resampler", ["Bicubic", "Lanczos"], index=0 if preset != "Tekstur Kain" else 1, disabled=not show_advanced)
    sharpen_amt_ui = st.slider("Penajaman (Sharpness)", 0.0, 2.0, 0.8, 0.1, disabled=not show_advanced)
    micro_contrast_ui = st.slider("Mikro-kontras", 1.0, 1.6, 1.1, 0.05, disabled=not show_advanced)
    usm_radius_ui = st.slider("USM Radius", 0.3, 2.0, 0.9, 0.1, disabled=not show_advanced)
    usm_percent_ui = st.slider("USM Percent", 50, 250, 120, 5, disabled=not show_advanced)
    usm_thresh_ui = st.slider("USM Threshold", 0, 10, 2, 1, disabled=not show_advanced)

uploaded = st.file_uploader("Pilih hingga 10 gambar", type=list(SUPPORTED), accept_multiple_files=True)
if uploaded and len(uploaded) > 10:
    st.warning("Maksimal 10 gambar per proses. Hanya 10 pertama diproses.")
    uploaded = uploaded[:10]

# ---------- Preset definitions ----------
def resolve_params(preset_name: str):
    """
    Kembalikan dict parameter:
    - resampler: 'bicubic' | 'lanczos'
    - sharpen_amt: 0.0..2.0  (ImageEnhance.Sharpness)
    - micro_contrast: 1.0..1.6 (ImageEnhance.Contrast)
    - usm_radius, usm_percent, usm_thresh: UnsharpMask
    """
    if preset_name == "Default Natural":
        return dict(resampler="bicubic", sharpen_amt=0.8, micro_contrast=1.10, usm_radius=0.9, usm_percent=120, usm_thresh=2)
    if preset_name == "Anti-Halo":
        # Lebih halus di tepi, minim ringing
        return dict(resampler="bicubic", sharpen_amt=0.6, micro_contrast=1.05, usm_radius=0.7, usm_percent=90, usm_thresh=3)
    if preset_name == "Tekstur Kain":
        # Tekankan detail serat/tekstur ringan (hati-hati halo)
        return dict(resampler="lanczos", sharpen_amt=1.1, micro_contrast=1.20, usm_radius=1.0, usm_percent=140, usm_thresh=2)
    # Kustom: ambil dari UI
    return dict(
        resampler="bicubic" if resampler_ui == "Bicubic" else "lanczos",
        sharpen_amt=sharpen_amt_ui,
        micro_contrast=micro_contrast_ui,
        usm_radius=usm_radius_ui,
        usm_percent=usm_percent_ui,
        usm_thresh=usm_thresh_ui,
    )

# ---------- Utils ----------
def orientation_of(w, h):
    if w > h: return "Landscape"
    if h > w: return "Portrait"
    return "Square"

def resize_with_factor(img: Image.Image, req_factor: float, method: str = "bicubic"):
    """Upscale proporsional; jika melebihi MAX_OUT_PIXELS, turunkan faktor efektif."""
    filt = Image.BICUBIC if method == "bicubic" else Image.LANCZOS
    w, h = img.size
    src_px = w * h
    max_factor = math.sqrt(MAX_OUT_PIXELS / max(1, src_px))
    eff = min(req_factor, max(1.0, max_factor))
    new_w = max(1, int(w * eff))
    new_h = max(1, int(h * eff))
    return img.resize((new_w, new_h), filt), eff, (eff < req_factor)

def gentle_sharpen(pil_img: Image.Image, sharpen_amt: float, micro_c: float, usm_radius: float, usm_percent: int, usm_thresh: int):
    out = pil_img.filter(ImageFilter.UnsharpMask(radius=usm_radius, percent=usm_percent, threshold=usm_thresh))
    if abs(sharpen_amt - 1.0) > 1e-3:
        out = ImageEnhance.Sharpness(out).enhance(sharpen_amt)
    if abs(micro_c - 1.0) > 1e-3:
        out = ImageEnhance.Contrast(out).enhance(micro_c)
    return out

def encode_jpeg_444(im: Image.Image, q: int) -> bytes:
    if im.mode != "RGB": im = im.convert("RGB")
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q, optimize=True, progressive=True, subsampling=0)  # 4:4:4
    return buf.getvalue()

def maximize_under_cap(im: Image.Image, cap_bytes: int):
    """Binary search kualitas agar ukuran mendekati cap (≤ MAX_MB)."""
    hi = encode_jpeg_444(im, 100)
    if len(hi) <= cap_bytes:
        return hi, 100, "q=100 (maks)"
    lo_q, hi_q = Q_MIN, Q_MAX
    best = (encode_jpeg_444(im, lo_q), lo_q)
    while lo_q <= hi_q:
        mid = (lo_q + hi_q) // 2
        data = encode_jpeg_444(im, mid)
        if len(data) <= cap_bytes:
            best = (data, mid); lo_q = mid + 1
        else:
            hi_q = mid - 1
    return best[0], best[1], "tight fit"

# ---------- Process ----------
if st.button("🚀 Proses (4× / 8× + preset)"):
    if not uploaded:
        st.warning("Unggah minimal satu gambar.")
    else:
        cap_bytes = int(MAX_MB * 1024 * 1024)
        params = resolve_params(preset)
        req_factor = 4.0 if factor_choice.startswith("4") else 8.0

        progress = st.progress(0)
        for i, f in enumerate(uploaded, start=1):
            try:
                img = Image.open(f)
                # Normalisasi mode ke RGB
                if img.mode == "RGBA":
                    bg = Image.new("RGB", img.size, (255,255,255))
                    bg.paste(img, mask=img.split()[-1]); img = bg
                elif img.mode not in ("RGB","L"):
                    img = img.convert("RGB")
                else:
                    img = img.convert("RGB")

                src_w, src_h = img.size
                src_orient = orientation_of(src_w, src_h)

                # 1) Upscale dengan faktor (aman Cloud)
                method = params["resampler"]
                out, eff_factor, was_capped = resize_with_factor(img, req_factor, method=method)

                # Jika sumber lebih besar & keep_if_bigger aktif, bisa jadi eff_factor=1.0
                if keep_if_bigger and ((src_w >= out.width) or (src_h >= out.height)):
                    out = img
                    eff_factor = 1.0
                    was_capped = False

                # 2) Penajaman sesuai preset
                out = gentle_sharpen(
                    out,
                    params["sharpen_amt"],
                    params["micro_contrast"],
                    params["usm_radius"],
                    params["usm_percent"],
                    params["usm_thresh"],
                )

                # 3) Encode maksimal hingga ≤ 12 MB (JPEG 4:4:4)
                data, used_q, note = maximize_under_cap(out, cap_bytes)

                # 4) Caption & download
                new_w, new_h = out.size
                size_mb = len(data) / (1024 * 1024)
                cap = (
                    f"{Path(f.name).stem}_{suffix}.jpg — {new_w}×{new_h}px | {src_orient} | "
                    f"req {req_factor:.1f}× → eff {eff_factor:.2f}×"
                    + (" (capped)" if was_capped else "")
                    + f" | preset: {preset} | q={used_q} | {size_mb:.2f} MB ({note})"
                )
                st.image(out, caption=cap, use_column_width=True)
                st.download_button(
                    f"⬇️ Unduh {Path(f.name).stem}_{suffix}.jpg",
                    data,
                    file_name=f"{Path(f.name).stem}_{suffix}.jpg",
                    mime="image/jpeg"
                )

            except Exception as e:
                st.error(f"Gagal memproses {f.name}: {e}")
            progress.progress(i / len(uploaded))
