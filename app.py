# app.py — 8× Proportional Upscaler (Cloud-Optimized) + Face Glow + Before/After
import io, math, gc
from pathlib import Path
import streamlit as st
from PIL import Image, ImageFilter, ImageFile, ImageEnhance
import numpy as np
import cv2

# Safety
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

SUPPORTED = ("png","jpg","jpeg","webp","bmp")
MAX_MB = 12                      # batas ukuran file
Q_MIN, Q_MAX = 60, 100           # kualitas JPEG 4:4:4
REQ_FACTOR = 8.0                 # faktor target
DPI_EXIF = (400, 400)            # metadata DPI 400

st.set_page_config(page_title="Upscaler 8× — Optimized", page_icon="🖼️", layout="wide")
st.title("🖼️ Proportional Upscaler — 8× (Optimized for Streamlit Cloud)")
st.caption("Upscale proporsional 8× (tanpa crop/padding), Face Glow otomatis, hasil ≤ 12 MB (JPEG 4:4:4, DPI 400). Mode ringan agar stabil di server gratis.")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Mode & Preset")
    lite_mode = st.toggle("Mode Ringan (disarankan)", value=True, help="Batasi ukuran output agar aman dari crash.")
    max_out_mp = st.slider("Batas output (MegaPixel)", 60, 140, 80, 5, help="Aktif jika Mode Ringan ON.")
    preset = st.selectbox("Preset", ["Default Natural", "Anti-Halo", "Tekstur Kain", "Kustom"], index=0)
    keep_if_bigger = st.toggle("Jika sumber sudah besar, jangan perkecil", value=True)
    suffix = st.text_input("Akhiran nama file", value="x8")

    st.markdown("---")
    st.subheader("Face Glow")
    enable_face_glow = st.toggle("Aktifkan Face Glow", value=True)
    glow_strength = st.slider("Intensitas Glow", 0, 100, 40, 5)
    face_pad = st.slider("Padding area wajah", 0.9, 1.6, 1.2, 0.05)

    st.markdown("---")
    st.subheader("Tweak (untuk Kustom)")
    resampler_ui = st.radio("Resampler", ["Bicubic", "Lanczos"], index=0, disabled=(preset!="Kustom"))
    sharpen_amt_ui = st.slider("Sharpness", 0.0, 2.0, 0.8, 0.1, disabled=(preset!="Kustom"))
    micro_contrast_ui = st.slider("Mikro-kontras", 1.0, 1.6, 1.1, 0.05, disabled=(preset!="Kustom"))
    usm_radius_ui = st.slider("USM Radius", 0.3, 2.0, 0.9, 0.1, disabled=(preset!="Kustom"))
    usm_percent_ui = st.slider("USM Percent", 50, 250, 120, 5, disabled=(preset!="Kustom"))
    usm_thresh_ui = st.slider("USM Threshold", 0, 10, 2, 1, disabled=(preset!="Kustom"))

uploaded = st.file_uploader("Pilih hingga 5 gambar (disarankan ≤10MB/berkas)", type=list(SUPPORTED), accept_multiple_files=True)
if uploaded and len(uploaded) > 5:
    st.warning("Maksimal 5 gambar per proses pada Mode Ringan. Hanya 5 pertama yang diproses.")
    uploaded = uploaded[:5]

# ---------------- Preset ----------------
def resolve_params(name: str):
    if name == "Default Natural":
        return dict(resampler="bicubic", sharpen_amt=0.8, micro_contrast=1.10, usm_radius=0.9, usm_percent=120, usm_thresh=2)
    if name == "Anti-Halo":
        return dict(resampler="bicubic", sharpen_amt=0.6, micro_contrast=1.05, usm_radius=0.7, usm_percent=90, usm_thresh=3)
    if name == "Tekstur Kain":
        return dict(resampler="lanczos", sharpen_amt=1.05, micro_contrast=1.18, usm_radius=1.0, usm_percent=140, usm_thresh=2)
    return dict(  # Kustom
        resampler=("bicubic" if resampler_ui == "Bicubic" else "lanczos"),
        sharpen_amt=sharpen_amt_ui,
        micro_contrast=micro_contrast_ui,
        usm_radius=usm_radius_ui,
        usm_percent=usm_percent_ui,
        usm_thresh=usm_thresh_ui,
    )

# ---------------- Cache resource (detector) ----------------
@st.cache_resource
def load_face_detector():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

FACE_CASCADE = load_face_detector()

# ---------------- Utils ringan ----------------
def orientation_of(w, h):
    if w > h: return "Landscape"
    if h > w: return "Portrait"
    return "Square"

def resample_filter(name: str):
    return Image.BICUBIC if name == "bicubic" else Image.LANCZOS

def capped_factor_for_cloud(src_w, src_h, req_factor: float, max_mp: int | None):
    if not max_mp:  # None berarti tidak dibatasi
        return req_factor
    src_px = src_w * src_h
    max_factor = math.sqrt((max_mp * 1_000_000) / max(1, src_px))
    return max(1.0, min(req_factor, max_factor))

def upscale_stepwise(img: Image.Image, factor: float, filt, step=1.8):
    """Upscale bertahap untuk stabilitas & kualitas."""
    if factor <= 1.0:
        return img
    w, h = img.size
    cur = img
    target_w, target_h = int(w * factor), int(h * factor)
    while max(cur.width, cur.height) < max(target_w, target_h):
        scale = min(step, max(target_w/cur.width, target_h/cur.height))
        nw = min(target_w, int(cur.width * scale))
        nh = min(target_h, int(cur.height * scale))
        if nw == cur.width and nh == cur.height: break
        cur = cur.resize((nw, nh), filt)
    if (cur.width, cur.height) != (target_w, target_h):
        cur = cur.resize((target_w, target_h), filt)
    return cur

def gentle_sharpen(img: Image.Image, p):
    out = img.filter(ImageFilter.UnsharpMask(radius=p["usm_radius"], percent=p["usm_percent"], threshold=p["usm_thresh"]))
    if abs(p["sharpen_amt"] - 1.0) > 1e-3:
        out = ImageEnhance.Sharpness(out).enhance(p["sharpen_amt"])
    if abs(p["micro_contrast"] - 1.0) > 1e-3:
        out = ImageEnhance.Contrast(out).enhance(p["micro_contrast"])
    return out

def detect_faces_fast(pil_img: Image.Image, max_side=1024):
    """Deteksi wajah di skala kecil lalu mapping ke ukuran asli (hemat RAM/CPU)."""
    rgb = np.array(pil_img.convert("RGB"))
    h, w = rgb.shape[:2]
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        small = cv2.resize(rgb, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    else:
        small = rgb
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24,24))
    faces = [(int(x/scale), int(y/scale), int(wd/scale), int(hd/scale)) for (x,y,wd,hd) in faces]
    return faces

def apply_face_glow(img: Image.Image, faces, strength=40, pad_scale=1.2):
    if len(faces) == 0 or strength <= 0:
        return img
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]
    base = arr.astype(np.float32) / 255.0

    mask = np.zeros((h, w), dtype=np.float32)
    for (x, y, fw, fh) in faces:
        cx, cy = x + fw/2, y + fh/2
        ew, eh = fw * pad_scale, fh * pad_scale
        axes = (max(1,int(ew/2)), max(1,int(eh/2)))
        center = (int(cx), int(cy))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)

    # feather ringan-adaptif
    k = max(21, ((w+h)//180)*2 + 1)
    mask = cv2.GaussianBlur(mask, (k, k), sigmaX=k/6)

    # brighten L channel
    lab = cv2.cvtColor((base*255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)
    gain = 1.0 + (strength/100.0)*0.32
    L = np.clip(L * gain, 0, 255)
    lab2 = cv2.merge((L, A, B)).astype(np.uint8)
    glow = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB).astype(np.float32)/255.0

    # smoothing lembut
    glow_bgr = cv2.cvtColor((glow*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    glow_bgr = cv2.bilateralFilter(glow_bgr, d=5, sigmaColor=15+int(strength*0.5), sigmaSpace=15+int(strength*0.5))
    glow = cv2.cvtColor(glow_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0

    out = base*(1.0 - mask[...,None]) + glow*mask[...,None]
    return Image.fromarray(np.clip(out*255, 0, 255).astype(np.uint8))

def encode_jpeg_444_dpi(im: Image.Image, q: int) -> bytes:
    if im.mode != "RGB": im = im.convert("RGB")
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q, optimize=True, progressive=True, subsampling=0, dpi=DPI_EXIF)
    return buf.getvalue()

def maximize_under_cap(im: Image.Image, cap_bytes: int):
    hi = encode_jpeg_444_dpi(im, 100)
    if len(hi) <= cap_bytes:
        return hi, 100, "q=100"
    lo, hiq = Q_MIN, Q_MAX
    best = (encode_jpeg_444_dpi(im, lo), lo)
    while lo <= hiq:
        mid = (lo + hiq)//2
        data = encode_jpeg_444_dpi(im, mid)
        if len(data) <= cap_bytes:
            best = (data, mid); lo = mid + 1
        else:
            hiq = mid - 1
    return best[0], best[1], "tight fit"

# ---------------- Process ----------------
if st.button("🚀 Proses 8× (Optimized)"):
    if not uploaded:
        st.warning("Unggah minimal satu gambar.")
    else:
        params = resolve_params(preset)
        cap_bytes = int(MAX_MB * 1024 * 1024)
        progress = st.progress(0)

        for i, f in enumerate(uploaded, start=1):
            try:
                img = Image.open(f)
                # Normalisasi ke RGB
                if img.mode == "RGBA":
                    bg = Image.new("RGB", img.size, (255,255,255))
                    bg.paste(img, mask=img.split()[-1]); img = bg
                elif img.mode not in ("RGB","L"):
                    img = img.convert("RGB")
                else:
                    img = img.convert("RGB")

                src_w, src_h = img.size
                src_orient = orientation_of(src_w, src_h)

                # --- 1) Hitung faktor efektif aman ---
                max_mp = max_out_mp if lite_mode else None
                eff_factor = capped_factor_for_cloud(src_w, src_h, REQ_FACTOR, max_mp)
                filt = resample_filter(params["resampler"])

                # Jika keep_if_bigger dan hasil tidak lebih besar → pakai asli
                target_w, target_h = int(src_w * eff_factor), int(src_h * eff_factor)
                if keep_if_bigger and (target_w <= src_w or target_h <= src_h):
                    out = img.copy()
                    eff_factor = 1.0
                    was_capped = False
                else:
                    # --- 2) Upscale bertahap (hemat memori) ---
                    out = upscale_stepwise(img, eff_factor, filt)
                    was_capped = eff_factor < REQ_FACTOR

                # --- 3) Sharpen natural ---
                out = gentle_sharpen(out, params)

                # --- 4) Face Glow (deteksi cepat skala kecil) ---
                if enable_face_glow:
                    faces = detect_faces_fast(out, max_side=1024 if lite_mode else 1600)
                    out = apply_face_glow(out, faces, strength=glow_strength, pad_scale=face_pad)

                # --- 5) Encode ≤ 12 MB (JPEG 4:4:4, DPI 400) ---
                data, used_q, note = maximize_under_cap(out, cap_bytes)

                # --- 6) Preview & download ---
                col1, col2 = st.columns(2, gap="large")
                with col1:
                    st.subheader("Sebelum")
                    st.image(img, caption=f"{f.name} — {src_w}×{src_h}px | {src_orient}", use_column_width=True)
                with col2:
                    st.subheader("Sesudah (8×)")
                    new_w, new_h = out.size
                    size_mb = len(data) / (1024 * 1024)
                    captext = (
                        f"{Path(f.name).stem}_{suffix}.jpg — {new_w}×{new_h}px | {src_orient} | "
                        f"req 8.0× → eff {eff_factor:.2f}×" + (" (capped)" if was_capped else "") +
                        (f" | FaceGlow on" if enable_face_glow else " | FaceGlow off") +
                        f" | preset: {preset} | q={used_q} | {size_mb:.2f} MB ({note}) | DPI 400"
                    )
                    st.image(out, caption=captext, use_column_width=True)
                    st.download_button(
                        "⬇️ Unduh hasil",
                        data,
                        file_name=f"{Path(f.name).stem}_{suffix}.jpg",
                        mime="image/jpeg"
                    )

                # Bebaskan memori setiap iterasi
                del out; gc.collect()

            except MemoryError:
                st.error(f"Gagal memproses {f.name}: kehabisan memori. Coba ulang dengan Mode Ringan ON / kurangi jumlah gambar.")
            except Exception as e:
                st.error(f"Gagal memproses {f.name}: {e}")

            progress.progress(i / len(uploaded))
