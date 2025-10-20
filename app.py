# app.py — Proportional 8× Upscaler + Face Glow + Preview, ≤12 MB, DPI 400
import io, math
from pathlib import Path
import streamlit as st
from PIL import Image, ImageFilter, ImageFile, ImageEnhance
import numpy as np
import cv2

# Safety
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

SUPPORTED = ("png","jpg","jpeg","webp","bmp")
MAX_MB = 12                        # ukuran file maksimum
Q_MIN, Q_MAX = 60, 100             # rentang kualitas JPEG (4:4:4)
MAX_OUT_PIXELS = 120_000_000       # pagar aman Cloud (~120 MP)
REQ_FACTOR = 8.0                   # faktor upscale tetap 8×

st.set_page_config(page_title="Proportional Upscaler 8× — Face Glow", page_icon="🖼️", layout="wide")
st.title("🖼️ Proportional Upscaler — 8× (Face Glow + Preview)")
st.caption("Upscale proporsional 8× tanpa crop/padding. Otomatis deteksi wajah dan memberi efek glow alami. Simpan JPEG 4:4:4 ≤ 12 MB, DPI 400. Unduh per gambar.")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Pengaturan")
    preset = st.selectbox("Preset kualitas", ["Default Natural", "Anti-Halo", "Tekstur Kain", "Kustom"], index=0)
    keep_if_bigger = st.toggle("Jika sumber sudah besar, jangan perkecil (keep)", value=True)
    suffix = st.text_input("Akhiran nama file", value="x8")

    st.markdown("---")
    st.subheader("Face Glow")
    enable_face_glow = st.toggle("Aktifkan Face Glow (auto)", value=True)
    glow_strength = st.slider("Intensitas Glow", 0, 100, 40, 5, help="Semakin besar semakin cerah/halus pada area wajah.")
    face_pad = st.slider("Lebar area wajah (padding)", 0.8, 1.6, 1.2, 0.05, help="Perbesar area efek di sekitar wajah.")

    st.markdown("---")
    st.subheader("Tweak lanjutan" + (" (aktif karena Kustom)" if preset == "Kustom" else " (opsional)"))
    resampler_ui = st.radio("Resampler", ["Bicubic", "Lanczos"], index=0 if preset != "Tekstur Kain" else 1, disabled=(preset!="Kustom"))
    sharpen_amt_ui = st.slider("Penajaman (Sharpness)", 0.0, 2.0, 0.8, 0.1, disabled=(preset!="Kustom"))
    micro_contrast_ui = st.slider("Mikro-kontras", 1.0, 1.6, 1.1, 0.05, disabled=(preset!="Kustom"))
    usm_radius_ui = st.slider("USM Radius", 0.3, 2.0, 0.9, 0.1, disabled=(preset!="Kustom"))
    usm_percent_ui = st.slider("USM Percent", 50, 250, 120, 5, disabled=(preset!="Kustom"))
    usm_thresh_ui = st.slider("USM Threshold", 0, 10, 2, 1, disabled=(preset!="Kustom"))

uploaded = st.file_uploader("Pilih hingga 10 gambar", type=list(SUPPORTED), accept_multiple_files=True)
if uploaded and len(uploaded) > 10:
    st.warning("Maksimal 10 gambar per proses. Hanya 10 pertama diproses.")
    uploaded = uploaded[:10]

# ---------- Preset ----------
def resolve_params(preset_name: str):
    if preset_name == "Default Natural":
        return dict(resampler="bicubic", sharpen_amt=0.8, micro_contrast=1.10, usm_radius=0.9, usm_percent=120, usm_thresh=2)
    if preset_name == "Anti-Halo":
        return dict(resampler="bicubic", sharpen_amt=0.6, micro_contrast=1.05, usm_radius=0.7, usm_percent=90, usm_thresh=3)
    if preset_name == "Tekstur Kain":
        return dict(resampler="lanczos", sharpen_amt=1.1, micro_contrast=1.20, usm_radius=1.0, usm_percent=140, usm_thresh=2)
    # Kustom
    return dict(
        resampler=("bicubic" if resampler_ui == "Bicubic" else "lanczos"),
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

# --- Face detection & glow ---
_haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_CASCADE = cv2.CascadeClassifier(_haar_path)

def detect_faces(pil_img: Image.Image):
    """Return list of (x,y,w,h) in upscaled RGB image."""
    rgb = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # parameters tuned for large images
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
    return faces

def apply_face_glow(pil_img: Image.Image, faces, strength:int=40, pad_scale:float=1.2) -> Image.Image:
    """
    Glow natural pada wajah:
    - LAB: boost L-channel (kecerahan) + sedikit smoothing bilateral
    - Feathered mask agar halus di tepi
    strength: 0..100 → di-mapping ke gain & softness
    pad_scale: perbesar area efek di sekitar bbox wajah
    """
    if len(faces) == 0 or strength <= 0:
        return pil_img

    img = np.array(pil_img.convert("RGB"))
    h, w = img.shape[:2]
    base = img.astype(np.float32) / 255.0

    # mask kosong
    mask = np.zeros((h, w), dtype=np.float32)

    for (x, y, fw, fh) in faces:
        # perluas area
        cx, cy = x + fw/2, y + fh/2
        ew, eh = fw * pad_scale, fh * pad_scale
        # ellipse mask
        axes = (int(ew/2), int(eh/2))
        center = (int(cx), int(cy))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, thickness=-1)

    # feathering
    blur_ksize = max(31, int((w+h)/200)*2+1)  # adaptif
    mask = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), sigmaX=blur_ksize/6)

    # buat versi "glow" dari gambar: brighten + soft smooth + slight clarity
    lab = cv2.cvtColor((base*255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)
    # mapping strength -> gain L (1.0..1.35) & bilateral strength
    gain = 1.0 + (strength/100.0)*0.35
    L = np.clip(L * gain, 0, 255)
    lab_glow = cv2.merge((L, A, B)).astype(np.uint8)
    glow = cv2.cvtColor(lab_glow, cv2.COLOR_LAB2RGB).astype(np.float32)/255.0

    # softening kecil untuk kulit (bilateral ringan)
    d = 5
    sc = 20 + int(strength*0.6)
    ss = 20 + int(strength*0.6)
    glow_bgr = cv2.cvtColor((glow*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    glow_bgr = cv2.bilateralFilter(glow_bgr, d=d, sigmaColor=sc, sigmaSpace=ss)
    glow = cv2.cvtColor(glow_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0

    # blend berdasarkan mask (per-channel)
    mask3 = np.dstack([mask, mask, mask])
    out = base*(1.0 - mask3) + glow*mask3
    out = np.clip(out, 0, 1)

    return Image.fromarray((out*255).astype(np.uint8))

def encode_jpeg_444_dpi(im: Image.Image, q: int, dpi_val=(400,400)) -> bytes:
    """JPEG 4:4:4 dengan DPI 400 untuk metadata kerapatan piksel."""
    if im.mode != "RGB": im = im.convert("RGB")
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q, optimize=True, progressive=True, subsampling=0, dpi=dpi_val)
    return buf.getvalue()

def maximize_under_cap(im: Image.Image, cap_bytes: int):
    """Binary search kualitas agar ukuran mendekati cap (≤ MAX_MB) + DPI 400."""
    hi = encode_jpeg_444_dpi(im, 100)
    if len(hi) <= cap_bytes:
        return hi, 100, "q=100 (maks)"
    lo_q, hi_q = Q_MIN, Q_MAX
    best = (encode_jpeg_444_dpi(im, lo_q), lo_q)
    while lo_q <= hi_q:
        mid = (lo_q + hi_q) // 2
        data = encode_jpeg_444_dpi(im, mid)
        if len(data) <= cap_bytes:
            best = (data, mid); lo_q = mid + 1
        else:
            hi_q = mid - 1
    return best[0], best[1], "tight fit"

# ---------- Process ----------
if st.button("🚀 Proses 8× (Face Glow + Preview)"):
    if not uploaded:
        st.warning("Unggah minimal satu gambar.")
    else:
        cap_bytes = int(MAX_MB * 1024 * 1024)
        params = resolve_params(preset)

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

                # 1) Upscale 8× (aman Cloud)
                out, eff_factor, was_capped = resize_with_factor(img, REQ_FACTOR, method=params["resampler"])
                # Jika sumber lebih besar & keep_if_bigger aktif → pakai asli (hindari downscale)
                if keep_if_bigger and (out.width <= src_w or out.height <= src_h):
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

                # 3) Face Glow (opsional)
                if enable_face_glow:
                    faces = detect_faces(out)
                    out = apply_face_glow(out, faces, strength=glow_strength, pad_scale=face_pad)

                # 4) Encode maksimal hingga ≤ 12 MB (JPEG 4:4:4, DPI 400)
                data, used_q, note = maximize_under_cap(out, cap_bytes)

                # 5) Preview Before / After + Download
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
                        (f" | FaceGlow on ({len(detect_faces(out))} wajah)" if enable_face_glow else " | FaceGlow off") +
                        f" | preset: {preset} | q={used_q} | {size_mb:.2f} MB ({note}) | DPI 400"
                    )
                    st.image(out, caption=captext, use_column_width=True)
                    st.download_button(
                        "⬇️ Unduh hasil",
                        data,
                        file_name=f"{Path(f.name).stem}_{suffix}.jpg",
                        mime="image/jpeg"
                    )

            except Exception as e:
                st.error(f"Gagal memproses {f.name}: {e}")
            progress.progress(i / len(uploaded))
