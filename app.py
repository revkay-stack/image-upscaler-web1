# app.py — Auto 8× Upscaler (Optimized) + Face Glow + Matte Skin + Face Count + ZIP
import io, math, gc, zipfile
from pathlib import Path
import streamlit as st
from PIL import Image, ImageFilter, ImageFile, ImageEnhance
import numpy as np

# OpenCV optional (graceful fallback)
try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

# Safety
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

SUPPORTED = ("png","jpg","jpeg","webp","bmp")
MAX_MB = 12
Q_MIN, Q_MAX = 60, 100
REQ_FACTOR = 8.0
DPI_EXIF = (400, 400)

st.set_page_config(page_title="Auto 8× Upscaler — Optimized", page_icon="🖼️", layout="wide")
st.title("🖼️ Auto Intelligent Proportional Upscaler — 8×")
st.caption("Upscale 8× proporsional (tanpa crop/padding). Auto preset & Face Glow. Matte Skin opsional. Hasil ≤ 12 MB (JPEG 4:4:4, DPI 400). Preview & tombol **Unduh semua**.")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Mode")
    mode = st.radio("Pilih mode", ["Auto (disarankan)", "Manual"], index=0)

    st.subheader("Kinerja")
    lite_mode = st.toggle("Mode Ringan (aman Cloud)", value=True)
    max_out_mp = st.slider("Batas output (MegaPixel)", 60, 140, 80, 5, help="Dipakai jika Mode Ringan ON.")
    keep_if_bigger = st.toggle("Jika sumber sudah besar, jangan perkecil", value=True)
    suffix = st.text_input("Akhiran nama file", value="x8")

    st.markdown("---")
    st.subheader("Face & Skin")
    if mode == "Manual":
        enable_face_glow = st.toggle("Face Glow", value=True and HAS_CV2, disabled=not HAS_CV2)
        glow_strength = st.slider("Intensitas Glow", 0, 100, 40, 5, disabled=not HAS_CV2)
        face_pad = st.slider("Padding area wajah", 0.9, 1.6, 1.2, 0.05, disabled=not HAS_CV2)
    else:
        st.caption("Auto: Face Glow menyala bila wajah terdeteksi, intensitas adaptif.")
        enable_face_glow = True; glow_strength = None; face_pad = None

    matte_on = st.toggle("Matte Skin (halus natural di wajah)", value=False, help="Menghaluskan kulit wajah tanpa ‘plastik’.")
    matte_strength = st.slider("Kekuatan Matte Skin", 0, 100, 30, 5, disabled=not matte_on)
    matte_pad = st.slider("Padding Matte Skin", 0.9, 1.6, 1.15, 0.05, disabled=not matte_on)

    st.markdown("---")
    st.subheader("Preset / Tweak")
    if mode == "Manual":
        preset = st.selectbox("Preset", ["Default Natural", "Anti-Halo", "Tekstur Kain", "Kustom"], index=0)
        resampler_ui = st.radio("Resampler", ["Bicubic", "Lanczos"], index=0, disabled=(preset!="Kustom"))
        sharpen_amt_ui = st.slider("Sharpness", 0.0, 2.0, 0.8, 0.1, disabled=(preset!="Kustom"))
        micro_contrast_ui = st.slider("Mikro-kontras", 1.0, 1.6, 1.1, 0.05, disabled=(preset!="Kustom"))
        usm_radius_ui = st.slider("USM Radius", 0.3, 2.0, 0.9, 0.1, disabled=(preset!="Kustom"))
        usm_percent_ui = st.slider("USM Percent", 50, 250, 120, 5, disabled=(preset!="Kustom"))
        usm_thresh_ui = st.slider("USM Threshold", 0, 10, 2, 1, disabled=(preset!="Kustom"))
    else:
        preset = "AUTO"

uploaded = st.file_uploader("Pilih hingga 5 gambar (disarankan ≤10MB/berkas)", type=list(SUPPORTED), accept_multiple_files=True)
if uploaded and len(uploaded) > 5:
    st.warning("Maksimal 5 gambar per proses pada Mode Ringan. Hanya 5 pertama yang diproses.")
    uploaded = uploaded[:5]

# ---------------- Utils ----------------
def orientation_of(w, h):
    if w > h: return "Landscape"
    if h > w: return "Portrait"
    return "Square"

def resample_filter(name: str):
    return Image.BICUBIC if name == "bicubic" else Image.LANCZOS

def capped_factor_for_cloud(src_w, src_h, req_factor: float, max_mp: int | None):
    if not max_mp:
        return req_factor
    src_px = src_w * src_h
    max_factor = math.sqrt((max_mp * 1_000_000) / max(1, src_px))
    return max(1.0, min(req_factor, max_factor))

def upscale_stepwise(img: Image.Image, factor: float, filt, step=1.8):
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

def gentle_sharpen(img: Image.Image, sharpen_amt: float, micro_c: float, usm_radius: float, usm_percent: int, usm_thresh: int):
    out = img.filter(ImageFilter.UnsharpMask(radius=usm_radius, percent=usm_percent, threshold=usm_thresh))
    if abs(sharpen_amt - 1.0) > 1e-3:
        out = ImageEnhance.Sharpness(out).enhance(sharpen_amt)
    if abs(micro_c - 1.0) > 1e-3:
        out = ImageEnhance.Contrast(out).enhance(micro_c)
    return out

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

# ---- Face detection & masks ----
if HAS_CV2:
    @st.cache_resource
    def load_cascade():
        return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    FACE_CASCADE = load_cascade()

def detect_faces_fast(pil_img: Image.Image, max_side=1024):
    if not HAS_CV2:
        return []
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
    return [(int(x/scale), int(y/scale), int(wd/scale), int(hd/scale)) for (x,y,wd,hd) in faces]

def _make_face_mask(h, w, faces, pad_scale):
    mask = np.zeros((h, w), dtype=np.float32)
    for (x, y, fw, fh) in faces:
        cx, cy = x + fw/2, y + fh/2
        ew, eh = fw * pad_scale, fh * pad_scale
        axes = (max(1,int(ew/2)), max(1,int(eh/2)))
        center = (int(cx), int(cy))
        if HAS_CV2:
            cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
        else:
            yy, xx = np.ogrid[:h, :w]
            mx = ((xx-center[0])/(axes[0]+1e-6))**2 + ((yy-center[1])/(axes[1]+1e-6))**2 <= 1
            mask[mx] = 1.0
    k = max(21, ((w+h)//180)*2 + 1)
    if HAS_CV2:
        mask = cv2.GaussianBlur(mask, (k, k), sigmaX=k/6)
    else:
        from scipy.ndimage import gaussian_filter
        mask = gaussian_filter(mask, sigma=k/6)
    return mask

def apply_face_glow(img: Image.Image, faces, strength=40, pad_scale=1.2):
    if len(faces) == 0 or strength <= 0:
        return img
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]
    base = arr.astype(np.float32) / 255.0
    mask = _make_face_mask(h, w, faces, pad_scale)

    # brighten in LAB + bilateral smoothing ringan
    if HAS_CV2:
        lab = cv2.cvtColor((base*255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        L, A, B = cv2.split(lab)
        gain = 1.0 + (strength/100.0)*0.32
        L = np.clip(L * gain, 0, 255)
        lab2 = cv2.merge((L, A, B)).astype(np.uint8)
        glow = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB).astype(np.float32)/255.0
        glow_bgr = cv2.cvtColor((glow*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        glow_bgr = cv2.bilateralFilter(glow_bgr, d=5, sigmaColor=15+int(strength*0.5), sigmaSpace=15+int(strength*0.5))
        glow = cv2.cvtColor(glow_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    else:
        glow = np.clip(base * (1.0 + (strength/100.0)*0.25), 0, 1.0)

    out = base*(1.0 - mask[...,None]) + glow*mask[...,None]
    return Image.fromarray(np.clip(out*255, 0, 255).astype(np.uint8))

def apply_matte_skin(img: Image.Image, faces, strength=30, pad_scale=1.15):
    """Haluskan kulit hanya pada wajah; blend lembut agar tidak ‘plastik’."""
    if len(faces) == 0 or not HAS_CV2 or strength <= 0:
        return img
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]
    base = arr.astype(np.float32) / 255.0
    mask = _make_face_mask(h, w, faces, pad_scale)

    # smoothing: bilateral sedikit lebih kuat + campur ke base
    sm = cv2.bilateralFilter(arr, d=7, sigmaColor=25+int(strength*0.7), sigmaSpace=25+int(strength*0.7))
    sm = sm.astype(np.float32) / 255.0

    # blend strength → alpha max 0.6
    alpha = min(0.6, 0.25 + strength/100.0 * 0.35)
    out = base*(1.0 - (mask*alpha)[...,None]) + sm*(mask*alpha)[...,None]
    return Image.fromarray(np.clip(out*255, 0, 255).astype(np.uint8))

# -------------- Auto params --------------
def auto_params(has_face: bool, res_pixels: int):
    if has_face:
        resampler = "bicubic"; sharpen = 0.8; micro_c = 1.08; usm_r, usm_p, usm_t = 0.8, 110, 3
    else:
        resampler = "lanczos"; sharpen = 1.0; micro_c = 1.18; usm_r, usm_p, usm_t = 1.0, 140, 2
    if res_pixels > 70_000_000:
        sharpen -= 0.1; micro_c -= 0.03; usm_p = int(usm_p * 0.9)
    return dict(resampler=resampler, sharpen_amt=max(0.5, sharpen),
                micro_contrast=max(1.0, micro_c),
                usm_radius=usm_r, usm_percent=usm_p, usm_thresh=usm_t)

# ---------------- Process ----------------
if st.button("🚀 Jalankan 8×"):
    if not uploaded:
        st.warning("Unggah minimal satu gambar.")
    else:
        cap_bytes = int(MAX_MB * 1024 * 1024)
        all_results = []
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

                # --- 1) Faktor efektif (Mode Ringan) ---
                max_mp = max_out_mp if lite_mode else None
                src_px = src_w * src_h
                eff_factor = capped_factor_for_cloud(src_w, src_h, REQ_FACTOR, max_mp)
                target_w, target_h = int(src_w * eff_factor), int(src_h * eff_factor)

                # --- 2) Upscale bertahap ---
                if keep_if_bigger and (target_w <= src_w or target_h <= src_h):
                    out = img.copy()
                    eff_factor = 1.0
                    was_capped = False
                else:
                    filt = resample_filter("bicubic")
                    out = upscale_stepwise(img, eff_factor, filt)
                    was_capped = eff_factor < REQ_FACTOR

                # --- 3) Params & efek ---
                if mode == "Auto":
                    faces = detect_faces_fast(out, max_side=1024) if HAS_CV2 else []
                    has_face = len(faces) > 0
                    p = auto_params(has_face, out.width*out.height)
                    out = gentle_sharpen(out, p["sharpen_amt"], p["micro_contrast"], p["usm_radius"], p["usm_percent"], p["usm_thresh"])
                    if has_face and HAS_CV2 and enable_face_glow:
                        g_strength = 35 if out.width*out.height < 50_000_000 else 28
                        out = apply_face_glow(out, faces, strength=g_strength, pad_scale=1.2)
                    if has_face and HAS_CV2 and matte_on:
                        out = apply_matte_skin(out, faces, strength=matte_strength, pad_scale=matte_pad)
                    preset_used = "AUTO (Human)" if has_face else "AUTO (Produk/Kain)"
                else:
                    # Manual preset/kustom
                    if preset == "Default Natural":
                        p = dict(resampler="bicubic", sharpen_amt=0.8, micro_contrast=1.10, usm_radius=0.9, usm_percent=120, usm_thresh=2)
                    elif preset == "Anti-Halo":
                        p = dict(resampler="bicubic", sharpen_amt=0.6, micro_contrast=1.05, usm_radius=0.7, usm_percent=90, usm_thresh=3)
                    elif preset == "Tekstur Kain":
                        p = dict(resampler="lanczos", sharpen_amt=1.05, micro_contrast=1.18, usm_radius=1.0, usm_percent=140, usm_thresh=2)
                    else:
                        p = dict(resampler=("bicubic" if resampler_ui=="Bicubic" else "lanczos"),
                                 sharpen_amt=sharpen_amt_ui, micro_contrast=micro_contrast_ui,
                                 usm_radius=usm_radius_ui, usm_percent=usm_percent_ui, usm_thresh=usm_thresh_ui)
                    out = gentle_sharpen(out, p["sharpen_amt"], p["micro_contrast"], p["usm_radius"], p["usm_percent"], p["usm_thresh"])
                    faces = detect_faces_fast(out, max_side=1024) if HAS_CV2 else []
                    if enable_face_glow and HAS_CV2:
                        out = apply_face_glow(out, faces, strength=glow_strength, pad_scale=1.2 if face_pad is None else face_pad)
                    if matte_on and HAS_CV2:
                        out = apply_matte_skin(out, faces, strength=matte_strength, pad_scale=matte_pad)
                    if (enable_face_glow or matte_on) and not HAS_CV2:
                        st.info("OpenCV tidak tersedia di server: efek wajah (Face Glow/Matte) dilewati.")
                    preset_used = preset

                # --- 4) Encode ≤ 12 MB ---
                data, used_q, note = maximize_under_cap(out, cap_bytes)

                # --- 5) Preview & per-image download ---
                col1, col2 = st.columns(2, gap="large")
                with col1:
                    st.subheader("Sebelum")
                    st.image(img, caption=f"{f.name} — {src_w}×{src_h}px | {src_orient}", use_column_width=True)
                with col2:
                    st.subheader("Sesudah (8×)")
                    new_w, new_h = out.size
                    size_mb = len(data) / (1024 * 1024)
                    faces_count = len(detect_faces_fast(out, max_side=1024)) if HAS_CV2 else 0
                    captext = (
                        f"{Path(f.name).stem}_{suffix}.jpg — {new_w}×{new_h}px | {src_orient} | "
                        f"req 8.0× → eff {eff_factor:.2f}×" + (" (capped)" if was_capped else "") +
                        (f" | faces: {faces_count}" if HAS_CV2 else " | faces: n/a") +
                        (f" | Matte:{'on' if (matte_on and HAS_CV2) else 'off'}" ) +
                        (f" | FaceGlow:{'on' if (enable_face_glow and HAS_CV2 and faces_count>0) else 'off'}") +
                        f" | preset: {preset_used} | q={used_q} | {size_mb:.2f} MB ({note}) | DPI 400"
                    )
                    st.image(out, caption=captext, use_column_width=True)
                    fname = f"{Path(f.name).stem}_{suffix}.jpg"
                    st.download_button("⬇️ Unduh hasil", data, file_name=fname, mime="image/jpeg")

                all_results.append((fname, data))
                del out; gc.collect()

            except MemoryError:
                st.error(f"Gagal memproses {f.name}: kehabisan memori. Aktifkan Mode Ringan / kurangi jumlah gambar.")
            except Exception as e:
                st.error(f"Gagal memproses {f.name}: {e}")

            progress.progress(i / len(uploaded))

        # ---------- Download all (ZIP) ----------
        if all_results:
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_STORED) as zf:
                for fname, blob in all_results:
                    zf.writestr(fname, blob)
            zip_buf.seek(0)
            st.success(f"Selesai! {len(all_results)} file siap diunduh.")
            st.download_button("📦 Unduh semua (ZIP)", zip_buf, file_name="results_upscaled.zip", mime="application/zip")                
