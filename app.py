# app.py — Batch Image Upscaler (Cloud-safe + Size Cap)
import io, math, zipfile
from datetime import datetime
from pathlib import Path
import streamlit as st
from PIL import Image, ImageFilter, ImageFile

# ---- Pillow safety ----
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---- Batas aman Cloud (silakan sesuaikan) ----
MAX_SRC_PIXELS = 80_000_000     # ~80 MP sumber
MAX_OUT_PIXELS = 120_000_000    # ~120 MP output

SUPPORTED_TYPES = ("png", "jpg", "jpeg", "webp", "bmp")

st.set_page_config(page_title="Batch Image Upscaler", page_icon="🖼️", layout="wide")
st.title("🖼️ Batch Image Upscaler (Cloud-safe + Size Cap)")

with st.sidebar:
    st.header("Upscale & Output")
    scale = st.select_slider("Skala target", options=[2,3,4], value=4)
    sharpen = st.slider("Penajaman", 0, 3, 1)
    suffix = st.text_input("Akhiran nama file", value=f"x{scale}")

    st.markdown("---")
    st.header("Batas ukuran file")
    size_mb = st.number_input("Maks ukuran output (MB)", 1.0, 50.0, 12.0, 0.5)
    force_jpeg = st.toggle("Paksa output JPEG (disarankan)", value=True,
                           help="Lebih mudah mengontrol ukuran file. PNG bisa sangat besar untuk foto.")
    min_quality = st.slider("Kualitas minimum JPEG", 20, 85, 40, help="Batas bawah saat pencarian kualitas.")

    st.markdown("---")
    st.caption(f"🔒 Cloud guard: sumber ≤ ~{MAX_SRC_PIXELS/1e6:.0f}MP, output ≤ ~{MAX_OUT_PIXELS/1e6:.0f}MP")

uploaded = st.file_uploader("Pilih hingga 10 gambar", type=list(SUPPORTED_TYPES), accept_multiple_files=True)
if uploaded and len(uploaded) > 10:
    st.warning("Maksimal 10 gambar per proses. Hanya 10 pertama dipakai.")
    uploaded = uploaded[:10]

# ---------- Helpers ----------
def upscale_lanczos(img: Image.Image, factor: int, sharpen_steps: int=1) -> Image.Image:
    new_size = (max(1, img.width * factor), max(1, img.height * factor))
    out = img.resize(new_size, Image.LANCZOS)
    for _ in range(sharpen_steps):
        out = out.filter(ImageFilter.UnsharpMask(radius=1.2, percent=150, threshold=3))
    return out

def clamp_for_cloud(img: Image.Image, requested_scale: int):
    """Shrink sumber jika > MAX_SRC_PIXELS dan turunkan skala efektif agar output <= MAX_OUT_PIXELS."""
    w, h = img.width, img.height
    src_px = w * h
    note_parts = []

    if src_px > MAX_SRC_PIXELS:
        shrink_ratio = math.sqrt(MAX_SRC_PIXELS / src_px)
        w2, h2 = max(1, int(w*shrink_ratio)), max(1, int(h*shrink_ratio))
        img = img.resize((w2, h2), Image.LANCZOS)
        note_parts.append(f"pre-shrink {w}×{h}→{w2}×{h2}")
        w, h = img.width, img.height
        src_px = w * h

    max_scale_float = math.sqrt(MAX_OUT_PIXELS / max(1, src_px))
    max_scale_int = max(1, int(max_scale_float))
    eff = min(requested_scale, max_scale_int)
    if eff < requested_scale:
        note_parts.append(f"scale {requested_scale}x→{eff}x (limit)")

    return img, eff, (" | ".join(note_parts) if note_parts else "")

def bytes_len(b: bytes) -> int:
    return len(b)

def save_jpeg_target_size(img: Image.Image, target_bytes: int, min_q: int = 40, max_q: int = 95):
    """
    Simpan ke JPEG <= target_bytes dengan binary search kualitas + downscale jika perlu.
    Mengembalikan (data_bytes, used_quality, downscale_note)
    """
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    # Helper encode
    def encode(q: int, im: Image.Image) -> bytes:
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=q, optimize=True, progressive=True, subsampling=2)
        return buf.getvalue()

    # 1) Binary search kualitas
    lo, hi = min_q, max_q
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        data = encode(mid, img)
        if bytes_len(data) <= target_bytes:
            best = (data, mid)
            lo = mid + 1  # coba kualitas lebih tinggi
        else:
            hi = mid - 1  # turunkan kualitas

    if best is not None:
        return best[0], best[1], ""

    # 2) Jika masih > target pada kualitas minimum → downscale bertahap hingga masuk
    work = img
    note = ""
    for _ in range(6):  # maksimal 6 iterasi downscale
        # Perkirakan rasio diperlukan (akar dari perbandingan ukuran)
        trial = encode(min_q, work)
        if bytes_len(trial) <= target_bytes:
            return trial, min_q, (note or "downscale minimal")
        ratio = math.sqrt(target_bytes / max(1, bytes_len(trial))) * 0.9  # margin 10%
        if ratio >= 1.0:
            # Harusnya sudah cukup, tapi fallback aman
            return trial, min_q, (note or "ok at min quality")
        new_w = max(1, int(work.width * ratio))
        new_h = max(1, int(work.height * ratio))
        work = work.resize((new_w, new_h), Image.LANCZOS)
        note = f"downscale→{new_w}×{new_h}"
    # Fallback terakhir: kembalikan hasil kualitas minimum
    return trial, min_q, (note or "min quality fallback")

def save_png_optimized(img: Image.Image):
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True, compress_level=9)
    return buf.getvalue()

# ---------- Main ----------
if st.button("🚀 Proses Upscale"):
    if not uploaded:
        st.warning("Unggah minimal 1 gambar.")
    else:
        results = []
        progress = st.progress(0)
        target_bytes = int(size_mb * 1024 * 1024)

        for i, f in enumerate(uploaded, start=1):
            try:
                # Load aman
                img = Image.open(f)
                try:
                    img.load()
                except Exception:
                    img = Image.open(io.BytesIO(f.getvalue())); img.load()

                if img.mode not in ("RGB", "RGBA", "L"):
                    img = img.convert("RGB")

                # Cloud clamp
                prepared, eff_scale, note = clamp_for_cloud(img, scale)

                # Upscale
                out = upscale_lanczos(prepared, eff_scale, sharpen)

                # Tentukan format & simpan dengan batas ukuran
                src_ext = (f.name.split(".")[-1] or "jpg").lower()
                out_name_base = f"{Path(f.name).stem}_{eff_scale}x_{suffix}"

                if force_jpeg or src_ext != "png":
                    data, used_q, size_note = save_jpeg_target_size(out, target_bytes, min_q=min_quality, max_q=95)
                    out_name = f"{out_name_base}.jpg"
                    caption_note = f"JPEG q≈{used_q}" + (f" | {size_note}" if size_note else "")
                    mime = "image/jpeg"
                else:
                    # Tetap PNG, tapi jika > target → sarankan paksa JPEG
                    data = save_png_optimized(out)
                    if bytes_len(data) > target_bytes:
                        # otomatis konversi ke JPEG agar memenuhi batas
                        data, used_q, size_note = save_jpeg_target_size(out, target_bytes, min_q=min_quality, max_q=95)
                        out_name = f"{out_name_base}.jpg"
                        caption_note = f"PNG→JPEG q≈{used_q}" + (f" | {size_note}" if size_note else "")
                        mime = "image/jpeg"
                    else:
                        out_name = f"{out_name_base}.png"
                        caption_note = "PNG optimized"
                        mime = "image/png"

                st.image(out, caption=f"{out_name} ({out.width}×{out.height}) — {caption_note}" + (f" — {note}" if note else ""), use_column_width=True)
                st.download_button(f"⬇️ Unduh {out_name}", data, file_name=out_name, mime=mime)
                results.append((out_name, data))
            except Exception as e:
                st.error(f"Gagal memproses {f.name}: {e}")

            progress.progress(i / len(uploaded))

        if results:
            zipbuf = io.BytesIO()
            with zipfile.ZipFile(zipbuf, "w", zipfile.ZIP_DEFLATED) as zf:
                for n, d in results:
                    zf.writestr(n, d)
            st.download_button("📦 Unduh semua (ZIP)", zipbuf.getvalue(), file_name="upscaled_images.zip", mime="application/zip")

st.markdown("---")
st.markdown("""
**Catatan**
- Output akan *diusahakan* ≤ batas ukuran yang kamu set (default 12 MB).
- Strategi: cari kualitas JPEG optimal → bila perlu, *downscale* sedikit agar lolos batas.
- PNG otomatis dikonversi ke JPEG jika PNG tidak bisa memenuhi batas ukuran.
""")
