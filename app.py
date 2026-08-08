import streamlit as st
from PIL import Image
import os
import tempfile
import torch
from utils import (
    generate_simple_image,
    generate_advanced_image,
    inpaint_engine,
    create_auto_mask,
    generate_outpainting,
    batch_inference,
    display_grid
)

# ============================================
# KONFIGURASI HALAMAN
# ============================================
st.set_page_config(
    page_title="VisualAI Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎨 VisualAI Studio")
st.markdown("Aplikasi Generasi & Editing Gambar berbasis Stable Diffusion")

# ============================================
# SIDEBAR - PENGATURAN GLOBAL
# ============================================
with st.sidebar:
    st.header("⚙️ Pengaturan Global")
    
    model_id = st.selectbox(
        "Model",
        ["runwayml/stable-diffusion-v1-5", "stabilityai/stable-diffusion-2-1-base"],
        index=0,
        help="Model yang digunakan untuk generate gambar"
    )
    
    scheduler_name = st.selectbox(
        "Scheduler",
        ["DPM++", "Euler A", "DDIM"],
        index=0,
        help="Algoritma sampling untuk diffusion"
    )
    
    # ⭐ BARU: Ukuran gambar
    st.subheader("📐 Ukuran Gambar")
    img_height = st.slider("Tinggi (height)", 256, 768, 384, step=64)
    img_width = st.slider("Lebar (width)", 256, 768, 384, step=64)
    st.caption("Ukuran lebih kecil = generasi lebih cepat & hemat resource")
    
    st.markdown("---")
    st.caption(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

# ============================================
# CACHE PIPELINE (AGAR TIDAK RELOAD)
# ============================================
@st.cache_resource
def get_pipeline(model_id, scheduler_name):
    """Memuat pipeline sekali dan menyimpannya di cache"""
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
        low_cpu_mem_usage=True
    ).to(device)
    # Set scheduler
    if scheduler_name == "DPM++":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True
        )
    # (untuk scheduler lain, bisa ditambahkan)
    if device == "cuda":
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except:
            pass
    return pipe

# ============================================
# TAB 1: TEXT-TO-IMAGE
# ============================================
tab1, tab2, tab3, tab4 = st.tabs(["📸 Text-to-Image", "🖌️ Inpainting", "🖼️ Outpainting", "📊 Batch Generation"])

with tab1:
    st.subheader("Generate Gambar dari Teks")
    col1, col2 = st.columns([2, 1])
    with col1:
        prompt_txt = st.text_area(
            "Prompt",
            value="An astronaut standing on the moon surface, Earth visible in the sky, digital art 2D illustration style, flat colors, vector art, cartoon style",
            height=100
        )
        neg_txt = st.text_area(
            "Negative Prompt",
            value="photorealistic, realistic, photograph, 3d render, messy, blurry, low quality, bad art, ugly, sketch, grainy, unfinished, chromatic aberration",
            height=80
        )
    with col2:
        mode = st.radio("Mode", ["Simple", "Advanced"], index=0)
        seed = st.number_input("Seed", value=222, step=1)
        if mode == "Simple":
            steps = st.slider("Inference Steps", 10, 50, 20)      # ⬅️ turunkan default
            guidance = st.slider("Guidance Scale", 1.0, 15.0, 7.5)
        else:
            steps = st.slider("Inference Steps", 10, 100, 30)     # ⬅️ turunkan default
            guidance = st.slider("Guidance Scale", 1.0, 20.0, 9.0)
        generate_btn = st.button("🚀 Generate", type="primary", use_container_width=True)

    if generate_btn:
        with st.spinner("Menghasilkan gambar..."):
            # Ambil pipeline dari cache
            pipe = get_pipeline(model_id, scheduler_name)
            # Generate dengan ukuran yang diatur
            if mode == "Simple":
                img = generate_simple_image(
                    prompt_txt, neg_txt, steps, guidance, seed, model_id,
                    height=img_height, width=img_width, pipe=pipe   # ⬅️ tambahkan pipe & ukuran
                )
            else:
                img = generate_advanced_image(
                    prompt_txt, neg_txt, steps, guidance, seed, model_id,
                    height=img_height, width=img_width, pipe=pipe
                )
            if img:
                st.image(img, caption=f"Seed: {seed}, Steps: {steps}, Guidance: {guidance}, Size: {img_width}x{img_height}")
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    img.save(tmp.name)
                    with open(tmp.name, "rb") as f:
                        st.download_button("📥 Download", data=f, file_name=f"generated_seed_{seed}.png", mime="image/png")
                    os.unlink(tmp.name)
            else:
                st.error("❌ Gagal generate gambar. Coba kurangi ukuran atau steps.")

# ============================================
# TAB 2: INPAINTING
# ============================================
with tab2:
    st.subheader("Inpainting (Edit gambar dengan mask)")
    uploaded_file = st.file_uploader("Upload gambar", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Gambar asli", width=300)
        mask_option = st.radio("Metode Mask", ["Manual (upload mask)", "Otomatis (edge detection)"])
        mask = None
        if mask_option == "Manual (upload mask)":
            mask_file = st.file_uploader("Upload mask (putih = area yang akan diubah)", type=["png", "jpg"])
            if mask_file is not None:
                mask = Image.open(mask_file).convert("L")
        else:
            if st.button("Buat Mask Otomatis"):
                try:
                    mask = create_auto_mask(image, method='edge')
                    st.image(mask, caption="Mask otomatis", width=300)
                except ImportError as e:
                    st.error(f"❌ {e}. Silakan upload mask manual.")
                except Exception as e:
                    st.error(f"❌ Gagal membuat mask: {e}")

        if mask is not None:
            prompt_inp = st.text_area("Prompt untuk inpainting", "a broken satellite floating in space, digital art 2D style")
            neg_inp = st.text_area("Negative prompt (inpainting)", "photorealistic, 3d render, blurry")
            seed_inp = st.number_input("Seed (inpainting)", value=9, step=1)
            if st.button("🖌️ Jalankan Inpainting"):
                with st.spinner("Proses inpainting..."):
                    try:
                        result = inpaint_engine(image, mask, prompt_inp, neg_inp, seed_inp)
                        st.image(result, caption="Hasil Inpainting")
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                            result.save(tmp.name)
                            with open(tmp.name, "rb") as f:
                                st.download_button("📥 Download", data=f, file_name="inpainted.png", mime="image/png")
                            os.unlink(tmp.name)
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

# ============================================
# TAB 3: OUTPAINTING
# ============================================
with tab3:
    st.subheader("Outpainting (Perluas gambar)")
    out_file = st.file_uploader("Upload gambar untuk outpainting", type=["png", "jpg", "jpeg"])
    if out_file is not None:
        img_out = Image.open(out_file).convert("RGB")
        st.image(img_out, caption="Gambar asli", width=300)
        direction = st.selectbox("Arah perluasan", ["right", "left", "top", "bottom", "all"])
        expand = st.slider("Jumlah piksel perluasan", 50, 300, 150)
        prompt_out = st.text_area("Prompt outpainting", "space scene with stars, nebula, continues the space theme, digital art 2D")
        neg_out = st.text_area("Negative prompt (outpainting)", "photorealistic, 3d render, blurry")
        seed_out = st.number_input("Seed (outpainting)", value=42, step=1)
        if st.button("🖼️ Jalankan Outpainting"):
            with st.spinner("Memperluas gambar..."):
                try:
                    result = generate_outpainting(img_out, prompt_out, neg_out, expand, direction, seed_out)
                    st.image(result, caption="Hasil Outpainting")
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        result.save(tmp.name)
                        with open(tmp.name, "rb") as f:
                            st.download_button("📥 Download", data=f, file_name="outpainted.png", mime="image/png")
                        os.unlink(tmp.name)
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# ============================================
# TAB 4: BATCH GENERATION
# ============================================
with tab4:
    st.subheader("Batch Generation (4 gambar sekaligus)")
    prompt_batch = st.text_area("Prompt (sama untuk semua)", value="astronaut on moon, digital art 2D, flat colors")
    neg_batch = st.text_area("Negative prompt (batch)", value="photorealistic, 3d render, blurry")
    seed_base = st.number_input("Seed awal", value=222, step=1)
    steps_batch = st.slider("Steps", 10, 50, 25)   # ⬅️ turunkan default
    guidance_batch = st.slider("Guidance", 1.0, 15.0, 7.5)
    if st.button("📊 Generate 4 Gambar"):
        with st.spinner("Menghasilkan 4 gambar..."):
            try:
                results = batch_inference(
                    prompt_batch, neg_batch, 4, seed_base,
                    steps_batch, guidance_batch,
                    height=img_height, width=img_width  # ⬅️ tambahkan ukuran
                )
                titles = [f"Seed {seed_base+i}" for i in range(4)]
                fig = display_grid(results, titles, rows=2, cols=2, figsize=(10, 10))
                st.pyplot(fig)
                for i, img in enumerate(results):
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        img.save(tmp.name)
                        with open(tmp.name, "rb") as f:
                            st.download_button(f"📥 Download Image {i+1}", data=f, file_name=f"batch_{i+1}.png", mime="image/png")
                        os.unlink(tmp.name)
            except Exception as e:
                st.error(f"❌ Error: {e}")

st.markdown("---")
st.caption("VisualAI Studio - Dibuat untuk APINDO AI Innovation Challenge 2026")