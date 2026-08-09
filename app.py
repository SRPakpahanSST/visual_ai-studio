# ============================================
# FIX: Set environment variables untuk tokenizers
# ============================================
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from PIL import Image
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
        ["Euler A", "DPM++", "DDIM"],  # ✅ Euler A sebagai default
        index=0,
        help="Algoritma sampling untuk diffusion"
    )
    
    st.subheader("📐 Ukuran Gambar")
    img_height = st.slider("Tinggi (height)", 256, 768, 384, step=64)
    img_width = st.slider("Lebar (width)", 256, 768, 384, step=64)
    st.caption("Ukuran lebih kecil = generasi lebih cepat & hemat resource")
    
    st.markdown("---")
    st.caption(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

# ============================================
# ✅ CACHE PIPELINE
# ============================================
@st.cache_resource
def get_pipeline(model_id, scheduler_name):
    """Memuat pipeline sekali dan menyimpannya di cache"""
    from diffusers import (
        StableDiffusionPipeline,
        DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler,
        DDIMScheduler
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
        low_cpu_mem_usage=True
    ).to(device)
    
    # ✅ Set scheduler
    scheduler_lower = scheduler_name.lower()
    
    if scheduler_lower == "euler a":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True
        )
    elif scheduler_lower == "dpm++":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            final_sigmas_type="zero"
        )
    elif scheduler_lower == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config
        )
    else:
        # Default ke Euler A
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True
        )
    
    if device == "cuda":
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except:
            pass
    
    print(f"✅ Pipeline loaded with {scheduler_name} scheduler on {device}")
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
            steps = st.slider("Inference Steps", 10, 50, 20)
            guidance = st.slider("Guidance Scale", 1.0, 15.0, 7.5)
        else:
            steps = st.slider("Inference Steps", 10, 100, 30)
            guidance = st.slider("Guidance Scale", 1.0, 20.0, 9.0)
        generate_btn = st.button("🚀 Generate", type="primary", use_container_width=True)

    if generate_btn:
        with st.spinner("Menghasilkan gambar..."):
            try:
                pipe = get_pipeline(model_id, scheduler_name)
                if mode == "Simple":
                    img = generate_simple_image(
                        prompt_txt, neg_txt, steps, guidance, seed, model_id,
                        height=img_height, width=img_width, pipe=pipe
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
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ============================================
# TAB 2: INPAINTING (Sama seperti sebelumnya)
# ============================================
# ... (kode TAB 2, 3, 4 tetap sama) ...

st.markdown("---")
st.caption("VisualAI Studio - Dibuat untuk APINDO AI Innovation Challenge 2026")