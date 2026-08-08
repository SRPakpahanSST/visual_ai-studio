import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler
from PIL import Image
import os
import gc

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_base_pipeline(model_id="runwayml/stable-diffusion-v1-5"):
    device = get_device()
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
        low_cpu_mem_usage=True,
        use_safetensors=True
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="dpmsolver++",
        solver_type="midpoint",
        final_sigmas_type="zero"
    )
    if device == "cuda":
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except:
            pass
    return pipe

# ============================================
# ✅ SATU DEFINISI FUNGSI generate_simple_image
# ============================================
def generate_simple_image(
    prompt,
    negative_prompt="",
    steps=25,
    guidance_scale=7.5,
    seed=42,
    model_id="runwayml/stable-diffusion-v1-5",
    height=384,
    width=384,
    pipe=None,
    save_path=None
):
    """
    Generate gambar dari teks prompt dengan parameter yang bisa diatur.
    
    Args:
        prompt: Teks prompt untuk generate gambar
        negative_prompt: Teks negatif prompt
        steps: Jumlah inference steps
        guidance_scale: Skala guidance
        seed: Seed untuk reproducibility
        model_id: ID model dari HuggingFace
        height: Tinggi gambar
        width: Lebar gambar
        pipe: Pipeline yang sudah diload (opsional)
        save_path: Path untuk menyimpan gambar
    
    Returns:
        PIL Image object
    """
    if pipe is None:
        pipe = load_base_pipeline(model_id)
    
    device = get_device()
    generator = torch.Generator(device=device).manual_seed(seed)
    
    try:
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                height=height,
                width=width,
                num_images_per_prompt=1
            )
    except IndexError as e:
        # ✅ Fallback: jika DPM++ error, coba ganti ke scheduler yang lebih stabil
        print(f"⚠️ Scheduler error: {e}. Mencoba fallback ke Euler A...")
        from diffusers import EulerAncestralDiscreteScheduler
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True
        )
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                height=height,
                width=width,
                num_images_per_prompt=1
            )
    
    image = result.images[0]
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        image.save(save_path)
    
    # Cleanup
    if pipe is not None:
        del pipe
    gc.collect()
    if get_device() == "cuda":
        torch.cuda.empty_cache()
    
    return image

# ============================================
# generate_advanced_image - reuse generate_simple_image
# ============================================
def generate_advanced_image(
    prompt,
    negative_prompt="",
    steps=35,
    guidance_scale=9.0,
    seed=42,
    model_id="runwayml/stable-diffusion-v1-5",
    height=384,
    width=384,
    pipe=None,
    save_path=None
):
    """
    Fungsi advanced dengan parameter default yang lebih tinggi.
    """
    return generate_simple_image(
        prompt, negative_prompt, steps, guidance_scale, seed,
        model_id, height, width, pipe, save_path
    )