import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
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
        use_karras_sigmas=True
    )
    if device == "cuda":
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except:
            pass
    return pipe

def generate_simple_image(prompt, negative_prompt="", steps=25, guidance_scale=7.5, seed=42, model_id="runwayml/stable-diffusion-v1-5", save_path=None):
    pipe = load_base_pipeline(model_id)
    generator = torch.Generator(device=get_device()).manual_seed(seed)
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=1
        )
    image = result.images[0]
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        image.save(save_path)
    del pipe
    gc.collect()
    if get_device() == "cuda":
        torch.cuda.empty_cache()
    return image

def generate_advanced_image(prompt, negative_prompt="", steps=35, guidance_scale=9.0, seed=42, model_id="runwayml/stable-diffusion-v1-5", save_path=None):
    return generate_simple_image(prompt, negative_prompt, steps, guidance_scale, seed, model_id, save_path)