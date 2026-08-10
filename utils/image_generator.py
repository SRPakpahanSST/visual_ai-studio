import torch
from diffusers import (
    StableDiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler
)
from PIL import Image
import os
import gc

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_base_pipeline(model_id="runwayml/stable-diffusion-v1-5", scheduler_name="Euler A"):
    device = get_device()
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
        low_cpu_mem_usage=True,
        use_safetensors=True
    ).to(device)
    
    # Set scheduler
    scheduler_lower = scheduler_name.lower()
    if scheduler_lower == "euler a":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True
        )
    elif scheduler_lower == "dpm++":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True,
            algorithm_type="dpmsolver++", solver_type="midpoint",
            final_sigmas_type="zero"
        )
    elif scheduler_lower == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    else:
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True
        )
    
    if device == "cuda":
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except:
            pass
    
    return pipe

def generate_simple_image(
    prompt, negative_prompt="", steps=25, guidance_scale=7.5,
    seed=42, model_id="runwayml/stable-diffusion-v1-5",
    height=384, width=384, pipe=None, save_path=None
):
    if pipe is None:
        pipe = load_base_pipeline(model_id)
    
    device = get_device()
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Clean prompt
    prompt = prompt.strip().replace('\n', ' ').replace('\r', ' ')
    if negative_prompt:
        negative_prompt = negative_prompt.strip().replace('\n', ' ').replace('\r', ' ')
    
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
    except (IndexError, RuntimeError) as e:
        print(f"⚠️ Scheduler error: {e}. Fallback ke Euler A...")
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True
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
    
    if pipe is not None:
        del pipe
    gc.collect()
    if get_device() == "cuda":
        torch.cuda.empty_cache()
    
    return image

def generate_advanced_image(
    prompt, negative_prompt="", steps=35, guidance_scale=9.0,
    seed=42, model_id="runwayml/stable-diffusion-v1-5",
    height=384, width=384, pipe=None, save_path=None
):
    return generate_simple_image(
        prompt, negative_prompt, steps, guidance_scale, seed,
        model_id, height, width, pipe, save_path
    )