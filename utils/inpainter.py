import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import os
import gc
import numpy as np
from .generator import get_device

# Coba import cv2, jika gagal beri fallback
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️ opencv-python tidak terinstall. Fungsi create_auto_mask tidak tersedia.")

def load_inpainting_pipeline(model_id="runwayml/stable-diffusion-inpainting"):
    device = get_device()
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
        low_cpu_mem_usage=True
    ).to(device)
    if device == "cuda":
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
    return pipe

def inpaint_engine(image, mask, prompt, negative_prompt="", seed=9, guidance_scale=7.5, num_inference_steps=50, strength=1.0, save_path=None):
    device = get_device()
    pipe = load_inpainting_pipeline()
    generator = torch.Generator(device=device).manual_seed(seed)
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            image=image,
            mask_image=mask,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )
    output = result.images[0]
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        output.save(save_path)
    del pipe
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return output

def create_auto_mask(image, method='edge', threshold=128):
    if not CV2_AVAILABLE:
        raise ImportError("opencv-python tidak terinstall. Install dengan: pip install opencv-python")
    img_np = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    if method == 'edge':
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.dilate(edges, kernel, iterations=2)
    else:
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return Image.fromarray(mask)