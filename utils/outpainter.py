from PIL import Image, ImageDraw
from .inpainter import inpaint_engine
import os

def prepare_outpainting(image, expand_pixels=150, direction='right', background_color=(20,20,40)):
    width, height = image.size
    
    if direction == 'right':
        new_width = width + expand_pixels
        new_height = height
        paste_pos = (0, 0)
        mask_region = (width, 0, new_width, height)
    elif direction == 'left':
        new_width = width + expand_pixels
        new_height = height
        paste_pos = (expand_pixels, 0)
        mask_region = (0, 0, expand_pixels, height)
    elif direction == 'top':
        new_width = width
        new_height = height + expand_pixels
        paste_pos = (0, expand_pixels)
        mask_region = (0, 0, width, expand_pixels)
    elif direction == 'bottom':
        new_width = width
        new_height = height + expand_pixels
        paste_pos = (0, 0)
        mask_region = (0, height, width, new_height)
    else:  # all / zoom out
        new_width = width + 2*expand_pixels
        new_height = height + 2*expand_pixels
        paste_pos = (expand_pixels, expand_pixels)
        mask_region = None
    
    expanded = Image.new('RGB', (new_width, new_height), background_color)
    expanded.paste(image, paste_pos)
    
    mask = Image.new('L', (new_width, new_height), 0)
    draw = ImageDraw.Draw(mask)
    
    if direction == 'all':
        draw.rectangle([0, 0, expand_pixels, new_height], fill=255)
        draw.rectangle([new_width - expand_pixels, 0, new_width, new_height], fill=255)
        draw.rectangle([0, 0, new_width, expand_pixels], fill=255)
        draw.rectangle([0, new_height - expand_pixels, new_width, new_height], fill=255)
    else:
        x1, y1, x2, y2 = mask_region
        draw.rectangle([x1, y1, x2, y2], fill=255)
    
    return expanded, mask

def generate_outpainting(
    image, prompt, negative_prompt="",
    expand_pixels=150, direction='right',
    seed=42, guidance_scale=8.0, num_inference_steps=60,
    save_path=None
):
    expanded, mask = prepare_outpainting(image, expand_pixels, direction)
    result = inpaint_engine(
        image=expanded,
        mask=mask,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        strength=1.0,
        save_path=save_path
    )
    return result