from .generator import generate_simple_image
from PIL import Image
from typing import List
import os

def batch_inference(
    prompt,
    negative_prompt="",
    num_images=4,
    seed_base=222,
    steps=30,
    guidance_scale=7.5,
    height=384,
    width=384,
    save_dir=None
) -> List[Image.Image]:
    results = []
    for i in range(num_images):
        seed = seed_base + i
        save_path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"batch_{i+1}_seed_{seed}.png")
        img = generate_simple_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            height=height,
            width=width,
            save_path=save_path
        )
        results.append(img)
    return results