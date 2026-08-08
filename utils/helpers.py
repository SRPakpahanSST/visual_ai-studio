import matplotlib.pyplot as plt
from PIL import Image
import os

def display_grid(images, titles=None, rows=1, cols=4, figsize=(20, 5)):
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    for i, img in enumerate(images):
        if i < len(axes):
            axes[i].imshow(img)
            axes[i].axis('off')
            if titles and i < len(titles):
                axes[i].set_title(titles[i], fontsize=10)
    for j in range(len(images), len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    return fig

def save_image(image: Image.Image, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    image.save(path)