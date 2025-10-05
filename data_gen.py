"""
Synthetic operator data generator.
Generates images for operators + - x ÷ using system fonts.
"""
from pathlib import Path
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np

OUT_DIR = Path(__file__).parent / "data" / "operators"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OPERATORS = ['+', '-', '×', '÷']
FONTS = ["DejaVuSans.ttf", "FreeMono.ttf", "LiberationSans-Regular.ttf"]


def generate_symbol_image(symbol, size=64, font_name=None, jitter=4):
    img = Image.new('L', (size, size), color=255)
    draw = ImageDraw.Draw(img)
    font_path = font_name
    try:
        font = ImageFont.truetype(font_path, int(size*0.7)) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), symbol, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (size - w)//2 + random.randint(-jitter, jitter)
    y = (size - h)//2 + random.randint(-jitter, jitter)
    draw.text((x, y), symbol, fill=0, font=font)
    # Convert to numpy and apply small random rotation
    arr = np.array(img)
    angle = random.uniform(-15, 15)
    pil = Image.fromarray(arr)
    pil = pil.rotate(angle, expand=False, fillcolor=255)
    return pil


def generate_dataset(samples_per_class=500):
    for op in OPERATORS:
        class_dir = OUT_DIR / op
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(samples_per_class):
            font = random.choice(FONTS)
            try:
                img = generate_symbol_image(op, size=64, font_name=font)
            except Exception:
                img = generate_symbol_image(op, size=64, font_name=None)
            img.save(class_dir / f"{op}_{i}.png")


if __name__ == '__main__':
    generate_dataset(1000)
