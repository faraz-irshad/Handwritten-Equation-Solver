from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random

def generate_operators(samples_per_class=1000):
    operators = ['+', '-', '×', '÷']
    base_dir = Path('data/operators')
    base_dir.mkdir(exist_ok=True)
    
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 40)
    except:
        font = ImageFont.load_default()
    
    for op in operators:
        op_dir = base_dir / op
        op_dir.mkdir(exist_ok=True)
        
        for i in range(samples_per_class):
            img = Image.new('L', (64, 64), color=255)
            draw = ImageDraw.Draw(img)
            
            bbox = draw.textbbox((0, 0), op, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x = (64 - w) // 2 + random.randint(-5, 5)
            y = (64 - h) // 2 + random.randint(-5, 5)
            
            draw.text((x, y), op, fill=0, font=font)
            
            if random.random() < 0.3:
                img = img.rotate(random.uniform(-15, 15), fillcolor=255)
            
            img.save(op_dir / f"{i}.png")
    
    print(f"Generated {samples_per_class} samples per operator class")

if __name__ == '__main__':
    generate_operators()