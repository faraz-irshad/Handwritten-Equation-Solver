"""
Simplified, error-free data generation for Google Colab
No OpenCV perspective transforms - uses only PIL and basic numpy
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import numpy as np

def generate_operators(samples_per_class=1000):
    """Generate operator data without problematic OpenCV transforms"""
    operators = ['+', '-', '×', '÷']
    base_dir = Path('data/operators')
    base_dir.mkdir(exist_ok=True)
    
    # Try to load fonts
    fonts = []
    font_paths = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/System/Library/Fonts/Arial.ttf',
        'arial.ttf'
    ]
    
    for font_path in font_paths:
        for size in [32, 36, 40, 44, 48]:
            try:
                fonts.append(ImageFont.truetype(font_path, size))
            except:
                continue
    
    # Fallback to default font
    if not fonts:
        fonts = [ImageFont.load_default()]
    
    print(f"Using {len(fonts)} font variations")
    
    for op in operators:
        op_dir = base_dir / op
        op_dir.mkdir(exist_ok=True)
        
        print(f"Generating {samples_per_class} samples for '{op}'...")
        
        for i in range(samples_per_class):
            # Create base image
            size = random.randint(60, 80)
            bg_color = random.randint(240, 255)
            img = Image.new('L', (size, size), color=bg_color)
            draw = ImageDraw.Draw(img)
            
            # Random font and color
            font = random.choice(fonts)
            text_color = random.randint(0, 80)
            
            # Get text dimensions and center
            try:
                bbox = draw.textbbox((0, 0), op, font=font)
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except:
                # Fallback for older PIL versions
                w, h = draw.textsize(op, font=font)
            
            # Random position with variation
            x = (size - w) // 2 + random.randint(-8, 8)
            y = (size - h) // 2 + random.randint(-8, 8)
            
            # Draw text with slight thickness variation
            thickness = random.randint(1, 3)
            for dx in range(thickness):
                for dy in range(thickness):
                    draw.text((x + dx, y + dy), op, fill=text_color, font=font)
            
            # Safe transformations using PIL only
            
            # 1. Rotation
            if random.random() < 0.5:
                angle = random.uniform(-25, 25)
                img = img.rotate(angle, expand=False, fillcolor=bg_color)
            
            # 2. Blur
            if random.random() < 0.3:
                radius = random.uniform(0.2, 1.5)
                img = img.filter(ImageFilter.GaussianBlur(radius=radius))
            
            # 3. Noise using numpy
            if random.random() < 0.4:
                arr = np.array(img)
                noise_level = random.uniform(8, 20)
                noise = np.random.normal(0, noise_level, arr.shape)
                arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
                img = Image.fromarray(arr)
            
            # 4. Simple shear using PIL
            if random.random() < 0.3:
                shear_x = random.uniform(-0.2, 0.2)
                shear_y = random.uniform(-0.2, 0.2)
                # Simple affine transform matrix for shear
                transform_matrix = (1, shear_x, 0, shear_y, 1, 0)
                img = img.transform(img.size, Image.AFFINE, transform_matrix, fillcolor=bg_color)
            
            # 5. Brightness/contrast variation
            if random.random() < 0.4:
                arr = np.array(img)
                # Brightness
                brightness = random.uniform(0.8, 1.2)
                arr = np.clip(arr * brightness, 0, 255).astype(np.uint8)
                
                # Contrast
                contrast = random.uniform(0.8, 1.2)
                arr = np.clip((arr - 128) * contrast + 128, 0, 255).astype(np.uint8)
                
                img = Image.fromarray(arr)
            
            # Resize to standard size
            img = img.resize((64, 64), Image.Resampling.LANCZOS)
            
            # Save image
            img.save(op_dir / f"{i}.png")
            
            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{samples_per_class} samples for '{op}'")
    
    print(f"✅ Generated {samples_per_class} samples per operator class")

if __name__ == '__main__':
    generate_operators()