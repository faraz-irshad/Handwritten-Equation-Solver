from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import numpy as np
import cv2

def generate_handwritten_stroke(img_array, start, end, thickness=2):
    """Generate more realistic handwritten strokes"""
    # Add slight curve and thickness variation
    points = []
    steps = max(abs(end[0] - start[0]), abs(end[1] - start[1]))
    
    for i in range(steps + 1):
        t = i / max(steps, 1)
        # Add slight curve
        curve_offset = int(3 * np.sin(t * np.pi) * random.uniform(-1, 1))
        x = int(start[0] + t * (end[0] - start[0]) + curve_offset)
        y = int(start[1] + t * (end[1] - start[1]) + curve_offset)
        
        # Vary thickness
        curr_thickness = max(1, thickness + random.randint(-1, 1))
        cv2.circle(img_array, (x, y), curr_thickness, 0, -1)
    
    return img_array

def generate_operators(samples_per_class=1000):
    operators = ['+', '-', '×', '÷']
    base_dir = Path('data/operators')
    base_dir.mkdir(exist_ok=True)
    
    # Handwriting-style fonts (more realistic)
    font_names = [
        "DejaVuSans.ttf", "Arial.ttf", "Times.ttf", 
        "Helvetica.ttf", "Comic Sans MS.ttf", "Brush Script MT.ttf"
    ]
    fonts = []
    
    for font_name in font_names:
        for size in [28, 32, 36, 40, 44, 48, 52]:
            try:
                fonts.append(ImageFont.truetype(font_name, size))
            except:
                continue
    
    if not fonts:
        fonts = [ImageFont.load_default()]
    
    for op in operators:
        op_dir = base_dir / op
        op_dir.mkdir(exist_ok=True)
        
        for i in range(samples_per_class):
            # Create base image
            size = random.randint(56, 72)
            bg_color = random.randint(245, 255)
            
            # 30% handdrawn, 70% font-based for variety
            if random.random() < 0.3 and op in ['+', '-']:
                # Hand-drawn style for simple operators
                img_array = np.full((size, size), bg_color, dtype=np.uint8)
                center = size // 2
                thickness = random.randint(2, 4)
                
                if op == '+':
                    # Vertical line
                    start_v = (center, center - random.randint(12, 18))
                    end_v = (center, center + random.randint(12, 18))
                    img_array = generate_handwritten_stroke(img_array, start_v, end_v, thickness)
                    
                    # Horizontal line
                    start_h = (center - random.randint(12, 18), center)
                    end_h = (center + random.randint(12, 18), center)
                    img_array = generate_handwritten_stroke(img_array, start_h, end_h, thickness)
                    
                elif op == '-':
                    # Horizontal line
                    start_h = (center - random.randint(15, 20), center)
                    end_h = (center + random.randint(15, 20), center)
                    img_array = generate_handwritten_stroke(img_array, start_h, end_h, thickness)
                
                img = Image.fromarray(img_array)
            else:
                # Font-based with variations
                img = Image.new('L', (size, size), color=bg_color)
                draw = ImageDraw.Draw(img)
                
                font = random.choice(fonts)
                text_color = random.randint(0, 60)
                
                # Get text dimensions
                bbox = draw.textbbox((0, 0), op, font=font)
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                
                # Random position with more variation
                x = (size - w) // 2 + random.randint(-10, 10)
                y = (size - h) // 2 + random.randint(-10, 10)
                
                # Draw text with slight boldness variation
                for dx in range(random.randint(0, 2)):
                    for dy in range(random.randint(0, 2)):
                        draw.text((x + dx, y + dy), op, fill=text_color, font=font)
            
            # Enhanced transformations
            if random.random() < 0.4:
                angle = random.uniform(-20, 20)
                img = img.rotate(angle, expand=False, fillcolor=bg_color)
            
            # Safe perspective distortion
            if random.random() < 0.2:
                arr = np.array(img)
                h, w = arr.shape
                # Create valid perspective points
                offset = random.uniform(1, 3)
                pts1 = np.float32([[0,0], [w,0], [0,h], [w,h]])
                pts2 = np.float32([
                    [offset, offset], [w-offset, offset], 
                    [offset, h-offset], [w-offset, h-offset]
                ])
                try:
                    M = cv2.getPerspectiveTransform(pts1, pts2)
                    arr = cv2.warpPerspective(arr, M, (w, h), borderValue=bg_color)
                    img = Image.fromarray(arr)
                except:
                    pass  # Skip if transform fails
            
            # Blur and noise
            if random.random() < 0.3:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))
            
            if random.random() < 0.4:
                arr = np.array(img)
                noise = np.random.normal(0, random.uniform(5, 15), arr.shape)
                arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
                img = Image.fromarray(arr)
            
            # Simple elastic deformation (safer)
            if random.random() < 0.2:
                try:
                    arr = np.array(img)
                    rows, cols = arr.shape
                    # Simple wave distortion
                    for i in range(rows):
                        shift = int(2 * np.sin(2 * np.pi * i / rows))
                        if shift != 0:
                            arr[i] = np.roll(arr[i], shift)
                    img = Image.fromarray(arr)
                except:
                    pass  # Skip if deformation fails
            
            # Resize to standard size
            img = img.resize((64, 64), Image.Resampling.LANCZOS)
            img.save(op_dir / f"{i}.png")
    
    print(f"Generated {samples_per_class} realistic handwritten-style samples per operator class")

if __name__ == '__main__':
    generate_operators()