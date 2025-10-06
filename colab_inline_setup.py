"""
Inline setup script for Google Colab - creates all necessary files if not uploaded
Run this cell if you don't have the project files
"""

import os
import zipfile
from pathlib import Path

def create_models_py():
    """Create models.py with all architectures"""
    content = '''import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class EfficientCNN(nn.Module):
    def __init__(self, out_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        self.block1 = self._make_block(32, 64, 2, 2)
        self.block2 = self._make_block(64, 128, 2, 2)
        self.block3 = self._make_block(128, 256, 2, 2)
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, out_classes)
        )
    
    def _make_block(self, in_ch, out_ch, blocks, stride):
        layers = []
        layers.append(nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            SEBlock(out_ch)
        ))
        
        for _ in range(1, blocks):
            layers.append(nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True),
                SEBlock(out_ch)
            ))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.head(x)
        return x

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model_class, path, device='cpu', out_classes=10):
    model = model_class(out_classes=out_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    return model

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomRotation(20),
            transforms.RandomAffine(0, translate=(0.15, 0.15), scale=(0.8, 1.2), shear=10),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
'''
    
    with open('models.py', 'w') as f:
        f.write(content)
    print("✅ Created models.py")

def create_data_gen_py():
    """Create data_gen.py for operator generation"""
    content = '''from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import numpy as np
import cv2

def generate_operators(samples_per_class=1000):
    operators = ['+', '-', '×', '÷']
    base_dir = Path('data/operators')
    base_dir.mkdir(exist_ok=True)
    
    fonts = []
    try:
        fonts.append(ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 40))
    except:
        fonts.append(ImageFont.load_default())
    
    for op in operators:
        op_dir = base_dir / op
        op_dir.mkdir(exist_ok=True)
        
        for i in range(samples_per_class):
            size = random.randint(56, 72)
            bg_color = random.randint(245, 255)
            
            img = Image.new('L', (size, size), color=bg_color)
            draw = ImageDraw.Draw(img)
            
            font = random.choice(fonts)
            text_color = random.randint(0, 60)
            
            bbox = draw.textbbox((0, 0), op, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            
            x = (size - w) // 2 + random.randint(-10, 10)
            y = (size - h) // 2 + random.randint(-10, 10)
            
            draw.text((x, y), op, fill=text_color, font=font)
            
            if random.random() < 0.4:
                angle = random.uniform(-20, 20)
                img = img.rotate(angle, expand=False, fillcolor=bg_color)
            
            if random.random() < 0.3:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))
            
            if random.random() < 0.4:
                arr = np.array(img)
                noise = np.random.normal(0, random.uniform(5, 15), arr.shape)
                arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
                img = Image.fromarray(arr)
            
            img = img.resize((64, 64), Image.Resampling.LANCZOS)
            img.save(op_dir / f"{i}.png")
    
    print(f"Generated {samples_per_class} realistic handwritten-style samples per operator class")

if __name__ == '__main__':
    generate_operators()
'''
    
    with open('data_gen.py', 'w') as f:
        f.write(content)
    print("✅ Created data_gen.py")

def create_train_digits_py():
    """Create train_digits.py"""
    content = '''import torch
from torch import nn, optim
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from models import EfficientCNN, save_model, get_transforms
import os

def train_digits():
    os.makedirs('models', exist_ok=True)
    
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)
    
    try:
        full_trainset = datasets.EMNIST('data/emnist', split='digits', train=True, download=True, transform=train_transform)
        testset = datasets.EMNIST('data/emnist', split='digits', train=False, download=True, transform=val_transform)
        print("Using EMNIST dataset (handwritten digits)")
    except:
        full_trainset = datasets.MNIST('data/mnist', train=True, download=True, transform=train_transform)
        testset = datasets.MNIST('data/mnist', train=False, download=True, transform=val_transform)
        print("Using MNIST dataset (fallback)")
    
    train_size = int(0.9 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    trainset, valset = random_split(full_trainset, [train_size, val_size])
    
    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(valset, batch_size=128, shuffle=False, num_workers=2)
    test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on {device}")
    
    model = EfficientCNN(out_classes=10).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    best_val_acc = 0
    
    for epoch in range(20):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f'Epoch {epoch+1}/20: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, 'models/digit_cnn.pth')
        
        scheduler.step()
    
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    
    test_acc = 100. * test_correct / test_total
    print(f'Final Test Accuracy: {test_acc:.2f}%')
    print("Digit model saved!")

if __name__ == '__main__':
    train_digits()
'''
    
    with open('train_digits.py', 'w') as f:
        f.write(content)
    print("✅ Created train_digits.py")

def create_train_operators_py():
    """Create train_operators.py"""
    content = '''import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
from pathlib import Path
from models import EfficientCNN, save_model, get_transforms, mixup_data, mixup_criterion
import json
import os

class OperatorDataset(Dataset):
    def __init__(self, transform=None):
        self.samples = []
        self.labels = []
        self.classes = ['+', '-', '×', '÷']
        self.transform = transform
        
        for idx, op in enumerate(self.classes):
            op_dir = Path('data/operators') / op
            for img_path in op_dir.glob('*.png'):
                self.samples.append(img_path)
                self.labels.append(idx)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert('L')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

def train_operators():
    os.makedirs('models', exist_ok=True)
    
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)
    
    full_dataset = OperatorDataset(transform=train_transform)
    
    if len(full_dataset) == 0:
        print("No operator data found! Run data_gen.py first.")
        return
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    trainset, valset = random_split(full_dataset, [train_size, val_size])
    
    val_dataset = OperatorDataset(transform=val_transform)
    val_indices = valset.indices
    valset = torch.utils.data.Subset(val_dataset, val_indices)
    
    train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=2)
    
    with open('models/op_classes.json', 'w') as f:
        json.dump(full_dataset.classes, f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on {device}")
    
    model = EfficientCNN(out_classes=4).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    
    best_val_acc = 0
    
    for epoch in range(15):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            if torch.rand(1) < 0.5:
                mixed_imgs, y_a, y_b, lam = mixup_data(imgs, labels, alpha=0.2)
                optimizer.zero_grad()
                outputs = model(mixed_imgs)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total if train_total > 0 else 0
        val_acc = 100. * val_correct / val_total if val_total > 0 else 0
        
        print(f'Epoch {epoch+1}/15: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, 'models/op_cnn.pth')
        
        scheduler.step()
    
    print(f'Best Validation Accuracy: {best_val_acc:.2f}%')
    print("Operator model saved!")

if __name__ == '__main__':
    train_operators()
'''
    
    with open('train_operators.py', 'w') as f:
        f.write(content)
    print("✅ Created train_operators.py")

def create_predict_py():
    """Create predict.py"""
    content = '''import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from models import EfficientCNN, load_model, get_transforms
import json
import cv2

def segment_image(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    img_area = gray.shape[0] * gray.shape[1]
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        
        if (area > 200 and area < img_area * 0.5 and w > 10 and h > 10 and 0.2 < w/h < 5.0):
            boxes.append((x, y, w, h))
    
    boxes = sorted(boxes, key=lambda b: b[0])
    
    symbols = []
    for x, y, w, h in boxes:
        pad = 5
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(gray.shape[1], x + w + pad)
        y2 = min(gray.shape[0], y + h + pad)
        
        crop = gray[y1:y2, x1:x2]
        
        if np.mean(crop) < 127:
            crop = 255 - crop
        
        size = max(crop.shape[0], crop.shape[1])
        size = max(size, 32)
        
        square = 255 * np.ones((size, size), dtype=np.uint8)
        
        start_y = (size - crop.shape[0]) // 2
        start_x = (size - crop.shape[1]) // 2
        square[start_y:start_y + crop.shape[0], start_x:start_x + crop.shape[1]] = crop
        
        symbols.append(square)
    
    return symbols

class Solver:
    def __init__(self, digit_model_path='models/digit_cnn.pth', op_model_path='models/op_cnn.pth', device='cpu'):
        self.device = device
        try:
            self.digit_model = load_model(EfficientCNN, digit_model_path, device, 10)
            self.op_model = load_model(EfficientCNN, op_model_path, device, 4)
        except Exception as e:
            print(f"Error loading models: {e}")
            return
        
        self.transform = get_transforms(train=False)
        
        try:
            with open('models/op_classes.json', 'r') as f:
                self.op_classes = json.load(f)
        except:
            self.op_classes = ['+', '-', '×', '÷']
    
    def preprocess_symbol(self, img_arr):
        pil = Image.fromarray(img_arr).convert('L')
        tensor = self.transform(pil).unsqueeze(0).to(self.device)
        return tensor
    
    def predict_image(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image.convert('L'))
        
        symbols = segment_image(image)
        if not symbols:
            return "No symbols detected", None
        
        expression = ''
        confidences = []
        
        for symbol in symbols:
            tensor = self.preprocess_symbol(symbol)
            
            with torch.no_grad():
                digit_out = self.digit_model(tensor)
                op_out = self.op_model(tensor)
                
                digit_probs = F.softmax(digit_out, dim=1)[0]
                op_probs = F.softmax(op_out, dim=1)[0]
                
                digit_conf = digit_probs.max().item()
                op_conf = op_probs.max().item()
                
                if op_conf > digit_conf and op_conf > 0.6:
                    symbol_pred = self.op_classes[op_probs.argmax()]
                    confidences.append(op_conf)
                elif digit_conf > 0.5:
                    symbol_pred = str(digit_probs.argmax().item())
                    confidences.append(digit_conf)
                else:
                    if op_conf > digit_conf:
                        symbol_pred = self.op_classes[op_probs.argmax()]
                        confidences.append(op_conf)
                    else:
                        symbol_pred = str(digit_probs.argmax().item())
                        confidences.append(digit_conf)
                
                expression += symbol_pred
        
        try:
            python_expr = expression.replace('×', '*').replace('÷', '/')
            result = eval(python_expr)
            avg_confidence = np.mean(confidences)
            return f"{expression} (conf: {avg_confidence:.2f})", result
        except Exception as e:
            return f"Error evaluating {expression}: {str(e)}", None
'''
    
    with open('predict.py', 'w') as f:
        f.write(content)
    print("✅ Created predict.py")

def create_app_py():
    """Create app.py"""
    content = '''import gradio as gr
from predict import Solver
from PIL import Image
import numpy as np
import torch
import os

def solve_equation(img):
    if img is None:
        return "Please upload an image", ""
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        solver = Solver(device=device)
        
        if isinstance(img, Image.Image):
            arr = np.array(img.convert('L'))
        else:
            arr = img
        
        expression, result = solver.predict_image(arr)
        return expression, str(result) if result is not None else "Could not evaluate"
    except Exception as e:
        return f"Error: {str(e)}", "Make sure models are trained"

def check_models():
    digit_model = os.path.exists('models/digit_cnn.pth')
    op_model = os.path.exists('models/op_cnn.pth')
    
    if not (digit_model and op_model):
        return "⚠️ Models not found. Run training first."
    return "✅ Models loaded successfully"

interface = gr.Interface(
    fn=solve_equation,
    inputs=gr.Image(type='pil', label='Upload handwritten equation'),
    outputs=[
        gr.Textbox(label='Detected Expression'),
        gr.Textbox(label='Result')
    ],
    title='🧮 Handwritten Equation Solver',
    description='Upload an image of handwritten arithmetic (digits and +, -, ×, ÷)',
    article=check_models()
)

if __name__ == '__main__':
    interface.launch(share=True)
'''
    
    with open('app.py', 'w') as f:
        f.write(content)
    print("✅ Created app.py")

def setup_colab_project():
    """Main setup function"""
    print("🚀 Setting up Handwritten Equation Solver for Google Colab...")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/operators', exist_ok=True)
    os.makedirs('data/mnist', exist_ok=True)
    os.makedirs('data/emnist', exist_ok=True)
    
    # Create all Python files
    create_models_py()
    create_data_gen_py()
    create_train_digits_py()
    create_train_operators_py()
    create_predict_py()
    create_app_py()
    
    print("\n✅ Project setup complete!")
    print("\nNext steps:")
    print("1. Run: generate_operators()")
    print("2. Run: train_digits()")
    print("3. Run: train_operators()")
    print("4. Test with the Gradio interface")

if __name__ == "__main__":
    setup_colab_project()