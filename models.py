import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms

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
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        # Efficient blocks with SE attention
        self.block1 = self._make_block(32, 64, 2, 2)
        self.block2 = self._make_block(64, 128, 2, 2)
        self.block3 = self._make_block(128, 256, 2, 2)
        
        # Head
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

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomRotation(20),
            transforms.RandomAffine(0, translate=(0.15, 0.15), scale=(0.8, 1.2), shear=10),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST/EMNIST stats
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])