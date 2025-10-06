import torch
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
    
    # Create datasets with different transforms
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)
    
    full_dataset = OperatorDataset(transform=train_transform)
    
    if len(full_dataset) == 0:
        print("No operator data found! Run data_gen.py first.")
        return
    
    # Split into train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    trainset, valset = random_split(full_dataset, [train_size, val_size])
    
    # Create validation dataset with different transform
    val_dataset = OperatorDataset(transform=val_transform)
    val_indices = valset.indices
    valset = torch.utils.data.Subset(val_dataset, val_indices)
    
    train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=2)
    
    # Save class mapping
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
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            # Apply mixup augmentation
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
        
        # Validation
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