import torch
from torch import nn, optim
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from models import EfficientCNN, save_model, get_transforms
import os

def train_digits():
    os.makedirs('models', exist_ok=True)
    
    # Load EMNIST (better for handwritten digits) with augmentation
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)
    
    try:
        # Try EMNIST first (better for handwritten)
        full_trainset = datasets.EMNIST('data/emnist', split='digits', train=True, download=True, transform=train_transform)
        testset = datasets.EMNIST('data/emnist', split='digits', train=False, download=True, transform=val_transform)
        print("Using EMNIST dataset (handwritten digits)")
    except:
        # Fallback to MNIST
        full_trainset = datasets.MNIST('data/mnist', train=True, download=True, transform=train_transform)
        testset = datasets.MNIST('data/mnist', train=False, download=True, transform=val_transform)
        print("Using MNIST dataset (fallback)")
    
    # Split training into train/val
    train_size = int(0.9 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    trainset, valset = random_split(full_trainset, [train_size, val_size])
    
    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(valset, batch_size=128, shuffle=False, num_workers=2)
    test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on {device}")
    
    model = EfficientCNN(out_classes=10).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for better generalization
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    best_val_acc = 0
    
    for epoch in range(20):
        # Training
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
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f'Epoch {epoch+1}/20: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, 'models/digit_cnn.pth')
        
        scheduler.step()
    
    # Test accuracy
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