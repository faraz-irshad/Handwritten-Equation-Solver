"""
Train a digit CNN using MNIST from torchvision.
"""
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import SimpleCNN, save_model


def train(save_path='models/digit_cnn.pth', epochs=5, batch_size=64, device='cpu'):
    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
    ])
    trainset = datasets.MNIST(root='data/mnist', train=True, download=True, transform=transform)
    loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    model = SimpleCNN(out_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            opt.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} loss={total_loss/len(loader):.4f}")
    save_path = save_path
    torch.save(model.state_dict(), save_path)
    print("Saved digit model to", save_path)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    train(epochs=3, device=device)
