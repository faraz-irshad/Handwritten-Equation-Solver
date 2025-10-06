"""
Train a digit CNN using MNIST from torchvision.
"""
import argparse
import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import SimpleCNN, default_training_setup


def train(save_path='models/digit_cnn.pth', epochs=5, batch_size=64, device='cpu', num_workers=4, pin_memory=True, use_amp=True):
    default_training_setup()
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.05))
    ])
    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    trainset = datasets.MNIST(root='data/mnist', train=True, download=True, transform=train_transform)
    loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    model = SimpleCNN(out_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.startswith('cuda'))

    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
    start_epoch = 0
    # resume support
    if os.path.isdir(save_path) or save_path.endswith('/'):
        os.makedirs(save_path, exist_ok=True)
    if os.path.isfile(save_path + '.pt'):
        ckpt = torch.load(save_path + '.pt', map_location=device)
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt.get('opt', opt.state_dict()))
        scaler.load_state_dict(ckpt.get('scaler', scaler.state_dict()))
        start_epoch = ckpt.get('epoch', 0) + 1

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp and device.startswith('cuda')):
                out = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} loss={total_loss/len(loader):.4f}")
        # scheduler step
        try:
            scheduler.step()
        except Exception:
            pass

        # checkpoint
        ckpt_dir = os.path.dirname(save_path) or '.'
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'opt': opt.state_dict(),
            'scaler': scaler.state_dict(),
        }
        torch.save(ckpt, os.path.join(ckpt_dir, f'digit_checkpoint_epoch{epoch+1}.pt'))

    # final save
    torch.save(model.state_dict(), save_path)
    print("Saved digit model to", save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--pin-memory', action='store_true')
    parser.add_argument('--no-amp', dest='use_amp', action='store_false')
    parser.add_argument('--save-path', type=str, default='models/digit_cnn.pth')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    train(save_path=args.save_path, epochs=args.epochs, batch_size=args.batch_size, device=device, num_workers=args.num_workers, pin_memory=args.pin_memory, use_amp=args.use_amp)
