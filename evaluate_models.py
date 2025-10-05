"""Evaluate saved digit and operator models on available datasets.
Prints accuracy for MNIST test set (digits) and the synthetic operator dataset.
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from train_operators import OperatorDataset
from models import SimpleCNN, load_model


def eval_digits(model_path, device='cpu'):
    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
    ])
    testset = datasets.MNIST(root='data/mnist', train=False, download=False, transform=transform)
    loader = DataLoader(testset, batch_size=128)
    model = load_model(SimpleCNN, model_path, device=device, out_classes=10)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            out = model(imgs)
            preds = out.argmax(dim=1).cpu()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total if total>0 else 0.0
    return acc, total


def eval_ops(model_path, device='cpu'):
    ds = OperatorDataset()
    loader = DataLoader(ds, batch_size=128)
    model = load_model(SimpleCNN, model_path, device=device, out_classes=len(ds.classes))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            out = model(imgs)
            preds = out.argmax(dim=1).cpu()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total if total>0 else 0.0
    return acc, total, ds.classes


if __name__ == '__main__':
    device = 'cpu'
    print('Evaluating digit model...')
    try:
        acc_d, n_d = eval_digits('models/digit_cnn.pth', device=device)
        print(f'Digit accuracy: {acc_d*100:.2f}% on {n_d} samples')
    except Exception as e:
        print('Digit evaluation failed:', e)

    print('\nEvaluating operator model...')
    try:
        acc_o, n_o, classes = eval_ops('models/op_cnn.pth', device=device)
        print(f'Operator accuracy: {acc_o*100:.2f}% on {n_o} samples')
        print('Operator classes (index -> label):')
        for i,c in enumerate(classes):
            print(i, c)
    except Exception as e:
        print('Operator evaluation failed:', e)
