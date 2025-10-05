"""
Model architectures and helpers for digits and operators.
Simple CNNs using PyTorch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, out_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Calculate correct size: 64x64 input -> conv1 -> conv2 -> pool -> 64x32x32
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, out_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model_class, path, device='cpu', out_classes=None):
    """Load a model state dict into model_class(out_classes=out_classes).

    If the path does not exist, raises FileNotFoundError.
    """
    if out_classes is None:
        # default constructor
        model = model_class()
    else:
        model = model_class(out_classes=out_classes)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
