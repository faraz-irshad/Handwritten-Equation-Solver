"""
Model architectures and helpers for digits and operators.
Simple CNNs using PyTorch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """A small, efficient CNN suitable for 64x64 grayscale inputs.

    This version reduces parameter count and works better on Colab GPUs.
    """
    def __init__(self, out_classes=10, in_channels=1):
        super().__init__()
        # Use smaller channel growth and depthwise separable style blocks
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32x32

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64x16x16
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, out_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


def default_training_setup(seed: int = 42):
    """Apply fast/deterministic defaults for training on Colab/GPU.

    - sets cudnn.benchmark True for fixed-size inputs
    - seeds torch/numpy for reproducibility
    """
    import random
    torch.manual_seed(seed)
    random.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:
        pass
    torch.backends.cudnn.benchmark = True
    return


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
