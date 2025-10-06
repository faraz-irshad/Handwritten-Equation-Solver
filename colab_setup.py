"""
Google Colab setup script for Handwritten Equation Solver
Run this first in Colab to set up the environment
"""

import os
import subprocess
import sys

def setup_colab():
    """Setup the environment for Google Colab"""
    
    print("Setting up Handwritten Equation Solver for Google Colab...")
    
    # Install required packages
    packages = [
        'torch',
        'torchvision', 
        'opencv-python-headless',
        'pillow',
        'numpy',
        'gradio'
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/operators', exist_ok=True)
    
    print("Setup complete! You can now run the training scripts.")
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU available: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
        else:
            print("No GPU available, using CPU")
    except ImportError:
        print("PyTorch not installed properly")

if __name__ == "__main__":
    setup_colab()