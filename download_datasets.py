#!/usr/bin/env python3
"""
Download and prepare better datasets for handwritten equation recognition
"""

import os
import requests
import zipfile
from pathlib import Path
import torch
from torchvision import datasets

def download_emnist():
    """Download EMNIST dataset (better than MNIST for handwritten digits)"""
    print("Downloading EMNIST dataset...")
    try:
        # This will download EMNIST automatically when used
        datasets.EMNIST('data/emnist', split='digits', train=True, download=True)
        datasets.EMNIST('data/emnist', split='digits', train=False, download=True)
        print("✓ EMNIST downloaded successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to download EMNIST: {e}")
        return False

def download_hasy():
    """Download HASYv2 dataset for mathematical symbols"""
    print("Downloading HASYv2 dataset...")
    try:
        url = "https://zenodo.org/record/259444/files/HASYv2.zip"
        data_dir = Path("data/hasy")
        data_dir.mkdir(exist_ok=True, parents=True)
        
        zip_path = data_dir / "HASYv2.zip"
        
        if not zip_path.exists():
            print("Downloading HASYv2...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        # Extract if not already extracted
        extract_dir = data_dir / "HASYv2"
        if not extract_dir.exists():
            print("Extracting HASYv2...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
        
        print("✓ HASYv2 downloaded and extracted successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to download HASYv2: {e}")
        return False

def setup_datasets():
    """Setup all recommended datasets"""
    print("Setting up improved datasets for handwritten equation recognition...")
    
    success_count = 0
    
    # Download EMNIST (replaces MNIST)
    if download_emnist():
        success_count += 1
    
    # Download HASYv2 (for mathematical symbols)
    if download_hasy():
        success_count += 1
    
    print(f"\nDataset setup complete: {success_count}/2 datasets ready")
    
    if success_count >= 1:
        print("✓ You can now train with improved datasets!")
        print("Run: python train_all.py")
    else:
        print("⚠️  Dataset download failed. Will use fallback datasets.")

if __name__ == "__main__":
    setup_datasets()