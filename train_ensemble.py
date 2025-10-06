#!/usr/bin/env python3
"""
Train multiple models for ensemble prediction - highest accuracy approach
"""

import torch
from train_digits import train_digits
from train_operators import train_operators
from models import EfficientCNN, ImprovedCNN, save_model
import os

def train_ensemble():
    """Train multiple models with different architectures for ensemble"""
    print("Training ensemble models for maximum accuracy...")
    
    # Train primary models
    print("\n[1/4] Training EfficientCNN models...")
    train_digits()
    train_operators()
    
    # Train secondary models with different architecture
    print("\n[2/4] Training ImprovedCNN models...")
    
    # Modify paths for secondary models
    import train_digits
    import train_operators
    
    # Backup original save calls
    original_digit_save = train_digits.save_model
    original_op_save = train_operators.save_model
    
    # Override save paths for ensemble models
    def save_digit_improved(model, path):
        original_digit_save(model, path.replace('.pth', '_improved.pth'))
    
    def save_op_improved(model, path):
        original_op_save(model, path.replace('.pth', '_improved.pth'))
    
    # Temporarily replace model classes and save functions
    train_digits.EfficientCNN = ImprovedCNN
    train_operators.EfficientCNN = ImprovedCNN
    train_digits.save_model = save_digit_improved
    train_operators.save_model = save_op_improved
    
    try:
        train_digits.train_digits()
        train_operators.train_operators()
    finally:
        # Restore original functions
        train_digits.EfficientCNN = EfficientCNN
        train_operators.EfficientCNN = EfficientCNN
        train_digits.save_model = original_digit_save
        train_operators.save_model = original_op_save
    
    print("\n[3/4] Training with different hyperparameters...")
    # Could add more variations here
    
    print("\n[4/4] Ensemble training complete!")
    print("Available models:")
    print("- models/digit_cnn.pth (EfficientCNN)")
    print("- models/digit_cnn_improved.pth (ImprovedCNN)")
    print("- models/op_cnn.pth (EfficientCNN)")
    print("- models/op_cnn_improved.pth (ImprovedCNN)")

if __name__ == "__main__":
    train_ensemble()