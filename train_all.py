#!/usr/bin/env python3
"""
Complete training script for both digit and operator models
Optimized for Google Colab with progress tracking
"""

import os
import time
from data_gen import generate_operators
from train_digits import train_digits
from train_operators import train_operators

def main():
    print("=" * 60)
    print("HANDWRITTEN EQUATION SOLVER - COMPLETE TRAINING")
    print("=" * 60)
    
    start_time = time.time()
    
    # Step 1: Generate operator data
    print("\n[1/3] Generating operator training data...")
    try:
        generate_operators(samples_per_class=1000)  # More samples for better accuracy
        print("✓ Operator data generation completed")
    except Exception as e:
        print(f"✗ Error generating operator data: {e}")
        return
    
    # Step 2: Train digit model
    print("\n[2/3] Training digit recognition model...")
    try:
        train_digits()
        print("✓ Digit model training completed")
    except Exception as e:
        print(f"✗ Error training digit model: {e}")
        return
    
    # Step 3: Train operator model
    print("\n[3/3] Training operator recognition model...")
    try:
        train_operators()
        print("✓ Operator model training completed")
    except Exception as e:
        print(f"✗ Error training operator model: {e}")
        return
    
    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Total training time: {total_time/60:.1f} minutes")
    print("\nTrained models saved:")
    print("- models/digit_cnn.pth (digit recognition)")
    print("- models/op_cnn.pth (operator recognition)")
    print("- models/op_classes.json (operator class mapping)")
    print("\nYou can now use the models with app.py or predict.py")
    print("=" * 60)

if __name__ == "__main__":
    main()