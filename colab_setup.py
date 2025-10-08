import os

def setup_colab():
    print("Setting up directories...")
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/operators', exist_ok=True)
    
    try:
        import torch
        print(f"GPU: {torch.cuda.is_available()}")
    except:
        print("Install: !pip install torch torchvision opencv-python-headless pillow numpy gradio")

if __name__ == "__main__":
    setup_colab()