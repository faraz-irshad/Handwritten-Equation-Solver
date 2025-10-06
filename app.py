import gradio as gr
from predict import Solver
from PIL import Image
import numpy as np
import torch
import os

def solve_equation(img):
    if img is None:
        return "Please upload an image", ""
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        solver = Solver(device=device)
        
        if isinstance(img, Image.Image):
            arr = np.array(img.convert('L'))
        else:
            arr = img
        
        expression, result = solver.predict_image(arr)
        return expression, str(result) if result is not None else "Could not evaluate"
    except Exception as e:
        return f"Error: {str(e)}", "Make sure models are trained"

def check_models():
    digit_model = os.path.exists('models/digit_cnn.pth')
    op_model = os.path.exists('models/op_cnn.pth')
    
    if not (digit_model and op_model):
        return "⚠️ Models not found. Run 'python train_all.py' first."
    return "✅ Models loaded successfully"

interface = gr.Interface(
    fn=solve_equation,
    inputs=gr.Image(type='pil', label='Upload handwritten equation'),
    outputs=[
        gr.Textbox(label='Detected Expression'),
        gr.Textbox(label='Result')
    ],
    title='🧮 Handwritten Equation Solver',
    description='Upload an image of handwritten arithmetic (digits and +, -, ×, ÷)',
    article=check_models()
)

if __name__ == '__main__':
    interface.launch(share=True)  # Enable sharing for Colab