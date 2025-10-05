"""
Gradio app for interactive testing.
"""
import gradio as gr
from predict import Solver
from PIL import Image
import numpy as np

solver = Solver(digit_model_path='models/digit_cnn.pth', op_model_path='models/op_cnn.pth', device='cpu')


def infer(img):
    if isinstance(img, Image.Image):
        arr = np.array(img.convert('L'))
    else:
        arr = img
    expr, result = solver.predict_image(arr)
    return expr, str(result)

iface = gr.Interface(fn=infer, inputs=gr.Image(type='pil', label='Handwritten equation'), outputs=[gr.Textbox(label='Expression'), gr.Textbox(label='Result')], title='Handwritten Equation Solver')

if __name__ == '__main__':
    iface.launch()
