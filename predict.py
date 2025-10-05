"""
Prediction pipeline: load models, segment image, classify symbols, reconstruct expression.
"""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from models import SimpleCNN, load_model
from segment import segment_image
import json

DIGIT_LABELS = [str(i) for i in range(10)]
OP_LABELS = None


def _load_op_labels():
    global OP_LABELS
    p = Path('models') / 'op_classes.json'
    if p.exists():
        try:
            OP_LABELS = json.loads(p.read_text())
        except Exception:
            OP_LABELS = ['+', '-', '×', '÷']
    else:
        OP_LABELS = ['+', '-', '×', '÷']


_load_op_labels()


def preprocess_symbol(img_arr, size=64):
    # expects grayscale numpy arr
    pil = Image.fromarray(img_arr)
    pil = pil.resize((64,64)).convert('L')
    arr = np.array(pil).astype(np.float32) / 255.0
    # Keep same polarity as training (ToTensor() makes foreground dark -> small values)
    # Ensure shape (1,1,H,W) and dtype float32
    arr = arr[np.newaxis, np.newaxis, ...]
    return torch.from_numpy(arr).float()


class Solver:
    def __init__(self, digit_model_path=None, op_model_path=None, device='cpu'):
        self.device = device
        if digit_model_path:
            self.digit_model = load_model(SimpleCNN, digit_model_path, device=device, out_classes=10)
        else:
            self.digit_model = None
        if op_model_path:
            # load operator classes mapping if available
            try:
                op_classes = OP_LABELS or ['+', '-', '×', '÷']
            except Exception:
                op_classes = ['+', '-', '×', '÷']
            self.op_model = load_model(SimpleCNN, op_model_path, device=device, out_classes=len(op_classes))
        else:
            self.op_model = None

    def predict_image(self, image):
        # image: numpy array or PIL
        if isinstance(image, Image.Image):
            image = np.array(image.convert('L'))
        symbols = segment_image(image)
        expr = ''
        for sym in symbols:
            t = preprocess_symbol(sym).to(self.device)
            with torch.no_grad():
                # run both models and pick by simple heuristic: operator if op model confidence > 0.6
                op_scores = None
                if self.op_model:
                    out = self.op_model(t)
                    probs = F.softmax(out, dim=1).cpu().numpy()[0]
                    op_scores = probs.max()
                    op_label = OP_LABELS[probs.argmax()]
                digit_label = None
                if self.digit_model:
                    out2 = self.digit_model(t)
                    probs2 = F.softmax(out2, dim=1).cpu().numpy()[0]
                    digit_label = DIGIT_LABELS[probs2.argmax()]
                # Conservative routing: prefer digit unless operator is very confident
                digit_conf = 0.0
                if digit_label is not None:
                    # find digit prob
                    digit_conf = float(probs2.max())
                op_conf = float(op_scores) if op_scores is not None else 0.0
                if op_conf > 0.80 and (op_conf - digit_conf) > 0.25:
                    expr += op_label
                else:
                    expr += digit_label if digit_label is not None else '?'
        # Map '×' and '÷' to Python operators
        expr_py = expr.replace('×', '*').replace('÷', '/')
        try:
            result = eval(expr_py)
        except Exception:
            result = None
        return expr, result
