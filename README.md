Handwritten Equation Solver

This project recognizes and solves handwritten arithmetic expressions.

What it does:
- Takes an image of a handwritten math equation (like "3 + 5 × 2") and outputs the calculated result (13).

How it works:
- Segments the equation image into individual symbols using OpenCV
- Classifies each symbol using two separate CNNs:
  - One for digits (0-9)
  - One for operators (+, -, ×, ÷)
- Reconstructs the mathematical expression from classified symbols
- Evaluates the expression to get the final answer

Quickstart
---------
1. Create a Python virtual environment and install dependencies:

   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

2. Train models (optional):
   - `train_digits.py` uses MNIST to train a digit model.
   - `train_operators.py` generates synthetic images to train an operator model.

3. Run the Gradio app:

   python app.py

Files
-----
- `data_gen.py` - Synthetic operator data generator.
- `models.py` - Model architectures and save/load helpers.
- `segment.py` - OpenCV symbol segmentation utilities.
- `predict.py` - Prediction pipeline combining segmentation and classifiers.
- `train_digits.py`, `train_operators.py` - Training scripts.
- `app.py` - Gradio web interface.

Notes
-----
This is a starter implementation with basic models and heuristics. Improvements:
- Better segmentation for touching symbols
- More operator training data and augmentation
- Use sequence models (CTC/attention) for end-to-end recognition
- Support parentheses and multi-digit numbers

License: MIT
