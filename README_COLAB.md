Quick Colab guide — train faster for free

1) Open a new Colab notebook: https://colab.research.google.com/
2) Runtime -> Change runtime type -> Hardware accelerator: GPU (prefer Tesla T4 when available)
3) Mount Drive and clone this repo:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content
!git clone https://github.com/<your-username>/Handwritten-Equation-Solver.git
%cd Handwritten-Equation-Solver
```

4) Install dependencies:

```bash
!pip install -r requirements.txt
# if torch wheel missing GPU support, install matching torch from pytorch.org (select CUDA version)
```

5) Example training commands (use GPU runtime):

```bash
# train digits with AMP and 4 workers, save checkpoints to Drive
python train_digits.py --epochs 10 --batch-size 128 --num-workers 4 --pin-memory --save-path /content/drive/MyDrive/hes/models/digit_cnn.pth

# train operators
python train_operators.py --epochs 12 --batch-size 64 --num-workers 4 --pin-memory --save-path /content/drive/MyDrive/hes/models/op_cnn.pth
```

Notes and tips:
- Mixed precision (AMP) is enabled by default on CUDA; it reduces memory and often speeds training.
- If you see OOM: reduce `--batch-size` or remove `--pin-memory`.
- Save paths pointing to Drive preserve checkpoints across Colab disconnects.
- For better accuracy: increase epochs, enable data augmentation (already in scripts), or pretrain on larger datasets.
