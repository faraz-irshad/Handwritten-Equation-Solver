# ⚡ Quick Start Guide - Google Colab

## 🎯 Option 1: Upload Project (Recommended)

### Step 1: Prepare Project
1. **Download/Clone** this project locally
2. **Create ZIP file** of the entire project folder
3. **Name it**: `handwritten-equation-solver.zip`

### Step 2: Colab Setup (2 minutes)
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. **Runtime** → **Change runtime type** → **GPU** → **Save**
3. **Upload** the notebook: `colab_train.ipynb`

### Step 3: Run Training (15-20 minutes)
1. **Run each cell** in sequence (Ctrl+Enter)
2. **Upload ZIP** when prompted in Step 2
3. **Wait for training** to complete
4. **Download models** in final step

---

## 🛠️ Option 2: Create from Scratch

### If you don't have project files:

#### Step 1: Setup Environment
```python
# Run in first Colab cell
!pip install torch torchvision opencv-python-headless pillow numpy scipy scikit-image gradio requests torchmetrics

# Check GPU
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
```

#### Step 2: Create Project Files
```python
# Run this to create all files
exec(open('https://raw.githubusercontent.com/your-repo/colab_inline_setup.py').read())
# OR copy-paste the colab_inline_setup.py content
```

#### Step 3: Train Models
```python
# Generate data
exec(open('data_gen.py').read())
generate_operators(1000)

# Train models
exec(open('train_digits.py').read())
train_digits()

exec(open('train_operators.py').read())
train_operators()
```

#### Step 4: Test & Download
```python
# Test
exec(open('predict.py').read())
solver = Solver(device='cuda')

# Launch interface
exec(open('app.py').read())

# Download models
from google.colab import files
files.download('models/digit_cnn.pth')
files.download('models/op_cnn.pth')
```

---

## 🚨 Troubleshooting

### GPU Not Available?
```python
# Check runtime type
!nvidia-smi
# If fails: Runtime → Change runtime type → GPU
```

### Training Fails?
```python
# Reduce batch size
# In train files, change: batch_size=128 → batch_size=64
```

### Out of Memory?
```python
# Restart runtime
# Runtime → Restart runtime
# Re-run from beginning
```

---

## ⏱️ Expected Timeline

| Step | Time (GPU) | Time (CPU) |
|------|------------|------------|
| Setup | 2 min | 2 min |
| Data Gen | 3 min | 8 min |
| Digit Training | 8 min | 30 min |
| Operator Training | 5 min | 15 min |
| Testing | 2 min | 2 min |
| **Total** | **20 min** | **57 min** |

---

## 🎯 Success Indicators

✅ **GPU detected**: `CUDA available: True`  
✅ **Data generated**: `Generated 1000 samples per class`  
✅ **Training progress**: `Epoch 20/20: Val Acc: 98.45%`  
✅ **Models saved**: `Digit/Operator model saved!`  
✅ **Interface works**: Public URL generated  
✅ **Download works**: ZIP file downloaded  

---

## 📞 Need Help?

1. **Check GPU**: Must show `Tesla T4` or similar
2. **Check files**: `!ls models/` should show `.pth` files
3. **Check errors**: Read error messages carefully
4. **Restart**: Runtime → Restart runtime if stuck

**Most issues are solved by ensuring GPU is enabled and restarting runtime.**