# 🚀 Complete Google Colab Setup Guide

## 📋 Prerequisites

### Before Starting:
1. **Google Account** - Required for Google Colab access
2. **Google Drive** (Optional) - For saving models permanently
3. **Project Files** - Either as ZIP file or GitHub repository

### Recommended Setup:
- **GPU Runtime** - Significantly faster training (15-20 min vs 60+ min)
- **High RAM** - For larger batch sizes and better performance

---

## 🎯 Step-by-Step Instructions

### **Phase 1: Environment Setup (5 minutes)**

#### 1.1 Access Google Colab
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Sign in with your Google account
3. Click **"New notebook"** or **"Upload"** if you have the notebook file

#### 1.2 Enable GPU (CRITICAL for speed)
1. Click **Runtime** → **Change runtime type**
2. Set **Hardware accelerator** to **"GPU"**
3. Set **Runtime shape** to **"High-RAM"** (if available)
4. Click **"Save"**

#### 1.3 Verify GPU Access
```python
# Run this cell to check GPU
!nvidia-smi
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
```

**Expected Output:**
```
CUDA available: True
GPU: Tesla T4 (or similar)
```

---

### **Phase 2: Project Upload (3 minutes)**

#### Option A: Upload ZIP File (Recommended)
1. **Prepare your project:**
   - Create ZIP file containing all project files
   - Include: `*.py`, `requirements.txt`, `README.md`

2. **Upload to Colab:**
   ```python
   from google.colab import files
   uploaded = files.upload()
   ```
   - Click **"Choose Files"**
   - Select your project ZIP
   - Wait for upload completion

3. **Extract files:**
   ```python
   import zipfile
   with zipfile.ZipFile('your-project.zip', 'r') as zip_ref:
       zip_ref.extractall('.')
   ```

#### Option B: Clone from GitHub
```python
!git clone https://github.com/your-username/handwritten-equation-solver.git
%cd handwritten-equation-solver
```

---

### **Phase 3: Dependencies Installation (2 minutes)**

#### 3.1 Install Required Packages
```python
!pip install torch torchvision opencv-python-headless pillow numpy scipy scikit-image gradio requests torchmetrics
```

#### 3.2 Verify Installation
```python
import cv2, numpy as np, PIL, torch, torchvision
print('✅ All packages installed successfully')
```

---

### **Phase 4: Training Process (15-60 minutes)**

#### 4.1 Generate Training Data (2-3 minutes)
```python
%%time
# Generate synthetic operator data
exec(open('data_gen.py').read())
generate_operators(samples_per_class=1000)
```

**Expected Output:**
```
Generated 1000 realistic handwritten-style samples per operator class
CPU times: user 45.2 s, sys: 2.1 s, total: 47.3 s
```

#### 4.2 Train Digit Model (8-25 minutes)
```python
%%time
# Train on EMNIST dataset
exec(open('train_digits.py').read())
train_digits()
```

**Expected Progress:**
```
Using EMNIST dataset (handwritten digits)
Training on cuda
Epoch 1/20: Train Acc: 85.23%, Val Acc: 87.45%
Epoch 2/20: Train Acc: 91.67%, Val Acc: 92.18%
...
Epoch 20/20: Train Acc: 98.45%, Val Acc: 97.89%
Final Test Accuracy: 98.12%
```

#### 4.3 Train Operator Model (5-15 minutes)
```python
%%time
# Train operator recognition
exec(open('train_operators.py').read())
train_operators()
```

**Expected Progress:**
```
Training on cuda
Epoch 1/15: Train Acc: 78.34%, Val Acc: 82.15%
...
Epoch 15/15: Train Acc: 96.78%, Val Acc: 95.23%
Best Validation Accuracy: 95.23%
```

---

### **Phase 5: Testing & Validation (2 minutes)**

#### 5.1 Test Models
```python
# Load and test solver
exec(open('predict.py').read())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
solver = Solver(device=device)

# Test with sample equations
test_equations = ['2+3', '7-4', '5×2', '8÷2']
for eq in test_equations:
    test_img = create_test_image(eq)
    expression, result = solver.predict_image(test_img)
    print(f'Input: {eq} → Detected: {expression} → Result: {result}')
```

**Expected Output:**
```
Input: 2+3 → Detected: 2+3 (conf: 0.96) → Result: 5
Input: 7-4 → Detected: 7-4 (conf: 0.94) → Result: 3
Input: 5×2 → Detected: 5×2 (conf: 0.92) → Result: 10
Input: 8÷2 → Detected: 8÷2 (conf: 0.89) → Result: 4.0
```

---

### **Phase 6: Web Interface (1 minute)**

#### 6.1 Launch Gradio Interface
```python
# Launch interactive web interface
exec(open('app.py').read())
```

**Expected Output:**
```
Running on public URL: https://xxxxx.gradio.live
```

#### 6.2 Test Interface
1. Click the public URL
2. Upload handwritten equation image
3. View detection results
4. Share URL with others for testing

---

### **Phase 7: Download Models (1 minute)**

#### 7.1 Check Available Models
```python
import os
for file in os.listdir('models'):
    if file.endswith(('.pth', '.json')):
        size = os.path.getsize(f'models/{file}') / (1024*1024)
        print(f'{file} ({size:.1f} MB)')
```

#### 7.2 Download Models
```python
from google.colab import files
import zipfile

# Create ZIP of all models
with zipfile.ZipFile('trained_models.zip', 'w') as zipf:
    for file in ['models/digit_cnn.pth', 'models/op_cnn.pth', 'models/op_classes.json']:
        if os.path.exists(file):
            zipf.write(file, os.path.basename(file))

# Download ZIP
files.download('trained_models.zip')
```

---

## 🔧 Troubleshooting

### Common Issues & Solutions:

#### **Issue 1: GPU Not Available**
```
RuntimeError: CUDA out of memory
```
**Solution:**
- Restart runtime: **Runtime** → **Restart runtime**
- Reduce batch size in training scripts
- Use CPU training (slower but works)

#### **Issue 2: Package Installation Fails**
```
ERROR: Could not install packages
```
**Solution:**
```python
!pip install --upgrade pip
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### **Issue 3: File Upload Fails**
```
Upload widget is only available when the cell has been executed
```
**Solution:**
- Run the upload cell again
- Try smaller file sizes (<25MB)
- Use GitHub clone instead

#### **Issue 4: Training Stops/Crashes**
```
Session crashed due to OOM
```
**Solution:**
- Enable High-RAM runtime
- Reduce `batch_size` in training scripts
- Restart and resume training

#### **Issue 5: Models Don't Load**
```
FileNotFoundError: models/digit_cnn.pth
```
**Solution:**
- Check if training completed successfully
- Verify model files exist: `!ls models/`
- Re-run training if files missing

---

## 📊 Performance Expectations

### Training Times:
| Component | GPU (T4) | CPU |
|-----------|----------|-----|
| Data Generation | 2-3 min | 5-8 min |
| Digit Training | 8-12 min | 25-40 min |
| Operator Training | 3-6 min | 10-20 min |
| **Total** | **15-20 min** | **45-70 min** |

### Expected Accuracy:
| Model | Accuracy | Confidence |
|-------|----------|------------|
| Digit Recognition | 98-99% | High |
| Operator Recognition | 95-98% | High |
| Overall System | 92-98% | High |

---

## 💾 Saving to Google Drive (Optional)

### Mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')

# Copy models to Drive
!cp -r models/ /content/drive/MyDrive/handwritten_equation_models/
```

### Load from Drive Later:
```python
# In new session
from google.colab import drive
drive.mount('/content/drive')
!cp -r /content/drive/MyDrive/handwritten_equation_models/ models/
```

---

## 🚀 Next Steps After Training

### Local Usage:
1. Download `trained_models.zip`
2. Extract to your local project
3. Run: `python app.py`
4. Open browser to `http://localhost:7860`

### Integration:
```python
from predict import Solver
solver = Solver(device='cpu')  # or 'cuda'
expression, result = solver.predict_image(your_image)
```

### Production Deployment:
- Use Gradio's sharing feature
- Deploy to Hugging Face Spaces
- Integrate into web applications
- Create mobile app with the models

---

## 📞 Support

### If You Need Help:
1. **Check the troubleshooting section above**
2. **Verify all steps were followed correctly**
3. **Check Colab runtime status**
4. **Try restarting runtime and re-running**

### Common Success Indicators:
- ✅ GPU detected and available
- ✅ All packages installed without errors
- ✅ Training shows increasing accuracy
- ✅ Test predictions are correct
- ✅ Models download successfully

**Total Setup Time: 20-25 minutes with GPU, 60-90 minutes with CPU**