# 🚀 Night Shield - Quick Start Guide

Complete guide to set up, train datasets, and run your surveillance system in minutes!

---

## 📋 Table of Contents
1. [Initial Setup](#initial-setup)
2. [Running the Application](#running-the-application)
3. [Training Datasets](#training-datasets)
4. [Feature Guide](#feature-guide)
5. [Troubleshooting](#troubleshooting)

---

## 🛠️ Initial Setup

### Step 1: Install Python
- Download Python 3.10 or 3.11 from [python.org](https://www.python.org/downloads/)
- ✅ Check "Add Python to PATH" during installation

### Step 2: Create Virtual Environment
Open PowerShell in the project folder:
```powershell
# Navigate to project root
cd "C:\Users\HP\Desktop\Night Surveillance System - Final1"

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```powershell
# Install required packages
pip install ultralytics opencv-python flask python-dotenv psycopg2-binary

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install specific versions for stability
pip install "numpy==1.24.3" "pillow==10.4.0" --force-reinstall --no-deps
```

### Step 3.1 (Recommended): Configure NUNIF Super-Resolution
```powershell
# Clone NUNIF once into your project (skip if folder already exists)
if (-not (Test-Path .\third_party\nunif)) {
  git clone https://github.com/nagadomi/nunif.git .\third_party\nunif
}

# Tell this app to use NUNIF backend
$env:SUPERRES_ENGINE = "nunif"

# Optional: model profile for camera footage (photo/art/art_scan)
$env:NUNIF_MODEL_TYPE = "photo"

# Verify backend and warm up SR models
python -c "import enhancement; print('backend:', enhancement.superres_backend_name()); print('models:', enhancement.ensure_superres_models())"
```

---

## 🎯 Running the Application

### Quick Launch (Default Settings)
```powershell
# Navigate to inner project folder
cd "Night Surveillance System - Final1"

# Run the application
python .\main.py
```

**Access the app**: Open browser → http://127.0.0.1:5000

### Launch with Custom Model
```powershell
# Set environment variables
$env:YOLO_WEIGHTS = "yolov8n.pt"
$env:DISABLE_DETECTION_DB = "1"

# Run the application
python .\main.py
```

### Launch with Anomaly Detection
```powershell
# Configure anomaly detection
$env:YOLO_WEIGHTS = "runs\detect\anomaly_eval15\weights\best.pt"
$env:ENABLE_ANOMALY_ALERTS = "1"
$env:ANOMALY_CLASSES = "person,car,knife,gun,backpack"
$env:ANOMALY_CONF_MIN = "0.5"

# Run the application
python .\main.py
```

### Launch with Recommended Hugging Face Detector (Grounding DINO)
```powershell
# Use Hugging Face zero-shot backend as primary detector
$env:DETECTION_BACKEND = "hf"
$env:HF_DETECTOR_TASK = "zero-shot-object-detection"
$env:HF_OBJECT_DETECTION_MODEL = "IDEA-Research/grounding-dino-base"

# Optional tuning for stricter detections
$env:HF_ZERO_SHOT_THRESHOLD = "0.30"
$env:HF_ZERO_SHOT_POST_CONF_MIN = "0.30"

# Run the application
python .\main.py
```

---

## 📚 Training Datasets

### Overview
Training custom models improves detection accuracy for your specific use case. Follow these steps:

### Method 1: Quick Training (Recommended for Beginners)

#### Train on LOL Low-Light Dataset (15 images - Fast)
```powershell
# Step 1: Register the dataset
python .\scripts\register_dataset_folder.py --name "LOL Eval15 Anomaly" --folder ".\datasets\lol_dataset\eval15\low"

# Step 2: Train the model (takes ~5-10 minutes)
python .\train_pseudolabel_yolo.py --dataset "LOL Eval15 Anomaly" --weights "..\yolov8s.pt" --epochs 10 --batch 4 --imgsz 640 --conf 0.25 --val_split 0.2 --resume-labels --infer-batch 4 --device cpu --label-every 1 --fast-preset --patience 3 --freeze 6 --workers 0 --name anomaly_eval15 --out ".\datasets\prepared\anomaly_eval15_pseudo" --seed 42

# Step 3: Use the trained model
$env:YOLO_WEIGHTS = "runs\detect\anomaly_eval15\weights\best.pt"
python .\main.py
```

#### Train on LOL Full Dataset (485 images - Longer)
```powershell
# Step 1: Register the dataset
python .\scripts\register_dataset_folder.py --name "LOL LowLight Anomaly" --folder ".\datasets\lol_dataset\our485\low"

# Step 2: Train the model (takes 30-60 minutes)
python .\train_pseudolabel_yolo.py --dataset "LOL LowLight Anomaly" --weights "..\yolov8s.pt" --epochs 20 --batch 8 --imgsz 640 --conf 0.25 --val_split 0.15 --resume-labels --infer-batch 8 --device cpu --label-every 2 --fast-preset --patience 5 --freeze 6 --workers 0 --name anomaly_lowlight --out ".\datasets\prepared\anomaly_lowlight_pseudo" --seed 42

# Step 3: Use the trained model
$env:YOLO_WEIGHTS = "runs\detect\anomaly_lowlight\weights\best.pt"
python .\main.py
```

### Method 2: Using PowerShell Helper Functions

#### Load Helper Functions
```powershell
# From project root
. "Night Surveillance System - Final1\Night Surveillance System - Final1\scripts\tasks.ps1"
```

#### Available Commands
```powershell
# Register LOL dataset
Register-LOL

# Train on LOL dataset (fast - 10 epochs)
Start-LOLTrainingFast

# Train on LOL dataset (full - 25 epochs)
Start-LOLTrainingFull

# Register COCO dataset
Register-COCO-Full

# Train on COCO (5000 images)
Start-COCOTraining5K

# Use trained weights
Use-App-Weights -RunName "anomaly_eval15"
```

### Method 3: Custom Dataset Training

#### Register Your Own Dataset
```powershell
python .\scripts\register_dataset_folder.py --name "My Custom Dataset" --folder ".\datasets\my_images"
```

#### Train on Your Dataset
```powershell
python .\train_pseudolabel_yolo.py `
  --dataset "My Custom Dataset" `
  --weights "..\yolov8s.pt" `
  --epochs 15 `
  --batch 8 `
  --imgsz 640 `
  --conf 0.30 `
  --val_split 0.15 `
  --resume-labels `
  --device cpu `
  --fast-preset `
  --workers 0 `
  --name my_custom_model `
  --out ".\datasets\prepared\my_custom_pseudo"
```

### Training Parameters Explained

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `--epochs` | Number of training cycles | 10-20 for small datasets |
| `--batch` | Images processed at once | 4-8 for CPU |
| `--imgsz` | Image size for training | 640 (standard) |
| `--conf` | Detection confidence threshold | 0.25-0.35 |
| `--device` | Hardware to use | `cpu` or `cuda` |
| `--fast-preset` | Speed optimization | Always use for CPU |
| `--patience` | Early stopping patience | 3-5 epochs |
| `--freeze` | Layers to freeze | 6-10 for transfer learning |

---

## 🎨 Feature Guide

### 1. Dashboard
- **Camera Control**: Click "Start Camera" to begin live video feed
- **Statistics**: View cameras, datasets, detections, and events
- **Camera Button**: Manual control - camera won't start automatically

### 2. Low-Light Detection
1. Navigate to **Low-Light Detection** menu
2. **Drag & drop** or **click to browse** for an image
3. Click **"Analyze Image"** button
4. View results with bounding boxes and confidence scores

### 3. Anomaly Detection
- Automatically detects configured anomaly classes (person, car, knife, etc.)
- **Red bounding boxes** = Anomalies detected
- **Green bounding boxes** = Normal detections
- Email alerts sent for anomalies (if configured)
- Anomaly images saved to `static/images/anomalies/`

### 4. Video Upload
- Upload pre-recorded videos for analysis
- Processes frame-by-frame with object detection
- Saves detected objects and events

### 5. Camera Management
- Add/remove surveillance cameras
- Configure camera names and URLs
- Manage multiple camera feeds

---

## 🔧 Configuration

### Environment Variables (.env file)

Create a `.env` file in the project root:

```env
# Email Alerts
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
ALERT_TO=recipient@example.com

# Model Configuration
YOLO_WEIGHTS=yolov8n.pt
DETECTION_BACKEND=hf
HF_DETECTOR_TASK=zero-shot-object-detection
HF_OBJECT_DETECTION_MODEL=IDEA-Research/grounding-dino-base

# Detection Settings
DISABLE_DETECTION_DB=1
IMPORTANT_CLASSES=person,car,truck,motorcycle
ALERT_CONF_MIN=0.6
HF_ZERO_SHOT_THRESHOLD=0.30
HF_ZERO_SHOT_POST_CONF_MIN=0.30
HF_ZERO_SHOT_MAX_LABELS=28

# Anomaly Detection
ENABLE_ANOMALY_ALERTS=1
ANOMALY_CLASSES=person,car,motorcycle,truck,knife,gun,backpack
ANOMALY_CONF_MIN=0.5

# Optional: Supabase Database
SUPABASE_DB_URL=postgresql://user:pass@host:6543/postgres?sslmode=require
```

---

## 🐛 Troubleshooting

### Common Issues & Solutions

#### 1. "No such file or directory" Error
**Problem**: Path issues with nested folders

**Solution**: Always navigate to inner folder first:
```powershell
cd "Night Surveillance System - Final1"
python .\main.py
```

#### 2. Import Errors (numpy, torch, PIL)
**Problem**: Corrupted package installation

**Solution**: Reinstall dependencies:
```powershell
pip uninstall numpy pillow torch torchvision -y
pip install "numpy==1.24.3" "pillow==10.4.0"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### 3. Training Fails or Crashes
**Problem**: Memory issues or incompatible settings

**Solution**: Reduce batch size and image size:
```powershell
python .\train_pseudolabel_yolo.py --batch 4 --imgsz 512 --workers 0 ...
```

#### 4. Camera Won't Start
**Problem**: Camera feed not loading

**Solution**: 
- Click the "Start Camera" button on dashboard
- Check camera permissions in system settings
- Verify camera index in code (default is 0)

#### 5. Detection Too Slow
**Problem**: CPU processing is slow

**Solution**: 
- Use `--fast-preset` flag during training
- Reduce image size: `--imgsz 512`
- Use smaller model: `yolov8n.pt` instead of `yolov8s.pt`

#### 6. Email Alerts Not Working
**Problem**: SMTP configuration issues

**Solution**:
1. Enable "Less Secure App Access" or use "App Password" for Gmail
2. Verify SMTP settings in `.env` file
3. Check `ENABLE_ANOMALY_ALERTS=1` is set

---

## 📞 Quick Reference Commands

### Essential Commands Cheat Sheet

```powershell
# ========== SETUP ==========
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Navigate to project folder
cd "Night Surveillance System - Final1"

# ========== RUN APP ==========
# Basic run
python .\main.py

# Run with anomaly detection
$env:ENABLE_ANOMALY_ALERTS="1"; python .\main.py

# ========== TRAINING ==========
# Quick train (15 images, ~10 min)
python .\scripts\register_dataset_folder.py --name "LOL Eval15 Anomaly" --folder ".\datasets\lol_dataset\eval15\low"
python .\train_pseudolabel_yolo.py --dataset "LOL Eval15 Anomaly" --weights "..\yolov8s.pt" --epochs 10 --batch 4 --imgsz 640 --conf 0.25 --val_split 0.2 --resume-labels --infer-batch 4 --device cpu --label-every 1 --fast-preset --patience 3 --freeze 6 --workers 0 --name anomaly_eval15 --out ".\datasets\prepared\anomaly_eval15_pseudo" --seed 42

# ========== USE TRAINED MODEL ==========
$env:YOLO_WEIGHTS="runs\detect\anomaly_eval15\weights\best.pt"
python .\main.py

# ========== TROUBLESHOOTING ==========
# Fix dependencies
pip install "numpy==1.24.3" "pillow==10.4.0" --force-reinstall --no-deps

# Clear Python cache
python -c "import shutil; shutil.rmtree('__pycache__', ignore_errors=True)"
```

---

## 🎓 Training Tips

### For Best Results:
1. **Start small**: Train on eval15 (15 images) first to verify everything works
2. **Use fast-preset**: Always include `--fast-preset` for CPU training
3. **Monitor progress**: Watch for early stopping - saves time
4. **Save frequently**: Models auto-save to `runs/detect/<name>/weights/best.pt`
5. **Test immediately**: Run app with new model to verify improvements

### Performance Expectations:
- **15 images**: ~5-10 minutes training time
- **485 images**: ~30-60 minutes training time  
- **5000 images**: ~3-5 hours training time

---

## 📖 Additional Resources

- **Main README**: `README.md` - Detailed technical documentation
- **Helper Scripts**: `scripts/tasks.ps1` - PowerShell automation
- **Example Config**: `.env.example` - Configuration template
- **Training Script**: `train_pseudolabel_yolo.py` - Full training options

---

## ✨ Quick Start Summary

**Complete Setup in 5 Commands:**
```powershell
# 1. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install ultralytics opencv-python flask python-dotenv torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install "numpy==1.24.3" "pillow==10.4.0" --force-reinstall --no-deps

# 3. Navigate to project
cd "Night Surveillance System - Final1"

# 4. Run the application
python .\main.py

# 5. Open browser
# http://127.0.0.1:5000
```

**Need Help?** Check the troubleshooting section or review error messages carefully!

---

🎉 **You're ready to go! Happy monitoring!** 🎉
