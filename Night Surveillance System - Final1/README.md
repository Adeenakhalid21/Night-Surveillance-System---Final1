````markdown
# Night Surveillance System – Quick Start + Training Guide (Windows)

This guide shows how to set up the environment, train on more data (small → full), and run the web app with your trained YOLOv8 weights. Commands are for Windows PowerShell 5.1.

Note: Commands below assume you run them from the repository root:
`C:\Users\HP\Desktop\Night Surveillance System - Final1`. If you run from this inner folder, prepend `Push-Location ..` first or adjust paths accordingly.

## Prerequisites
- Python 3.10 or 3.11 (64-bit)
- Windows PowerShell (default on your system)
- Optional: Supabase Postgres URL if you want the app to use Postgres instead of local SQLite

## 1) Create and activate a virtual environment
```powershell
# From repo root: C:\Users\HP\Desktop\Night Surveillance System - Final1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 2) Install dependencies
Ultralytics + Flask + OpenCV + dotenv + Postgres client. Install PyTorch separately for your platform.

```powershell
pip install ultralytics opencv-python flask python-dotenv psycopg2-binary
# CPU-only PyTorch (recommended for Windows CPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

If you have a CUDA GPU and correct drivers, install the matching CUDA wheel from PyTorch.org instead of the CPU wheel.

## 3) Datasets expected by the training script
The training script reads image paths from the local SQLite database `night_surveillance.db` (in repo root). It expects tables `datasets` and `training_data` to be populated. Example dataset names include:
- `COCO Train2017 Sample` (small sample)
- Your custom datasets you’ve registered into `datasets`/`training_data`

Tips:
- If image paths moved, use `--path-repair-from` and `--path-repair-to` (see below). 
- You can sanity-check datasets with the SQL helper in `scripts/run_sql.py`.

List datasets with row counts (optional):
```powershell
python "Night Surveillance System - Final1\Night Surveillance System - Final1\scripts\run_sql.py" "SELECT d.dataset_name, COUNT(*) as n FROM datasets d JOIN training_data t ON d.dataset_id=t.dataset_id GROUP BY d.dataset_name ORDER BY n DESC;"
```

## 4) Train with pseudo-labels + fine-tune YOLOv8
The training script is in the inner folder. From the repo root, use:

Small test run (fast, CPU friendly):
```powershell
python "Night Surveillance System - Final1\Night Surveillance System - Final1\train_pseudolabel_yolo.py" `
  --dataset "COCO Train2017 Sample" `
  --weights "Night Surveillance System - Final1\yolov8s.pt" `
  --epochs 5 --batch 8 --imgsz 512 --conf 0.40 `
  --val_split 0.1 --resume-labels --infer-batch 8 `
  --device cpu --label-every 3 --fast-preset --patience 3 --freeze 10 `
  --workers 0 --name coco_quick_test `
  --out "Night Surveillance System - Final1\datasets\prepared\coco_quick_test" `
  --seed 42
```

5K subset run (balanced speed/quality on CPU):
```powershell
python "Night Surveillance System - Final1\Night Surveillance System - Final1\train_pseudolabel_yolo.py" `
  --dataset "COCO Train2017 Sample" `
  --weights "Night Surveillance System - Final1\yolov8s.pt" `
  --epochs 15 --batch 16 --imgsz 640 --conf 0.35 `
  --val_split 0.1 --resume-labels --infer-batch 8 `
  --device cpu --label-every 3 --fast-preset --cos-lr --patience 3 --freeze 10 `
  --workers 0 --name coco_sample_fast `
  --max-images 5000 `
  --out "Night Surveillance System - Final1\datasets\prepared\coco_sample_pseudo" `
  --seed 42
```

Full run (all images; remove `--fast-preset` or switch to GPU to go faster):
```powershell
python "Night Surveillance System - Final1\Night Surveillance System - Final1\train_pseudolabel_yolo.py" `
  --dataset "COCO Train2017 Sample" `
  --weights "Night Surveillance System - Final1\yolov8s.pt" `
  --epochs 30 --batch 16 --imgsz 640 --conf 0.35 `
  --val_split 0.1 --resume-labels --infer-batch 8 `
  --device cpu --label-every 0 --cos-lr --patience 5 --freeze 0 `
  --workers 0 --name coco_full_run `
  --max-images 0 `
  --out "Night Surveillance System - Final1\datasets\prepared\coco_full_pseudo" `
  --seed 42
```

Notes:
- `--resume-labels` skips generating labels if a `.txt` already exists.
- `--label-every 3` sparsely labels every 3rd image on the first pass; set `0` to label all.
- Use `--path-repair-from` and `--path-repair-to` if stored paths have moved. Example:
  ```powershell
  --path-repair-from "D:/old_root/" --path-repair-to "C:/Users/HP/Desktop/Night Surveillance System - Final1/"
  ```
- If you must skip missing paths instead of failing: add `--allow-missing`.

Where outputs go:
- Prepared dataset: under your `--out` path
- Training artifacts: Ultralytics saves to `runs/detect/<name>/` with weights in `weights/best.pt`

## 5) Run the web app with trained weights
The app lazy-loads YOLOv8. Point it to your trained weights via `YOLO_WEIGHTS`. By default it tries `runs/detect/coco_sample_pseudo/weights/best.pt`, else falls back to `yolov8n.pt`.

SQLite (default):
```powershell
Push-Location "Night Surveillance System - Final1"
$env:YOLO_WEIGHTS = "runs\detect\coco_sample_fast\weights\best.pt"  # update to your run name
$env:DISABLE_DETECTION_DB = "1"  # keep DB lean; snapshots still save to static/images
python .\main.py
Pop-Location
```

Supabase Postgres (optional): set the pooled URL in the same shell before launch:
```powershell
# Example format; replace with your pooled URL and ensure sslmode=require
$env:SUPABASE_DB_URL = "postgresql://<user>:<password>@<pooled-hostname>:6543/postgres?sslmode=require"
Push-Location "Night Surveillance System - Final1"
$env:YOLO_WEIGHTS = "runs\detect\coco_sample_fast\weights\best.pt"
$env:DISABLE_DETECTION_DB = "1"
python .\main.py
Pop-Location
```

Hints:
- The app quotes the reserved table name `user` for Postgres automatically.
- Detection rows are disabled by default (`DISABLE_DETECTION_DB=1`), but timestamped snapshots land in `static/images/` and are git‑ignored.

## 6) SQL helper (works with SQLite or Supabase)
Safely run SQL without PowerShell quoting issues:
```powershell
python "Night Surveillance System - Final1\Night Surveillance System - Final1\scripts\run_sql.py" "SELECT 1;"
```
Examples:
```powershell
# List datasets with counts
python "Night Surveillance System - Final1\Night Surveillance System - Final1\scripts\run_sql.py" `
  "SELECT d.dataset_name, COUNT(*) as n FROM datasets d JOIN training_data t ON d.dataset_id=t.dataset_id GROUP BY d.dataset_name ORDER BY n DESC;"

# Basic users check
python "Night Surveillance System - Final1\Night Surveillance System - Final1\scripts\run_sql.py" "SELECT email, firstname FROM user LIMIT 5;"
```

## 7) Troubleshooting
- PowerShell path issues: If you see the folder repeated in the error path, call the script like this from repo root:
  ```powershell
  python "Night Surveillance System - Final1\Night Surveillance System - Final1\train_pseudolabel_yolo.py" --help
  ```
  Or change into the inner folder first:
  ```powershell
  Push-Location "Night Surveillance System - Final1"; python .\train_pseudolabel_yolo.py --help; Pop-Location
  ```
- Torch import or install problems: Use the CPU wheel in step 2 and ensure Python 3.10/3.11.
- Slow CPU runs: add `--fast-preset` and keep `--workers 0` on Windows.
- No images found: confirm dataset names, run the SQL helper to list datasets, use `--path-repair-*` or `--allow-missing`.

## 8) Handy one-liners
Single-line command equivalent for the 5K subset:
```powershell
python "Night Surveillance System - Final1\Night Surveillance System - Final1\train_pseudolabel_yolo.py" --dataset "COCO Train2017 Sample" --weights "Night Surveillance System - Final1\yolov8s.pt" --epochs 15 --batch 16 --imgsz 640 --conf 0.35 --val_split 0.1 --resume-labels --infer-batch 8 --device cpu --label-every 3 --fast-preset --cos-lr --patience 3 --freeze 10 --workers 0 --name coco_sample_fast --max-images 5000 --out "Night Surveillance System - Final1\datasets\prepared\coco_sample_pseudo" --seed 42
```

That's it — train more data, then point the app to your new `best.pt` via `YOLO_WEIGHTS` and run! If you want, I can also add a `requirements.txt` and a small script to set env vars automatically.

---

## 9) New Features: Low-Light Detection & Anomaly Alerts

### Low-Light Image Upload & Detection
Upload low-light images for object detection via a web interface:

1. **Start the app** (see section 5)
2. **Navigate to** http://127.0.0.1:5000/lowlight_detection
3. **Upload an image**:
   - Drag & drop a JPG/PNG/BMP image, or click to browse
   - The backend enhances the image using `enhancement.py`
   - Runs YOLOv8 detection and returns results with bounding boxes
4. **View results**: Original and detected images displayed side-by-side with confidence scores

Features:
- Drag-and-drop upload interface
- Automatic low-light image enhancement before detection
- JSON API response with detection coordinates and class names
- Uploaded images saved to `static/lowlight_uploads/`

### Anomaly Detection System
Detect and alert on anomalies in video streams:

**Step 1: Train Anomaly Models** (optional; or use existing weights with class filtering)
```powershell
# Automated training script registers datasets and trains two models
.\.venv\Scripts\python.exe "Night Surveillance System - Final1\Night Surveillance System - Final1\scripts\train_anomaly_model.py"
```

This will:
- Register LOL low-light dataset (`lol_dataset/our485/low`) as "LOL LowLight Anomaly"
- Register COCO train2017 (max 10K images) as "COCO Anomaly Patterns"
- Train `anomaly_lowlight` model: 20 epochs, conf 0.25, optimized for low-light conditions
- Train `anomaly_general` model: 15 epochs, conf 0.30, for general anomaly patterns
- Output models to `runs/detect/anomaly_lowlight/weights/best.pt` and `runs/detect/anomaly_general/weights/best.pt`

**Step 2: Configure Anomaly Detection**
Set environment variables before starting the app:
```powershell
# Use anomaly-trained model (or any COCO model with class filtering)
$env:YOLO_WEIGHTS = "runs\detect\anomaly_lowlight\weights\best.pt"

# Enable anomaly alerts
$env:ENABLE_ANOMALY_ALERTS = "1"

# Define anomaly classes to detect (comma-separated)
$env:ANOMALY_CLASSES = "person,car,motorcycle,truck,knife,gun,backpack"

# Minimum confidence threshold for anomaly detection
$env:ANOMALY_CONF_MIN = "0.5"
```

**Step 3: Configure Email Alerts** (create `.env` file)
```env
# SMTP Configuration for Email Alerts
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
ALERT_TO=recipient@example.com

# Anomaly Detection Configuration
ENABLE_ANOMALY_ALERTS=1
ANOMALY_CLASSES=person,car,motorcycle,truck,knife,gun,backpack
ANOMALY_CONF_MIN=0.5
```

**Step 4: Run the App**
```powershell
Push-Location "Night Surveillance System - Final1"
python .\main.py
Pop-Location
```

**How Anomaly Detection Works**:
- Video streams are processed frame-by-frame
- Detected objects matching `ANOMALY_CLASSES` with confidence ≥ `ANOMALY_CONF_MIN` trigger anomaly alerts
- Anomalies are marked with **red bounding boxes** (normal detections use green)
- Anomaly frames are saved to `static/images/anomalies/` with timestamps
- Events logged to `surveillance_events` table with severity='high', event_type='anomaly_detected'
- Email alerts sent to configured recipients (if `ENABLE_ANOMALY_ALERTS=1`)

**Testing Without Training Custom Models**:
You can test anomaly detection immediately using existing COCO-trained weights by just setting the anomaly classes:
```powershell
$env:YOLO_WEIGHTS = "yolov8n.pt"  # or your existing trained model
$env:ANOMALY_CLASSES = "person,knife,gun"
$env:ANOMALY_CONF_MIN = "0.6"
$env:ENABLE_ANOMALY_ALERTS = "1"
```

### Configuration Reference
See `.env.example` for complete configuration options:
- **Model Config**: `YOLO_WEIGHTS` (path to trained model)
- **Detection Config**: `DISABLE_DETECTION_DB`, `IMPORTANT_CLASSES`, `ALERT_CONF_MIN`
- **Anomaly Config**: `ENABLE_ANOMALY_ALERTS`, `ANOMALY_CLASSES`, `ANOMALY_CONF_MIN`
- **Email Alerts**: `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `ALERT_TO`
- **Database**: `SUPABASE_DB_URL` (optional Postgres URL)

---

## 10) Quick Command Summary

**Test Low-Light Upload** (no training needed):
```powershell
Push-Location "Night Surveillance System - Final1"; python .\main.py; Pop-Location
# Visit: http://127.0.0.1:5000/lowlight_detection
```

**Train Anomaly Models** (long-running):
```powershell
.\.venv\Scripts\python.exe "Night Surveillance System - Final1\Night Surveillance System - Final1\scripts\train_anomaly_model.py"
```

**Run with Anomaly Detection**:
```powershell
$env:YOLO_WEIGHTS = "runs\detect\anomaly_lowlight\weights\best.pt"
$env:ENABLE_ANOMALY_ALERTS = "1"
$env:ANOMALY_CLASSES = "person,car,knife,gun"
$env:ANOMALY_CONF_MIN = "0.5"
Push-Location "Night Surveillance System - Final1"; python .\main.py; Pop-Location
```
````
