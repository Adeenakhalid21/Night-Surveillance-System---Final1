# Night Shield - Night Surveillance System
## Complete Command Reference & User Guide

AI-powered surveillance system with low-light enhancement, object detection, and anomaly detection using YOLOv8.

**Project Location**: `C:\Users\HP\Desktop\Night Surveillance System - Final1\Night Surveillance System - Final1`

---

## 📋 Table of Contents
1. [Quick Start (5 Commands)](#-quick-start-5-essential-commands)
2. [Installation & Setup](#-installation--setup)
3. [Running the Application](#-running-the-application)
4. [Training Commands](#-training-commands)
5. [Dataset Management](#-dataset-management)
6. [Database Operations](#-database-operations)
7. [Configuration & Environment](#-configuration--environment-variables)
8. [Testing & Debugging](#-testing--debugging)
9. [Troubleshooting](#-troubleshooting)
10. [Development Commands](#-development-commands)
11. [Git & Version Control](#-git--version-control)
12. [Project Structure](#-project-structure)

---

## 🚀 Quick Start (5 Essential Commands)

```powershell
# 1. Navigate to project directory
cd "C:\Users\HP\Desktop\Night Surveillance System - Final1\Night Surveillance System - Final1"

# 2. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies (first time only)
pip install -r requirements.txt

# 4. Run the application
python main.py

# 5. Access web interface
# Open browser: http://127.0.0.1:5000
```

---

## 🔧 Installation & Setup

### Prerequisites
- **Python 3.10.11** (recommended) or 3.11
- **Windows PowerShell** 5.1 (default on Windows)
- **Webcam** or video files for surveillance
- **Optional**: Git for version control

### Create Virtual Environment
```powershell
# Create new virtual environment
python -m venv .venv

# Verify creation
Test-Path .\.venv
```

### Activate Virtual Environment
```powershell
# Windows PowerShell (recommended)
.\.venv\Scripts\Activate.ps1

# Windows CMD
.\.venv\Scripts\activate.bat

# Verify activation (prompt should show (.venv))
python --version
```

### Deactivate Virtual Environment
```powershell
deactivate
```

### Install All Dependencies
```powershell
# Install from requirements.txt
pip install -r requirements.txt

# Or install manually
pip install flask opencv-python ultralytics numpy pillow python-dotenv psycopg2-binary

# Install PyTorch CPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Install Specific Package Versions (Recommended)
```powershell
# Core packages with exact versions
pip install numpy==1.24.3
pip install pillow==10.4.0
pip install opencv-python==4.12.0.88
pip install ultralytics==8.3.228
pip install flask==3.0.0

# PyTorch CPU (for Windows)
pip install torch==2.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

### Upgrade pip
```powershell
python -m pip install --upgrade pip
```

---

## ▶️ Running the Application

### Start Flask Server
```powershell
# Standard startup
python main.py

# Run in debug mode (auto-reload on code changes)
$env:FLASK_DEBUG = "1"; python main.py

# Run on custom port
python main.py --port 8080

# Run with verbose logging
python main.py --verbose
```

### Stop the Application
```powershell
# Press Ctrl+C in the terminal

# Or kill Python process
Get-Process python | Stop-Process -Force

# Kill specific port (if stuck)
$port = 5000
netstat -ano | findstr :$port
# Note the PID, then:
taskkill /PID <PID> /F
```

### Web Interface URLs
```powershell
# Homepage
http://127.0.0.1:5000/

# Dashboard (main surveillance page)
http://127.0.0.1:5000/dashboard

# Low-Light Image Detection
http://127.0.0.1:5000/lowlight_detection

# Upload Video
http://127.0.0.1:5000/upload

# Add Camera
http://127.0.0.1:5000/addCamera

# View Datasets
http://127.0.0.1:5000/datasets

# Surveillance Events
http://127.0.0.1:5000/surveillance_events

# Analytics
http://127.0.0.1:5000/dataset_analytics
```

---

## 🧠 Training Commands

### Train Anomaly Detection Model

#### Method 1: Using PowerShell Helper Functions (Easiest)
```powershell
# Load helper functions
. .\tasks.ps1

# Fast training (5 epochs, ~10-15 minutes)
Start-LOLTrainingFast

# Full training (50 epochs, ~2-3 hours)
Start-LOLTrainingFull

# Custom training with parameters
Start-LOLTraining -EpochCount 20 -ImageSize 640 -BatchSize 8

# List all available functions
Get-Command -Name *LOL*
```

#### Method 2: Direct Python Training
```powershell
# Train on LOL eval15 dataset (default)
python train_anomaly_model.py

# Train with custom epochs
python -m ultralytics train data=lol_eval15.yaml model=yolov8n.pt epochs=20

# Full training command with all parameters
python -m ultralytics train `
  data=lol_eval15.yaml `
  model=yolov8n.pt `
  epochs=50 `
  imgsz=640 `
  batch=8 `
  patience=10 `
  device=cpu `
  workers=0 `
  name=anomaly_model
```

#### Method 3: Train Multiple Models (Automated)
```powershell
# Run automated training script (LOL + COCO datasets)
python scripts\train_anomaly_model.py

# This trains:
# 1. anomaly_lowlight (20 epochs on LOL dataset)
# 2. anomaly_general (15 epochs on COCO dataset)
```

### Training Parameters Explained
```powershell
# epochs     - Number of training iterations (5=fast, 50=best quality)
# imgsz      - Image size (416=fast, 640=balanced, 1280=quality)
# batch      - Batch size (4=low memory, 16=balanced, 32=fast)
# patience   - Early stopping patience (stop if no improvement)
# device     - cpu or cuda (use 'cpu' if no GPU)
# workers    - Data loading threads (0 for Windows)
# conf       - Confidence threshold (0.25=default, 0.5=strict)
# name       - Output folder name in runs/detect/
```

### Check Training Progress
```powershell
# View training logs
Get-Content runs\detect\<model_name>\train.log -Tail 50

# Open results folder
explorer runs\detect\<model_name>

# View training charts
explorer runs\detect\<model_name>\results.png

# Check best model weights
ls runs\detect\<model_name>\weights\best.pt
```

---

## 📊 Dataset Management

### Register New Dataset
```powershell
# Register LOL dataset
python register_dataset_folder.py "C:\path\to\LOL\eval15\low" --dataset-name "LOL Eval15"

# Register custom dataset
python register_dataset_folder.py "C:\path\to\your\images" --dataset-name "My Custom Dataset"

# Register with specific image types
python register_dataset_folder.py "C:\path\to\images" --dataset-name "Test" --extensions jpg,png,bmp
```

### List Available Datasets
```powershell
# Using PowerShell helper
. .\tasks.ps1
Get-Datasets

# Using Python
python -c "from main import get_db_connection; conn = get_db_connection(); import pandas as pd; print(pd.read_sql('SELECT * FROM datasets', conn))"

# Using SQLite directly
sqlite3 night_surveillance.db "SELECT dataset_id, dataset_name, image_count FROM datasets;"
```

### View Dataset Statistics
```powershell
# Count images in each dataset
sqlite3 night_surveillance.db "SELECT d.dataset_name, COUNT(*) as images FROM datasets d JOIN training_data t ON d.dataset_id=t.dataset_id GROUP BY d.dataset_name;"

# View dataset details
python -c "from main import get_db_connection; conn = get_db_connection(); print(conn.execute('SELECT * FROM datasets WHERE dataset_name LIKE \"%LOL%\"').fetchall())"
```

### View Training Results
```powershell
# List all training runs
ls runs\detect

# Open specific training results
explorer runs\detect\anomaly_eval15

# View best model metrics
python -c "from ultralytics import YOLO; model = YOLO('runs/detect/anomaly_eval15/weights/best.pt'); model.val()"

# Check model info
python -c "from ultralytics import YOLO; model = YOLO('runs/detect/anomaly_eval15/weights/best.pt'); print(model.info())"
```

---

## 🗄️ Database Operations

### Initialize Database
```powershell
# Database auto-creates on first run
python main.py

# Manually initialize
python -c "from main import init_db; init_db()"
```

### View Database Contents
```powershell
# Open SQLite shell
sqlite3 night_surveillance.db

# Common queries in SQLite
sqlite3 night_surveillance.db "SELECT * FROM cameras;"
sqlite3 night_surveillance.db "SELECT * FROM datasets ORDER BY created_at DESC LIMIT 10;"
sqlite3 night_surveillance.db "SELECT * FROM surveillance_events ORDER BY timestamp DESC LIMIT 20;"

# View recent events
sqlite3 night_surveillance.db "SELECT event_type, severity, description, timestamp FROM surveillance_events ORDER BY timestamp DESC LIMIT 10;"

# Count total detections
sqlite3 night_surveillance.db "SELECT COUNT(*) as total_detections FROM detections;"
```

### Database Queries Using Python
```powershell
# List all cameras
python -c "from main import get_db_connection; conn = get_db_connection(); print(conn.execute('SELECT * FROM cameras').fetchall())"

# List recent anomalies
python -c "from main import get_db_connection; conn = get_db_connection(); print(conn.execute('SELECT * FROM surveillance_events WHERE event_type=\"anomaly_detected\" ORDER BY timestamp DESC LIMIT 5').fetchall())"

# Get stats
python -c "from main import get_db_connection; conn = get_db_connection(); print('Cameras:', conn.execute('SELECT COUNT(*) FROM cameras').fetchone()[0]); print('Datasets:', conn.execute('SELECT COUNT(*) FROM datasets').fetchone()[0])"
```

### Backup Database
```powershell
# Create backup with timestamp
$date = Get-Date -Format "yyyyMMdd_HHmmss"
Copy-Item night_surveillance.db "backups\night_surveillance_$date.db"

# Create simple backup
Copy-Item night_surveillance.db night_surveillance_backup.db
```

### Reset Database
```powershell
# Backup first!
Copy-Item night_surveillance.db night_surveillance_backup.db

# Remove database (will recreate on next run)
Remove-Item night_surveillance.db

# Restart application
python main.py
```

### Export Database to CSV
```powershell
# Export datasets table
sqlite3 -header -csv night_surveillance.db "SELECT * FROM datasets;" > datasets.csv

# Export events table
sqlite3 -header -csv night_surveillance.db "SELECT * FROM surveillance_events;" > events.csv

# Export detections table
sqlite3 -header -csv night_surveillance.db "SELECT * FROM detections;" > detections.csv
```

---

## ⚙️ Configuration & Environment Variables

### Create .env File
```powershell
# Copy example file
Copy-Item .env.example .env

# Edit configuration
notepad .env
```

### Essential Configuration Options

#### YOLO Model Configuration
```powershell
# Set model path
$env:YOLO_MODEL_PATH = "yolov8n.pt"  # nano model (fastest)
$env:YOLO_MODEL_PATH = "yolov8s.pt"  # small model (balanced)
$env:YOLO_MODEL_PATH = "runs\detect\anomaly_eval15\weights\best.pt"  # trained model
```

#### Anomaly Detection Configuration
```powershell
# Enable anomaly alerts
$env:ENABLE_ANOMALY_ALERTS = "1"

# Define anomaly classes (comma-separated)
$env:ANOMALY_CLASSES = "person,car,motorcycle,truck,knife,gun,backpack"

# Set confidence threshold
$env:ANOMALY_CONF_MIN = "0.5"

# Example: Detect only weapons with high confidence
$env:ANOMALY_CLASSES = "knife,gun,weapon"
$env:ANOMALY_CONF_MIN = "0.7"
```

#### Email Alert Configuration
```powershell
# SMTP settings
$env:SMTP_HOST = "smtp.gmail.com"
$env:SMTP_PORT = "587"
$env:SMTP_USER = "your-email@gmail.com"
$env:SMTP_PASSWORD = "your-app-password"
$env:ALERT_TO = "recipient@example.com"
```

#### Database Configuration
```powershell
# Use SQLite (default)
Remove-Item env:SUPABASE_DB_URL -ErrorAction SilentlyContinue

# Use Supabase Postgres
$env:SUPABASE_DB_URL = "postgresql://user:pass@host:6543/postgres?sslmode=require"

# Disable detection logging (saves disk space)
$env:DISABLE_DETECTION_DB = "1"
```

#### Detection Configuration
```powershell
# Important classes for alerts
$env:IMPORTANT_CLASSES = "person,car,truck,motorcycle"

# Alert confidence threshold
$env:ALERT_CONF_MIN = "0.6"

# Frame skip for performance
$env:FRAME_SKIP = "2"  # process every 2nd frame
```

### View Current Environment
```powershell
# View all Python-related env vars
Get-ChildItem env: | Where-Object { $_.Name -like "*PYTHON*" -or $_.Name -like "*YOLO*" -or $_.Name -like "*ANOMALY*" }

# View specific variable
echo $env:YOLO_MODEL_PATH
```

### Clear Environment Variables
```powershell
# Clear specific variable
Remove-Item env:YOLO_MODEL_PATH -ErrorAction SilentlyContinue

# Clear all custom variables
Remove-Item env:ENABLE_ANOMALY_ALERTS -ErrorAction SilentlyContinue
Remove-Item env:ANOMALY_CLASSES -ErrorAction SilentlyContinue
Remove-Item env:ANOMALY_CONF_MIN -ErrorAction SilentlyContinue
```

---

## 🔍 Testing & Debugging

### Test Camera
```powershell
# Test if camera is accessible
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera working:', cap.isOpened()); cap.release()"

# Test multiple cameras
python -c "import cv2; for i in range(3): cap = cv2.VideoCapture(i); print(f'Camera {i}:', cap.isOpened()); cap.release()"

# Show camera properties
python -c "import cv2; cap = cv2.VideoCapture(0); print('Width:', cap.get(3)); print('Height:', cap.get(4)); print('FPS:', cap.get(5)); cap.release()"
```

### Test YOLO Model
```powershell
# Test model loading
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('Model loaded successfully')"

# Test detection on image
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); results = model('test.jpg'); print(results[0].boxes)"

# Test model info
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print(model.info())"
```

### Test Image Enhancement
```powershell
# Test enhancement module
python enhancement.py --input test_image.jpg --output enhanced_image.jpg

# Test with Python
python -c "from enhancement import enhance_image; import cv2; img = cv2.imread('test.jpg'); enhanced = enhance_image(img); cv2.imwrite('enhanced.jpg', enhanced)"
```

### Check Package Versions
```powershell
# Check Python version
python --version

# Check specific package
pip show ultralytics
pip show opencv-python
pip show torch
pip show numpy

# List all packages
pip list

# Check for outdated packages
pip list --outdated
```

### View Application Logs
```powershell
# Run with verbose output
python main.py --verbose

# View Flask logs (if logging to file)
Get-Content flask.log -Tail 50 -Wait

# View error logs
Get-Content error.log -Tail 50 -Wait
```

### Debug Mode
```powershell
# Run Flask in debug mode
$env:FLASK_DEBUG = "1"
python main.py

# Run with Python debugger
python -m pdb main.py

# Add breakpoints in code
# import pdb; pdb.set_trace()
```

---

## 🐛 Troubleshooting

### Fix Import Errors

#### NumPy Errors
```powershell
# Fix numpy version conflict
pip uninstall numpy -y
pip install numpy==1.24.3
```

#### Pillow/PIL Errors
```powershell
# Fix Pillow corruption
pip uninstall pillow -y
pip install pillow==10.4.0
```

#### PyTorch Errors
```powershell
# Reinstall PyTorch CPU
pip uninstall torch torchvision torchaudio -y
pip install torch==2.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

#### OpenCV Errors
```powershell
# Reinstall OpenCV
pip uninstall opencv-python opencv-python-headless -y
pip install opencv-python==4.12.0.88
```

#### Matplotlib Errors
```powershell
# Fix matplotlib import
pip uninstall matplotlib kiwisolver -y
pip install matplotlib kiwisolver
```

### Fix Module Not Found

```powershell
# Verify virtual environment is activated
where.exe python
# Should show: .venv\Scripts\python.exe

# If not activated
.\.venv\Scripts\Activate.ps1

# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall
```

### Clear Python Cache
```powershell
# Remove __pycache__ folders
Get-ChildItem -Recurse -Filter "__pycache__" | Remove-Item -Recurse -Force

# Remove .pyc files
Get-ChildItem -Recurse -Filter "*.pyc" | Remove-Item -Force

# Clear pip cache
pip cache purge
```

### Port Already in Use
```powershell
# Find process using port 5000
netstat -ano | findstr :5000

# Kill process by PID
taskkill /PID <PID_NUMBER> /F

# Kill all Python processes
taskkill /F /IM python.exe

# Or run on different port
python main.py --port 8080
```

### Camera Not Working
```powershell
# Check camera index
python -c "import cv2; print([i for i in range(5) if cv2.VideoCapture(i).isOpened()])"

# Test with different index
python -c "import cv2; cap = cv2.VideoCapture(1); print(cap.isOpened()); cap.release()"

# Check camera permissions
# Settings > Privacy > Camera > Allow desktop apps
```

### Low Detection Accuracy
```powershell
# Use larger model
$env:YOLO_MODEL_PATH = "yolov8s.pt"  # or yolov8m.pt

# Lower confidence threshold
$env:ANOMALY_CONF_MIN = "0.3"

# Train on more data
python train_anomaly_model.py --epochs 50 --batch 16
```

### Slow Performance
```powershell
# Skip frames for faster processing
$env:FRAME_SKIP = "3"

# Use smaller image size
python -m ultralytics train imgsz=416

# Use nano model
$env:YOLO_MODEL_PATH = "yolov8n.pt"

# Enable fast preset
python train_anomaly_model.py --fast-preset
```

### Database Locked Error
```powershell
# Close all connections
Get-Process python | Stop-Process -Force

# Backup and recreate
Copy-Item night_surveillance.db night_surveillance_backup.db
Remove-Item night_surveillance.db
python main.py
```

---

## 💻 Development Commands

### Create Requirements File
```powershell
# Generate from current environment
pip freeze > requirements.txt

# Create minimal requirements
@"
flask==3.0.0
opencv-python==4.12.0.88
ultralytics==8.3.228
numpy==1.24.3
pillow==10.4.0
python-dotenv==1.0.0
psycopg2-binary==2.9.9
"@ | Out-File -FilePath requirements.txt -Encoding utf8
```

### Code Formatting
```powershell
# Install formatters
pip install black flake8 autopep8

# Format Python files
black *.py

# Check code style
flake8 *.py

# Auto-fix style issues
autopep8 --in-place --aggressive --aggressive *.py
```

### Run Tests
```powershell
# Install pytest
pip install pytest

# Run tests
pytest

# Run with coverage
pip install pytest-cov
pytest --cov=. --cov-report=html
```

### Create Executable
```powershell
# Install PyInstaller
pip install pyinstaller

# Create standalone executable
pyinstaller --onefile `
  --add-data "templates;templates" `
  --add-data "static;static" `
  --add-data "yolov8n.pt;." `
  --name NightShield `
  main.py

# Executable will be in dist\NightShield.exe
```

### Generate Documentation
```powershell
# Install Sphinx
pip install sphinx sphinx-rtd-theme

# Initialize docs
sphinx-quickstart docs

# Generate HTML docs
sphinx-build -b html docs docs\_build
```

---

## 🔄 Git & Version Control

### Initialize Repository
```powershell
# Initialize git
git init

# Create .gitignore
@"
.venv/
__pycache__/
*.pyc
*.pyo
*.db
*.pt
runs/
.env
static/images/
static/lowlight_uploads/
night_surveillance.db
"@ | Out-File -FilePath .gitignore -Encoding utf8

# Add files
git add .

# Initial commit
git commit -m "Initial commit: Night Shield surveillance system"
```

### Daily Workflow
```powershell
# Check status
git status

# View changes
git diff

# Add all changes
git add .

# Add specific files
git add main.py templates/dashboard.html

# Commit changes
git commit -m "Add low-light detection feature"

# View commit history
git log --oneline -10
```

### Branch Management
```powershell
# Create new branch
git checkout -b feature/anomaly-alerts

# Switch branches
git checkout main

# List branches
git branch -a

# Merge branch
git checkout main
git merge feature/anomaly-alerts

# Delete branch
git branch -d feature/anomaly-alerts
```

### Remote Repository (GitHub)
```powershell
# Add remote
git remote add origin https://github.com/yourusername/night-shield.git

# Push to remote
git push -u origin main

# Pull from remote
git pull origin main

# View remotes
git remote -v
```

### Undo Changes
```powershell
# Discard unstaged changes
git checkout -- main.py

# Unstage files
git reset HEAD main.py

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1
```

---

## 📁 Project Structure

```
Night Surveillance System - Final1/
│
├── main.py                          # Flask application entry point
├── enhancement.py                   # Image enhancement module
├── train_anomaly_model.py          # Anomaly training script
├── register_dataset_folder.py      # Dataset registration tool
├── tasks.ps1                       # PowerShell helper functions
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── QUICKSTART.md                   # Quick start guide
├── .env                            # Environment variables (create from .env.example)
├── .env.example                    # Environment template
├── .gitignore                      # Git ignore rules
│
├── yolov8n.pt                      # YOLO nano model (fastest)
├── yolov8s.pt                      # YOLO small model (balanced)
├── night_surveillance.db           # SQLite database
│
├── .venv/                          # Virtual environment (created by you)
│   ├── Scripts/
│   │   ├── python.exe
│   │   ├── Activate.ps1
│   │   └── ...
│   └── Lib/
│
├── static/                         # Static web assets
│   ├── css/
│   │   ├── dashboard-style.css     # Main dashboard styles
│   │   ├── home-style.css
│   │   └── ...
│   ├── js/
│   │   ├── dashboard-script.js     # Dashboard JavaScript
│   │   ├── home-script.js
│   │   └── ...
│   ├── images/                     # Static images & detection snapshots
│   │   ├── anomalies/              # Anomaly detection images
│   │   └── detections/             # Regular detection images
│   ├── videos/                     # Uploaded videos
│   ├── lowlight_uploads/           # Low-light uploaded images
│   └── variables.txt               # Application state
│
├── templates/                      # HTML templates
│   ├── dashboard.html              # Main surveillance dashboard
│   ├── lowlight_detection.html     # Low-light image upload
│   ├── home.html                   # Homepage
│   ├── upload_video.html           # Video upload
│   ├── addCamera.html              # Add camera interface
│   ├── services.html
│   ├── about.html
│   ├── contact.html
│   └── ...
│
├── runs/                           # Training outputs
│   └── detect/
│       ├── anomaly_eval15/         # LOL dataset training
│       │   ├── weights/
│       │   │   ├── best.pt         # Best model weights
│       │   │   └── last.pt         # Last epoch weights
│       │   ├── results.png         # Training charts
│       │   ├── confusion_matrix.png
│       │   └── ...
│       ├── anomaly_lowlight/       # Low-light model
│       └── anomaly_general/        # General anomaly model
│
├── scripts/                        # Utility scripts
│   └── train_anomaly_model.py      # Automated training
│
└── __pycache__/                    # Python cache (auto-generated)
```

---

## 📘 Common Usage Scenarios

### Scenario 1: First Time Setup
```powershell
# 1. Navigate to project
cd "C:\Users\HP\Desktop\Night Surveillance System - Final1\Night Surveillance System - Final1"

# 2. Create virtual environment
python -m venv .venv

# 3. Activate environment
.\.venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run application
python main.py

# 6. Open browser
Start-Process "http://127.0.0.1:5000"
```

### Scenario 2: Daily Use
```powershell
# Quick start
cd "C:\Users\HP\Desktop\Night Surveillance System - Final1\Night Surveillance System - Final1"
.\.venv\Scripts\Activate.ps1
python main.py
```

### Scenario 3: Upload Low-Light Image
```powershell
# 1. Start application
python main.py

# 2. Navigate to low-light detection
# URL: http://127.0.0.1:5000/lowlight_detection

# 3. Drag & drop image or click to upload

# 4. Click "Analyze Image"

# 5. View enhanced image and detections
```

### Scenario 4: Train Custom Model
```powershell
# 1. Register your dataset
python register_dataset_folder.py "C:\path\to\your\images" --dataset-name "Custom Dataset"

# 2. Train model (fast)
. .\tasks.ps1
Start-LOLTrainingFast

# 3. Use trained model
$env:YOLO_MODEL_PATH = "runs\detect\anomaly_eval15\weights\best.pt"
python main.py
```

### Scenario 5: Enable Anomaly Alerts
```powershell
# 1. Configure environment
$env:ENABLE_ANOMALY_ALERTS = "1"
$env:ANOMALY_CLASSES = "person,knife,gun"
$env:ANOMALY_CONF_MIN = "0.6"

# 2. Configure email (edit .env file)
notepad .env
# Add: SMTP_HOST, SMTP_USER, SMTP_PASSWORD, ALERT_TO

# 3. Run with alerts enabled
python main.py

# 4. Monitor dashboard
# Anomalies will show red bounding boxes and trigger emails
```

### Scenario 6: Monitor Live Camera
```powershell
# 1. Start application
python main.py

# 2. Go to dashboard
# URL: http://127.0.0.1:5000/dashboard

# 3. Click "Start Camera" button

# 4. View live detections

# 5. Click "Stop Camera" when done
```

---

## 🆘 Quick Reference Card

### Essential Commands
```powershell
# ACTIVATE
.\.venv\Scripts\Activate.ps1

# RUN
python main.py

# TRAIN (Fast)
. .\tasks.ps1; Start-LOLTrainingFast

# INSTALL
pip install -r requirements.txt

# DEACTIVATE
deactivate
```

### Emergency Fixes
```powershell
# Fix numpy
pip uninstall numpy -y; pip install numpy==1.24.3

# Fix torch
pip uninstall torch -y; pip install torch==2.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Kill port 5000
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Kill all Python
taskkill /F /IM python.exe

# Clear cache
Get-ChildItem -Recurse -Filter "__pycache__" | Remove-Item -Recurse -Force
pip cache purge
```

### Quick Tests
```powershell
# Test camera
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"

# Test YOLO
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Test packages
python --version; pip list | Select-String "ultralytics|opencv|torch"
```

---

## 📞 Support & Resources

### Documentation
- **Quick Start Guide**: See `QUICKSTART.md`
- **YOLOv8 Documentation**: https://docs.ultralytics.com/
- **Flask Documentation**: https://flask.palletsprojects.com/
- **OpenCV Documentation**: https://docs.opencv.org/

### Common Issues & Solutions
| Issue | Solution |
|-------|----------|
| Camera not working | Check permissions, try different camera index |
| Port in use | Kill process: `taskkill /F /IM python.exe` |
| Import errors | Reinstall package: `pip install --force-reinstall` |
| Slow performance | Use yolov8n.pt, set FRAME_SKIP=3 |
| Low accuracy | Train with more data, use yolov8s.pt or larger |
| Database locked | Close all connections, restart app |

### Getting Help
1. Check this README for commands
2. Review `QUICKSTART.md` for setup issues
3. Check error messages in terminal
4. Verify Python version: `python --version`
5. Check package versions: `pip list`

---

## 📄 License

Copyright © 2024 Night Shield. All rights reserved.

---

## 👥 Contributing

For contributions and issues, please visit the GitHub repository:
- Repository: Night-Surveillance-System---Final1
- Owner: Adeenakhalid21

---

**Made with ❤️ by the Night Shield Team**

*Last Updated: December 1, 2025*
