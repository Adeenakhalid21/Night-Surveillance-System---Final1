# Night Surveillance System

A Flask-based night surveillance system with motion detection, YOLOv8 object detection, SQLite persistence, and dataset management (imports, analytics, and event logging).

## Features
- Motion detection with OpenCV and enhanced frame processing
- YOLOv8 object detection (Ultralytics)
- SQLite database for users, cameras, datasets, training samples, detections, and events
- Dataset pages: list, details, analytics; import scripts for LOL and VisDrone
- Email alerts with image attachments (configurable via environment variables)

## Requirements
- Python 3.10+
- Windows PowerShell 5.1 (for the commands below)

## Setup
1. Create and activate a virtual environment:
   ```powershell
   py -3.10 -m venv .venv; .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. (Optional) Configure environment variables by creating a `.env` file (see `.env.example`).

## Environment Variables
Create a `.env` file in the project root if you want to configure email alerts.

```
# SMTP configuration
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password

# Alerts
ALERT_TO=recipient_email@example.com
```

Notes:
- For Gmail, enable 2FA and create an app password.
- `.env` is ignored by git via `.gitignore`.

## Running the App
```powershell
# From the project folder
python main.py
# Then open http://127.0.0.1:5000
```

## Database Utilities
- `db_manager.py` creates/checks the SQLite DB and seeds sample data.
  ```powershell
  python db_manager.py --create
  python db_manager.py --check
  ```
- `test_database.py` runs basic DB CRUD sanity checks.

## Dataset Imports
- LOL Dataset import:
  ```powershell
  python import_lol_dataset.py "C:\Users\HP\Downloads\LOL Dataset\lol_dataset"
  ```
- VisDrone Dataset import:
  ```powershell
  python import_visdrone_dataset.py "C:\Users\HP\Downloads\VisDrone"
  ```

## Notes
- Model weights (`*.pt`), datasets, DB file, and videos are ignored by git (see `.gitignore`).
- If YOLO cannot find weights, place `yolov8n.pt` in the project folder (already present locally).
- SMTP credentials should not be hardcoded; use `.env` to override defaults.

## GitHub
Typical first-time push:
```powershell
git init; git add .; git commit -m "Initial commit"; git branch -M main
# Replace with your GitHub repo URL
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```
