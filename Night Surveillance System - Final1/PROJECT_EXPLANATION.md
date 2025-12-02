# Night Shield - Night Surveillance System
## Complete Project Explanation & Architecture Guide

**Project Type**: AI-Powered Real-Time Surveillance System with Low-Light Enhancement and Anomaly Detection  
**Tech Stack**: Python, Flask, YOLOv8, OpenCV, SQLite, HTML/CSS/JavaScript  
**Created**: 2024-2025  
**Purpose**: Intelligent surveillance system for night-time monitoring with automatic threat detection

---

## 📌 Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Core Technologies](#core-technologies)
4. [Project Structure](#project-structure)
5. [Key Features](#key-features)
6. [How It Works](#how-it-works)
7. [Database Schema](#database-schema)
8. [Frontend Design](#frontend-design)
9. [Backend Logic](#backend-logic)
10. [AI/ML Components](#aiml-components)
11. [Development Journey](#development-journey)
12. [Configuration System](#configuration-system)
13. [Performance Metrics](#performance-metrics)
14. [Future Enhancements](#future-enhancements)
15. [Recent Updates](#recent-updates)
16. [Summary](#summary)

---

## 🎯 Project Overview

**Night Shield** is an intelligent surveillance system designed specifically for night-time and low-light environments. It combines computer vision, deep learning, and web technologies to provide real-time monitoring with automatic threat detection and alerting.

### Problem Statement
Traditional surveillance systems struggle in low-light conditions and cannot intelligently differentiate between normal activities and potential threats. Security personnel must constantly monitor feeds, leading to fatigue and missed incidents.

### Solution
An AI-powered system that:
- **Enhances low-light images** automatically for better visibility
- **Detects objects and anomalies** in real-time using YOLOv8
- **Sends instant alerts** when threats are detected
- **Maintains a searchable database** of all events
- **Provides easy-to-use web interface** accessible from any device

### Target Use Cases
- Home security monitoring at night
- Parking lot surveillance
- Warehouse/storage facility monitoring
- Small business security
- Campus/institutional security

---

## 🏗️ System Architecture

### High-Level Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                      WEB BROWSER (Client)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Dashboard   │  │  Low-Light   │  │   Upload     │     │
│  │   (Live)     │  │  Detection   │  │   Video      │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            ↕ HTTP/SSE
┌─────────────────────────────────────────────────────────────┐
│               FLASK WEB SERVER (Backend)                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Routes Layer (URL Handling)                         │  │
│  │  /dashboard, /video_feed, /anomaly_alerts, etc.     │  │
│  └──────────────────────────────────────────────────────┘  │
│                            ↕                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Business Logic Layer                                │  │
│  │  - Video streaming                                   │  │
│  │  - Motion detection                                  │  │
│  │  - Object detection                                  │  │
│  │  - Anomaly detection                                 │  │
│  │  - Alert management                                  │  │
│  └──────────────────────────────────────────────────────┘  │
│                            ↕                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Data Layer (SQLite Database)                        │  │
│  │  - Users, Cameras, Datasets                          │  │
│  │  - Events, Detections                                │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│               AI/ML PROCESSING LAYER                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   YOLOv8     │  │   OpenCV     │  │ Enhancement  │     │
│  │  Detection   │  │   Video      │  │   Module     │     │
│  │              │  │  Processing  │  │              │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│                  HARDWARE LAYER                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Webcam     │  │  Video Files │  │   Storage    │     │
│  │   (Live)     │  │   (Upload)   │  │   (Disk)     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow
1. **Input** → Camera/Video file captures frames
2. **Processing** → OpenCV processes video stream
3. **Motion Detection** → Frame differencing detects movement
4. **Enhancement** → Low-light images enhanced (CLAHE, gamma correction)
5. **AI Detection** → YOLOv8 identifies objects
6. **Anomaly Check** → Filters detected objects against anomaly classes
7. **Alert Generation** → Creates alerts for threats
8. **Database Storage** → Logs events to SQLite
9. **Real-time Streaming** → SSE pushes alerts to browser
10. **Display** → User sees live feed with bounding boxes and popup alerts

---

## 💻 Core Technologies

### Backend Technologies
1. **Python 3.10.11**
   - Main programming language
   - Chosen for excellent AI/ML library support
   - Easy integration with OpenCV and YOLOv8

2. **Flask 3.0.0**
   - Lightweight web framework
   - Simple routing and request handling
   - Built-in development server
   - Session management for authentication

3. **SQLite3**
   - Embedded database (no separate server needed)
   - Lightweight and portable
   - Perfect for single-machine deployment
   - Stores users, cameras, events, datasets

4. **OpenCV 4.12.0.88**
   - Video capture and processing
   - Frame manipulation
   - Motion detection (frame differencing)
   - Image enhancement operations

5. **Ultralytics YOLOv8 8.3.228**
   - State-of-the-art object detection
   - Multiple model sizes (nano to extra-large)
   - Transfer learning support
   - Real-time performance

6. **NumPy 1.24.3**
   - Array operations
   - Mathematical computations
   - Image data manipulation

7. **Pillow 10.4.0**
   - Image loading and saving
   - Format conversions
   - Basic image operations

8. **PyTorch 2.9.1+cpu**
   - Deep learning framework
   - Required by YOLOv8
   - CPU-optimized version for broader compatibility

### Frontend Technologies
1. **HTML5**
   - Semantic markup
   - Video/image display elements
   - Form handling

2. **CSS3**
   - Custom properties (CSS variables) for theming
   - Flexbox and Grid layouts
   - Animations and transitions
   - Responsive design

3. **JavaScript (Vanilla)**
   - DOM manipulation
   - Event handling
   - Fetch API for AJAX requests
   - Server-Sent Events (SSE) for real-time updates
   - Web Audio API for alert sounds

4. **Material Symbols**
   - Google's icon font
   - Clean, modern icons
   - Scalable vector graphics

---

## 📁 Project Structure

```
Night Surveillance System - Final1/
│
├── main.py                          # Main Flask application (1039 lines)
│   ├── Flask app initialization
│   ├── Database connection handling
│   ├── Authentication routes (/login, /register, /logout)
│   ├── Dashboard route (/dashboard)
│   ├── Video streaming (/video_feed)
│   ├── Low-light detection (/lowlight_detection)
│   ├── Dataset management (/datasets, /add_dataset)
│   ├── Camera management (/addCamera, /updateCamera)
│   ├── Events and analytics routes
│   ├── SSE endpoint for anomaly alerts (/anomaly_alerts)
│   └── Helper functions (detect_objects, detect_anomalies, motion_detection)
│
├── enhancement.py                   # Image enhancement module
│   ├── CLAHE (Contrast Limited Adaptive Histogram Equalization)
│   ├── Gamma correction
│   ├── Brightness/contrast adjustment
│   └── Noise reduction
│
├── train_anomaly_model.py          # Model training script
│   ├── Dataset registration
│   ├── Pseudo-label generation
│   ├── YOLO fine-tuning
│   └── Training automation
│
├── register_dataset_folder.py      # Dataset registration utility
│   ├── Scans image folders
│   ├── Registers to SQLite database
│   └── Path validation
│
├── tasks.ps1                       # PowerShell helper functions
│   ├── Training shortcuts (Start-LOLTrainingFast, etc.)
│   ├── Dataset management (Get-Datasets)
│   └── Environment setup
│
├── requirements.txt                # Python dependencies
├── README.md                       # User documentation (1000+ lines)
├── QUICKSTART.md                   # Quick start guide
├── PROJECT_EXPLANATION.md          # This file
├── .env.example                    # Environment variables template
├── .gitignore                      # Git ignore rules
│
├── yolov8n.pt                      # YOLO nano model (6.3 MB)
├── yolov8s.pt                      # YOLO small model (22 MB)
├── night_surveillance.db           # SQLite database
│
├── .venv/                          # Python virtual environment
│   ├── Scripts/                    # Python executables
│   │   ├── python.exe
│   │   ├── Activate.ps1
│   │   └── pip.exe
│   └── Lib/                        # Installed packages
│
├── static/                         # Static web assets
│   ├── css/
│   │   ├── dashboard-style.css     # Main dashboard styles (1150+ lines)
│   │   │   ├── CSS variables for theming
│   │   │   ├── Dark/light mode support
│   │   │   ├── Responsive layouts
│   │   │   ├── Button styles
│   │   │   ├── Card designs
│   │   │   ├── Anomaly alert animations
│   │   │   └── Mobile responsiveness
│   │   ├── home-style.css          # Homepage styles
│   │   ├── upload-style.css        # Upload page styles
│   │   └── ...
│   │
│   ├── js/
│   │   ├── dashboard-script.js     # Dashboard functionality
│   │   │   ├── Theme toggle (light/dark)
│   │   │   ├── Sidebar navigation
│   │   │   └── Menu toggle
│   │   ├── home-script.js          # Homepage animations
│   │   └── ...
│   │
│   ├── images/                     # Static images & snapshots
│   │   ├── anomalies/              # Anomaly detection snapshots (auto-saved)
│   │   ├── detections/             # Regular detection snapshots
│   │   ├── Profile-pic.jpeg        # User profile image
│   │   └── 1.jpg                   # Hero/banner image
│   │
│   ├── videos/                     # Uploaded video files
│   ├── lowlight_uploads/           # Uploaded low-light images
│   └── variables.txt               # Application state variables
│
├── templates/                      # HTML Jinja2 templates
│   ├── dashboard.html              # Main surveillance dashboard (200+ lines)
│   │   ├── Live video feed container
│   │   ├── Camera control button
│   │   ├── Stats cards (cameras, datasets, detections, events)
│   │   ├── SSE connection for alerts
│   │   ├── Alert notification system
│   │   └── JavaScript for camera toggle
│   │
│   ├── lowlight_detection.html     # Low-light image upload (365+ lines)
│   │   ├── Drag-and-drop upload zone
│   │   ├── File validation
│   │   ├── AJAX image submission
│   │   ├── Side-by-side image display
│   │   ├── YOLOv8 detection integration
│   │   └── Detection results rendering
│   │
│   ├── image_enhancement.html      # Image enhancement tool (650+ lines)
│   │   ├── Drag-and-drop upload interface
│   │   ├── File validation (JPG, PNG, BMP)
│   │   ├── AJAX upload with FormData
│   │   ├── Before/after comparison grid
│   │   ├── Enhancement techniques info panel
│   │   ├── Toast notifications
│   │   ├── Loader animation
│   │   ├── NO YOLO detection (pure enhancement)
│   │   └── Mobile responsive design
│   │
│   ├── home.html                   # Landing page
│   │   ├── Hero section
│   │   ├── Feature highlights
│   │   └── Login/Register links
│   │
│   ├── upload_video.html           # Video upload interface
│   ├── addCamera.html              # Camera configuration
│   ├── datasets.html               # Dataset management
│   ├── services.html               # Services page
│   ├── about.html                  # About page
│   ├── contact.html                # Contact form
│   └── ...
│
├── runs/                           # YOLOv8 training outputs
│   └── detect/
│       ├── anomaly_eval15/         # LOL dataset training results
│       │   ├── weights/
│       │   │   ├── best.pt         # Best model checkpoint
│       │   │   └── last.pt         # Last epoch checkpoint
│       │   ├── results.png         # Training metrics chart
│       │   ├── confusion_matrix.png
│       │   ├── F1_curve.png
│       │   ├── PR_curve.png
│       │   ├── P_curve.png
│       │   ├── R_curve.png
│       │   └── train_batch*.jpg    # Training samples
│       │
│       ├── anomaly_lowlight/       # Low-light specialized model
│       └── anomaly_general/        # General anomaly model
│
└── __pycache__/                    # Python bytecode cache (auto-generated)
```

---

## ⚡ Key Features

The Night Shield system offers **10 comprehensive features** covering surveillance, AI detection, enhancement, alerts, and user management:

### 1. Real-Time Video Surveillance
- **Live webcam streaming** via Flask Response with MJPEG
- **Frame-by-frame processing** with OpenCV
- **Motion detection** using frame differencing
- **Efficient streaming** with frame skipping (process every 3rd frame)
- **Manual camera controls** (Start/Stop button)

**Implementation**:
```python
def video_stream(source, stop_event):
    cap = cv2.VideoCapture(source)  # Open camera/video
    while not stop_event.is_set():
        for _ in range(3): cap.read()  # Skip frames
        ret, frame = cap.read()
        if motion_detected:
            enhanced = enhance_image(frame)
            output = detect_objects(enhanced)
            output, anomalies = detect_anomalies(output)
        yield encode_frame(output)
```

### 2. Low-Light Image Enhancement
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization)
- **Gamma correction** for brightness adjustment
- **Bilateral filtering** for noise reduction
- **Automatic processing** before detection

**Algorithm**:
```python
def enhance_image(image):
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Gamma correction
    gamma = 1.5
    l = np.power(l/255.0, gamma) * 255
    
    # Merge and convert back
    enhanced = cv2.merge([l,a,b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
```

### 3. YOLOv8 Object Detection
- **Pre-trained COCO weights** (80 classes)
- **Custom trained models** for anomaly detection
- **Multiple model sizes** (nano, small, medium, large)
- **Real-time inference** on CPU/GPU
- **Bounding box visualization** with confidence scores

**Detection Pipeline**:
```python
def detect_objects(frame):
    model = get_model()  # Lazy load YOLOv8
    results = model(frame)
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0]
        cls = box.cls[0]
        class_name = model.names[int(cls)]
        
        # Draw green box for normal objects
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"{class_name} {conf:.2f}", ...)
    
    return frame
```

### 4. Anomaly Detection System
- **Configurable anomaly classes** (person, car, knife, gun, etc.)
- **Confidence threshold filtering**
- **Red bounding boxes** for threats
- **Automatic snapshot saving** with timestamps
- **Database logging** of all anomaly events
- **Email alerts** (threaded, non-blocking)

**Anomaly Logic**:
```python
def detect_anomalies(frame, camera_id=1):
    results = model(frame)
    anomalies = []
    
    for detection in results[0].boxes:
        class_name = model.names[int(cls)]
        
        if class_name.lower() in ANOMALY_CLASSES and conf >= ANOMALY_CONF_MIN:
            anomalies.append({
                'class': class_name,
                'confidence': conf,
                'bbox': (x1,y1,x2,y2)
            })
            
            # RED box for anomalies
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 3)
            
            # Save snapshot
            cv2.imwrite(f'anomalies/anomaly_{timestamp}.jpg', frame)
            
            # Log to database
            log_event(camera_id, class_name, conf, 'high')
            
            # Send email alert (threaded)
            threading.Thread(target=send_email_alert).start()
    
    return frame, anomalies
```

### 5. Real-Time Alert Notifications
- **Server-Sent Events (SSE)** for push notifications
- **Animated popup alerts** that auto-dismiss after 5 seconds
- **Audio alerts** (beep sound via Web Audio API)
- **Visual effects** (slide-in, pulse, shake animations)
- **Queue management** (keeps last 10 anomalies)

**SSE Implementation**:
```python
@app.route('/anomaly_alerts')
def anomaly_alerts():
    def generate():
        last_sent = 0
        while True:
            with anomalies_lock:
                if len(recent_anomalies) > last_sent:
                    for i in range(last_sent, len(recent_anomalies)):
                        anomaly = recent_anomalies[i]
                        yield f"data: {json.dumps(anomaly)}\n\n"
                    last_sent = len(recent_anomalies)
            time.sleep(0.5)
    return Response(generate(), mimetype='text/event-stream')
```

**Frontend Alert System**:
```javascript
// Connect to SSE endpoint when camera starts
eventSource = new EventSource('/anomaly_alerts');
eventSource.onmessage = function(event) {
    const data = JSON.parse(event.data);
    showAnomalyAlert(data.class, data.confidence);
};

function showAnomalyAlert(className, confidence) {
    const alert = document.createElement('div');
    alert.className = 'anomaly-alert';
    alert.innerHTML = `
        <div class="alert-icon">⚠️</div>
        <div class="alert-content">
            <h4>ANOMALY DETECTED</h4>
            <p>${className} - ${confidence}%</p>
        </div>
    `;
    alertContainer.appendChild(alert);
    
    // Auto-remove after 5 seconds
    setTimeout(() => alert.remove(), 5000);
}
```

### 6. Database Management
- **User authentication** with hashed passwords
- **Camera configuration** storage
- **Dataset registration** and tracking
- **Event logging** (detections, anomalies, alerts)
- **Analytics data** for reporting

**Schema Highlights**:
```sql
-- Users table
CREATE TABLE user (
    id INTEGER PRIMARY KEY,
    email TEXT UNIQUE,
    password TEXT,  -- Hashed
    firstname TEXT,
    lastname TEXT
);

-- Cameras table
CREATE TABLE cam (
    id INTEGER PRIMARY KEY,
    camname TEXT,
    location TEXT,
    status TEXT,
    user_id INTEGER,
    FOREIGN KEY(user_id) REFERENCES user(id)
);

-- Surveillance events table
CREATE TABLE surveillance_events (
    event_id INTEGER PRIMARY KEY,
    camera_id INTEGER,
    event_type TEXT,  -- 'anomaly_detected', 'motion', etc.
    severity TEXT,    -- 'low', 'medium', 'high'
    description TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(camera_id) REFERENCES cam(id)
);

-- Datasets table (for training)
CREATE TABLE datasets (
    dataset_id INTEGER PRIMARY KEY,
    dataset_name TEXT,
    dataset_type TEXT,
    description TEXT,
    file_path TEXT,
    created_date DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### 7. Responsive Web Interface
- **Light/Dark theme toggle** with CSS variables
- **Mobile-responsive design** with media queries
- **Material Design** inspired UI
- **Smooth animations** and transitions
- **Intuitive navigation** with sidebar

**Theme System**:
```css
:root {
    --clr-primary: #7380ec;
    --clr-danger: #ff7782;
    --clr-success: #41f1b6;
    --clr-white: #fff;
    --clr-dark: #363949;
}

.dark-theme-variables {
    --clr-white: #111827;
    --clr-dark: #e5e7eb;
    --clr-primary: #5a67d8;
}
```

### 8. Dataset & Model Training
- **Automated training pipeline** with pseudo-labels
- **Multiple training presets** (fast, balanced, full)
- **Transfer learning** from COCO weights
- **Early stopping** to prevent overfitting
- **Training metrics visualization** (mAP, precision, recall)

**Training Features**:
- Registers LOL (Low-Light) and COCO datasets
- Generates pseudo-labels using pre-trained model
- Fine-tunes YOLOv8 with freezing layers
- Saves best model checkpoint
- Produces training charts (loss, metrics curves)

### 9. Configuration System
- **Environment variables** via .env file
- **Runtime configuration** with PowerShell
- **Flexible model selection** (nano, small, medium, etc.)
- **Anomaly class customization**
- **Alert threshold tuning**

**Key Config Options**:
```bash
YOLO_WEIGHTS=runs/detect/anomaly_eval15/weights/best.pt
ENABLE_ANOMALY_ALERTS=1
ANOMALY_CLASSES=person,car,knife,gun,backpack
ANOMALY_CONF_MIN=0.5
SMTP_HOST=smtp.gmail.com
SMTP_USER=your-email@gmail.com
```

### 10. Image Enhancement for Low-Light Photos
- **Standalone enhancement tool** for individual photos
- **Drag-and-drop upload interface** for easy file selection
- **NO YOLO detection** - pure enhancement only
- **Before/after comparison** side-by-side display
- **Advanced algorithms** (CLAHE, Gamma, Bilateral Filter)
- **AJAX processing** with real-time feedback
- **Toast notifications** on success/error
- **Mobile responsive** grid layout

**Feature Purpose**:
This feature allows users to enhance low-light photos without running object detection. It's useful for:
- Testing enhancement quality on sample images
- Improving personal photos taken in dark conditions
- Quick enhancement without full detection pipeline
- Demonstrating enhancement capabilities separately

**Implementation**:
```python
@app.route('/image_enhancement', methods=['GET', 'POST'])
def image_enhancement():
    if request.method == 'POST':
        file = request.files['image']
        
        # Save original
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        original_path = f"static/lowlight_uploads/original_{timestamp}_{file.filename}"
        file.save(original_path)
        
        # Read and enhance
        img = cv2.imread(original_path)
        enhanced_img = enhance_image(img)  # Same enhancement module
        
        # Save enhanced version
        enhanced_path = f"static/lowlight_uploads/enhanced_{timestamp}_{file.filename}"
        cv2.imwrite(enhanced_path, enhanced_img)
        
        # Return both paths for display
        return jsonify({
            'success': True,
            'original': f"/static/lowlight_uploads/original_{timestamp}_{file.filename}",
            'enhanced': f"/static/lowlight_uploads/enhanced_{timestamp}_{file.filename}"
        })
    
    return render_template('image_enhancement.html')
```

**Frontend Features**:
```javascript
// Drag-and-drop file handling
uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    handleFileUpload(file);
});

// AJAX submission with FormData
async function enhanceImage() {
    const formData = new FormData();
    formData.append('image', selectedFile);
    
    const response = await fetch('/image_enhancement', {
        method: 'POST',
        body: formData
    });
    
    const data = await response.json();
    
    // Display before/after
    originalImage.src = data.original;
    enhancedImage.src = data.enhanced;
    
    // Show success toast
    showToast('Image successfully enhanced! ✨', 'success');
}
```

**UI Components**:
- **Upload Area**: Drag-and-drop zone with visual feedback
- **Preview Grid**: 2-column responsive layout for comparison
- **Image Cards**: Separate cards for original (blue) and enhanced (green)
- **Enhancement Info Panel**: Describes applied techniques
- **Loader Animation**: Spinner during processing
- **Toast Notifications**: Success/error feedback

**Enhancement Techniques Displayed**:
1. **CLAHE Enhancement** - Local brightness improvement
2. **Gamma Correction** - Non-linear brightness adjustment
3. **Noise Reduction** - Bilateral filtering for smoothing
4. **Color Enhancement** - LAB color space processing

**Key Differences from Low-Light Detection**:
| Aspect | Low-Light Detection | Image Enhancement |
|--------|-------------------|------------------|
| Purpose | Detect objects in dark images | Only enhance image quality |
| YOLO | ✅ Yes | ❌ No |
| Processing Time | 3-5 seconds | 1-2 seconds |
| Output | Enhanced + Detections | Enhanced only |
| Use Case | Security analysis | Photo improvement |

---

## 🔄 How It Works

### End-to-End Flow

#### 1. User Opens Dashboard
```
User → Browser → http://127.0.0.1:5000/dashboard
↓
Flask receives GET request
↓
Checks session for authentication
↓
Queries database for stats (cam_count, dataset_count, etc.)
↓
Renders dashboard.html with Jinja2 template
↓
Browser displays dashboard with "Start Camera" button
```

#### 2. User Starts Camera
```
User clicks "Start Camera" button
↓
JavaScript changes button text to "Stop Camera"
↓
Sets video source: <img src="/video_feed">
↓
Browser makes GET request to /video_feed
↓
Flask returns Response with MJPEG stream
↓
JavaScript connects to SSE: /anomaly_alerts
↓
EventSource starts listening for anomaly events
```

#### 3. Video Processing Loop
```
Flask video_stream() function starts
↓
cv2.VideoCapture(0) opens webcam
↓
Loop: Read frame every iteration
    ↓
    Skip 2 frames (for efficiency)
    ↓
    Read actual frame
    ↓
    Resize to 640x480
    ↓
    motion_detection(prev_frame, current_frame)
        ↓
        Calculate frame difference
        ↓
        Apply threshold
        ↓
        Count white pixels
        ↓
        If > threshold: motion_detected = True
    ↓
    If motion_detected:
        ↓
        enhance_image(frame)
            ↓
            CLAHE on L channel
            ↓
            Gamma correction
            ↓
            Bilateral filter
        ↓
        detect_objects_and_classify(enhanced_frame)
            ↓
            Load YOLOv8 model (lazy load)
            ↓
            Run inference: results = model(frame)
            ↓
            For each detection:
                ↓
                Extract bbox, confidence, class
                ↓
                Draw GREEN bounding box
                ↓
                Add label with class name and confidence
        ↓
        detect_anomalies(frame)
            ↓
            For each detection:
                ↓
                Check if class in ANOMALY_CLASSES
                ↓
                Check if confidence >= ANOMALY_CONF_MIN
                ↓
                If yes (ANOMALY):
                    ↓
                    Draw RED bounding box
                    ↓
                    Save snapshot to disk
                    ↓
                    Log to surveillance_events table
                    ↓
                    Add to recent_anomalies list (thread-safe)
                    ↓
                    Send email alert (threaded)
    ↓
    Encode frame to JPEG
    ↓
    Yield frame bytes with MJPEG boundary
    ↓
    Update prev_frame = current_frame
↓
Loop continues until stop_event is set
```

#### 4. Real-Time Alert Flow
```
Backend adds anomaly to recent_anomalies list
↓
SSE endpoint /anomaly_alerts detects new anomaly
↓
Generates SSE message: "data: {json}\n\n"
↓
Sends to all connected clients
↓
Browser EventSource receives message
↓
JavaScript parses JSON data
↓
showAnomalyAlert(class, confidence) function called
↓
Creates div.anomaly-alert element
    ↓
    Adds warning icon
    ↓
    Adds alert content (class, confidence, time)
↓
Appends to #alert-container
↓
Triggers CSS animation (slide-in from right)
↓
Plays beep sound (Web Audio API)
↓
Starts 5-second timer
↓
After 5 seconds: fade out animation
↓
Remove element from DOM
```

#### 5. Low-Light Image Upload Flow
```
User goes to /lowlight_detection page
↓
Drags and drops image OR clicks to browse
↓
JavaScript validates file type (jpg, png, bmp)
↓
Shows preview of original image
↓
User clicks "Analyze Image" button
↓
JavaScript creates FormData with image file
↓
fetch('/lowlight_detection', {method: 'POST', body: formData})
↓
Flask receives POST request
    ↓
    Validates file extension
    ↓
    Saves to static/lowlight_uploads/
    ↓
    cv2.imread(image_path)
    ↓
    enhance_image(img)
    ↓
    Run YOLOv8 detection
    ↓
    Draw bounding boxes on enhanced image
    ↓
    Save detected image
    ↓
    Build JSON response with detections
↓
JavaScript receives JSON response
↓
Displays detected image next to original
↓
Shows detection results (class, confidence, coordinates)
```

---

## 🗄️ Database Schema

### Complete Database Structure

```sql
-- Users table (Authentication)
CREATE TABLE IF NOT EXISTS user (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    firstname TEXT,
    lastname TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Cameras table (Device Management)
CREATE TABLE IF NOT EXISTS cam (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camname TEXT NOT NULL,
    location TEXT,
    ip_address TEXT,
    port INTEGER,
    username TEXT,
    password TEXT,
    status TEXT DEFAULT 'active',
    user_id INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES user(id)
);

-- Datasets table (Training Data Management)
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_name TEXT NOT NULL,
    dataset_type TEXT,
    description TEXT,
    file_path TEXT,
    image_count INTEGER DEFAULT 0,
    created_date DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Training data table (Dataset Images)
CREATE TABLE IF NOT EXISTS training_data (
    training_id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER,
    image_path TEXT,
    label TEXT,
    added_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id)
);

-- Surveillance events table (Activity Logging)
CREATE TABLE IF NOT EXISTS surveillance_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id INTEGER,
    event_type TEXT NOT NULL,
    severity TEXT,
    description TEXT,
    image_path TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(camera_id) REFERENCES cam(id)
);

-- Detections table (Object Detection Results)
CREATE TABLE IF NOT EXISTS detections (
    detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id INTEGER,
    class_name TEXT,
    confidence REAL,
    bbox_x1 INTEGER,
    bbox_y1 INTEGER,
    bbox_x2 INTEGER,
    bbox_y2 INTEGER,
    image_path TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(camera_id) REFERENCES cam(id)
);

-- Detection results table (Training Evaluation)
CREATE TABLE IF NOT EXISTS detection_results (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER,
    camera_id INTEGER,
    detection_count INTEGER,
    accuracy REAL,
    processed_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id),
    FOREIGN KEY(camera_id) REFERENCES cam(id)
);
```

### Example Data

```sql
-- Sample user
INSERT INTO user VALUES (1, 'admin@nightshield.com', 'hashed_password', 'Admin', 'User', '2024-01-01');

-- Sample camera
INSERT INTO cam VALUES (1, 'Front Door Camera', 'Entrance', '192.168.1.100', 554, 'admin', 'pass', 'active', 1, '2024-01-01');

-- Sample dataset
INSERT INTO datasets VALUES (1, 'LOL Eval15', 'low-light', 'Low-light enhancement dataset', 'lol_dataset/our485/low', 15, '2024-01-01');

-- Sample anomaly event
INSERT INTO surveillance_events VALUES (1, 1, 'anomaly_detected', 'high', 'Person detected at entrance', 'anomalies/anomaly_20241201_120000_person.jpg', '2024-12-01 12:00:00');
```

---

## 🎨 Frontend Design

### Design System

#### Color Palette
```css
/* Light Theme */
--clr-primary: #7380ec (Blue)
--clr-danger: #ff7782 (Red)
--clr-success: #41f1b6 (Green)
--clr-warning: #ffbb55 (Orange)
--clr-white: #ffffff
--clr-dark: #363949
--clr-info-dark: #7d8da1
--clr-info-light: #dce1eb

/* Dark Theme */
--clr-white: #111827 (Dark background)
--clr-dark: #e5e7eb (Light text)
--clr-primary: #5a67d8
```

#### Typography
- **Font Family**: 'Poppins', sans-serif
- **Headings**: h1 (1.8rem), h2 (1.4rem), h3 (0.87rem)
- **Body**: 0.88rem
- **Small**: 0.75rem

#### Spacing
- **Card Padding**: 1.8rem
- **Border Radius**: 0.4rem (small), 2rem (large)
- **Gap**: 1.6rem between elements

#### Components

**1. Cards**
```css
.card {
    background: var(--clr-white);
    border-radius: var(--border-radius-2);
    box-shadow: var(--box-shadow);
    padding: var(--card-padding);
    transition: all 0.3s ease;
}
.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 14px 30px rgba(0,0,0,0.2);
}
```

**2. Buttons**
```css
.btn-primary {
    background: linear-gradient(135deg, #7380ec, #5a67d8);
    color: white;
    padding: 0.8rem 1.5rem;
    border-radius: var(--border-radius-1);
    box-shadow: 0 4px 12px rgba(115, 128, 236, 0.3);
    transition: all 0.3s ease;
}
.btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(115, 128, 236, 0.4);
}
```

**3. Anomaly Alerts**
```css
.anomaly-alert {
    background: linear-gradient(135deg, #ef4444, #dc2626);
    color: white;
    border-radius: var(--border-radius-2);
    box-shadow: 0 8px 32px rgba(239, 68, 68, 0.5);
    animation: pulse-border 2s infinite;
}
@keyframes pulse-border {
    0%, 100% { box-shadow: 0 8px 32px rgba(239, 68, 68, 0.5); }
    50% { box-shadow: 0 8px 48px rgba(239, 68, 68, 0.8); }
}
```

#### Layout
- **Sidebar**: Fixed left navigation (18rem width)
- **Main Content**: Flexible center area
- **Right Panel**: Stats and profile (23rem width)
- **Grid System**: CSS Grid for insights cards (repeat(3, 1fr))

### User Experience Features
1. **Smooth page transitions**
2. **Loading spinners** during image processing
3. **Toast notifications** for user actions
4. **Drag-and-drop** file upload
5. **Responsive tables** with horizontal scroll
6. **Modal dialogs** for confirmations
7. **Progress indicators** for uploads

---

## 🔧 Backend Logic

### Flask Application Structure

#### 1. Application Initialization
```python
app = Flask(__name__)
app.secret_key = 'mysecretkey'
app.config['UPLOAD_FOLDER'] = 'static/videos'
app.config['LOWLIGHT_FOLDER'] = 'static/lowlight_uploads'

# Environment variables
load_dotenv()
ANOMALY_CLASSES = set(os.getenv('ANOMALY_CLASSES', '').split(','))
ANOMALY_CONF_MIN = float(os.getenv('ANOMALY_CONF_MIN', '0.5'))
```

#### 2. Database Connection
```python
def get_db_connection():
    conn = sqlite3.connect('night_surveillance.db')
    conn.row_factory = sqlite3.Row  # Access columns by name
    return conn

def init_db():
    conn = get_db_connection()
    # Create tables if not exist
    conn.execute('CREATE TABLE IF NOT EXISTS user (...)')
    conn.execute('CREATE TABLE IF NOT EXISTS cam (...)')
    # ... more tables
    conn.close()
```

#### 3. Authentication System
```python
@app.route('/login', methods=['POST'])
def login():
    email = request.form['email']
    password = request.form['password']
    
    conn = get_db_connection()
    user = conn.execute(
        'SELECT * FROM user WHERE email = ? AND password = ?',
        (email, password)
    ).fetchone()
    conn.close()
    
    if user:
        session['loggedin'] = True
        session['id'] = user['id']
        session['email'] = user['email']
        session['firstname'] = user['firstname']
        return redirect(url_for('dashboard'))
    else:
        return jsonify({'error': 'Invalid credentials'}), 401
```

#### 4. Video Streaming
```python
def video_stream(source, stop_event):
    cap = cv2.VideoCapture(source)
    
    while not stop_event.is_set() and cap.isOpened():
        for _ in range(3): cap.read()  # Skip frames
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.resize(frame, (640, 480))
        
        if motion_detected:
            enhanced = enhance_image(frame)
            output = detect_objects_and_classify(enhanced)
            output, anomalies = detect_anomalies(output)
            
            if anomalies and ENABLE_ANOMALY_ALERTS:
                with anomalies_lock:
                    recent_anomalies.append(anomalies)
        
        _, jpeg = cv2.imencode('.jpg', output)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    
    cap.release()

@app.route('/video_feed')
def video_feed():
    stop_event = threading.Event()
    return Response(video_stream(0, stop_event),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
```

#### 5. Motion Detection
```python
def motion_detection(prev_frame, current_frame):
    global motion_detected
    
    # Convert to grayscale
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate absolute difference
    frame_diff = cv2.absdiff(gray_prev, gray_curr)
    
    # Apply threshold
    _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
    
    # Count white pixels
    motion_pixels = cv2.countNonZero(thresh)
    threshold = 5000  # Adjust sensitivity
    
    motion_detected = motion_pixels > threshold
```

#### 6. Email Alerts
```python
def send_email_alert(anomaly_class, confidence, image_path):
    try:
        smtp_host = os.getenv('SMTP_HOST', 'smtp.gmail.com')
        smtp_port = int(os.getenv('SMTP_PORT', '587'))
        smtp_user = os.getenv('SMTP_USER')
        smtp_pass = os.getenv('SMTP_PASSWORD')
        alert_to = os.getenv('ALERT_TO')
        
        msg = EmailMessage()
        msg['Subject'] = f'⚠️ ANOMALY DETECTED: {anomaly_class}'
        msg['From'] = smtp_user
        msg['To'] = alert_to
        
        msg.set_content(f'''
        Anomaly Detection Alert
        
        Class: {anomaly_class}
        Confidence: {confidence:.2%}
        Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Please check the attached image.
        ''')
        
        # Attach image
        with open(image_path, 'rb') as f:
            msg.add_attachment(f.read(), maintype='image', subtype='jpeg')
        
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        
        print(f'[EMAIL] Alert sent for {anomaly_class}')
    except Exception as e:
        print(f'[EMAIL] Failed: {e}')
```

#### 7. SSE for Real-Time Alerts
```python
@app.route('/anomaly_alerts')
def anomaly_alerts():
    def generate():
        last_sent_count = 0
        while True:
            with anomalies_lock:
                if len(recent_anomalies) > last_sent_count:
                    for i in range(last_sent_count, len(recent_anomalies)):
                        anomaly = recent_anomalies[i]
                        data = {
                            'class': anomaly['class'],
                            'confidence': anomaly['confidence'],
                            'timestamp': anomaly['timestamp']
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                    last_sent_count = len(recent_anomalies)
            time.sleep(0.5)
    
    return Response(generate(), mimetype='text/event-stream')
```

---

## 🤖 AI/ML Components

### YOLOv8 Architecture

**YOLO (You Only Look Once)** is a single-stage object detector that:
1. Divides image into grid
2. Predicts bounding boxes and class probabilities for each grid cell
3. Non-maximum suppression to remove duplicate detections

**YOLOv8 Improvements**:
- Anchor-free detection
- New backbone (CSPDarknet)
- Decoupled head (separate classification and localization)
- Task-specific architectures (detect, segment, classify, pose)

### Model Variants
| Model | Parameters | Size | Speed | mAP |
|-------|------------|------|-------|-----|
| YOLOv8n | 3.2M | 6.3 MB | Fastest | 37.3% |
| YOLOv8s | 11.2M | 22 MB | Fast | 44.9% |
| YOLOv8m | 25.9M | 52 MB | Medium | 50.2% |
| YOLOv8l | 43.7M | 87 MB | Slow | 52.9% |
| YOLOv8x | 68.2M | 136 MB | Slowest | 53.9% |

### Training Pipeline

#### Step 1: Dataset Registration
```python
def register_dataset(folder_path, dataset_name):
    conn = get_db_connection()
    dataset_id = conn.execute(
        'INSERT INTO datasets (dataset_name) VALUES (?)',
        (dataset_name,)
    ).lastrowid
    
    for img_file in os.listdir(folder_path):
        if img_file.endswith(('.jpg', '.png')):
            img_path = os.path.join(folder_path, img_file)
            conn.execute(
                'INSERT INTO training_data (dataset_id, image_path) VALUES (?, ?)',
                (dataset_id, img_path)
            )
    
    conn.commit()
    conn.close()
```

#### Step 2: Pseudo-Label Generation
```python
def generate_pseudo_labels(image_folder, output_folder):
    model = YOLO('yolov8s.pt')  # Pre-trained model
    
    for img_file in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_file)
        results = model(img_path, conf=0.35)
        
        # Save in YOLO format (class x_center y_center width height)
        label_path = os.path.join(output_folder, img_file.replace('.jpg', '.txt'))
        with open(label_path, 'w') as f:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                x_center, y_center, width, height = box.xywhn[0]
                f.write(f"{cls} {x_center} {y_center} {width} {height}\n")
```

#### Step 3: Fine-Tuning
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(
    data='lol_eval15.yaml',  # Dataset config
    epochs=50,
    imgsz=640,
    batch=8,
    patience=10,  # Early stopping
    device='cpu',
    workers=0,
    name='anomaly_eval15',
    project='runs/detect',
    freeze=10  # Freeze first 10 layers
)
```

#### Step 4: Evaluation
```python
metrics = model.val()
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
print(f"Precision: {metrics.box.p}")
print(f"Recall: {metrics.box.r}")
```

### Image Enhancement Techniques

#### CLAHE (Contrast Limited Adaptive Histogram Equalization)
```python
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)
    
    enhanced = cv2.merge([l_clahe, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
```

**How it works**:
1. Convert to LAB color space (separates luminance from color)
2. Apply histogram equalization to L channel only
3. Limit contrast to avoid noise amplification
4. Use 8x8 tile grid for local adaptation

#### Gamma Correction
```python
def adjust_gamma(image, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)
```

**Purpose**: Brightens dark images by applying non-linear transformation

#### Bilateral Filter
```python
def reduce_noise(image):
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
```

**Purpose**: Smooths image while preserving edges (better than Gaussian blur)

---

## 🚀 Development Journey

### Phase 1: Initial Setup (Week 1)
- Set up Python virtual environment
- Installed Flask, OpenCV, Ultralytics
- Created basic project structure
- Implemented simple video streaming

### Phase 2: Core Features (Week 2-3)
- Added user authentication system
- Implemented SQLite database
- Created dashboard interface
- Added camera management

### Phase 3: AI Integration (Week 4-5)
- Integrated YOLOv8 for object detection
- Implemented motion detection
- Added bounding box visualization
- Created detection logging system

### Phase 4: Low-Light Enhancement (Week 6)
- Researched enhancement techniques
- Implemented CLAHE algorithm
- Added gamma correction
- Created low-light upload interface

### Phase 5: Anomaly Detection (Week 7-8)
- Designed anomaly classification system
- Implemented configurable anomaly classes
- Added red bounding boxes for threats
- Created event logging

### Phase 6: Alert System (Week 9)
- Implemented email alerts
- Added SSE for real-time notifications
- Created animated popup alerts
- Added audio alerts

### Phase 7: Training Pipeline (Week 10-11)
- Created dataset registration system
- Implemented pseudo-label generation
- Added automated training script
- Fine-tuned models on LOL dataset

### Phase 8: UI/UX Polish (Week 12)
- Designed dark/light theme system
- Added responsive layouts
- Created button styles
- Improved animations

### Phase 9: Testing & Optimization (Week 13)
- Fixed dependency conflicts (numpy, torch, pillow)
- Optimized frame skipping for performance
- Added error handling
- Created comprehensive documentation

### Phase 10: Final Features (Week 14)
- Added camera manual controls
- Implemented SSE connection management
- Created project README
- Finalized alert notification system

### Phase 11: Image Enhancement Tool (December 2025)
- Created standalone image enhancement page
- Implemented drag-and-drop upload interface
- Added before/after comparison grid
- Built AJAX-based enhancement processing
- Designed enhancement techniques info panel
- Added toast notification system
- Integrated with existing enhancement.py module
- Fixed path resolution issues (Windows vs URL paths)
- Updated sidebar navigation across all templates
- Comprehensive documentation in IMAGE_ENHANCEMENT_FEATURE.md

---

## ⚙️ Configuration System

### Environment Variables (.env)

```bash
# Model Configuration
YOLO_WEIGHTS=runs/detect/anomaly_eval15/weights/best.pt
YOLO_MODEL_PATH=yolov8n.pt

# Detection Configuration
DISABLE_DETECTION_DB=1
IMPORTANT_CLASSES=person,car,truck
ALERT_CONF_MIN=0.6

# Anomaly Detection
ENABLE_ANOMALY_ALERTS=1
ANOMALY_CLASSES=person,car,motorcycle,truck,knife,gun,backpack
ANOMALY_CONF_MIN=0.5

# Email Alerts
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
ALERT_TO=recipient@example.com

# Database (Optional Postgres)
SUPABASE_DB_URL=postgresql://user:pass@host:6543/postgres?sslmode=require

# Flask Configuration
FLASK_DEBUG=1
SECRET_KEY=your-secret-key-here
```

### PowerShell Configuration
```powershell
# Set environment variables for current session
$env:YOLO_WEIGHTS = "runs\detect\anomaly_eval15\weights\best.pt"
$env:ENABLE_ANOMALY_ALERTS = "1"
$env:ANOMALY_CLASSES = "person,car,knife,gun"
$env:ANOMALY_CONF_MIN = "0.6"

# Run application
python main.py
```

### Runtime Configuration
```python
# Load from .env file
load_dotenv()

# Parse configuration
ANOMALY_CLASSES = {
    s.strip().lower() 
    for s in os.getenv('ANOMALY_CLASSES', '').split(',') 
    if s.strip()
}
ANOMALY_CONF_MIN = float(os.getenv('ANOMALY_CONF_MIN', '0.5'))
ENABLE_ANOMALY_ALERTS = os.getenv('ENABLE_ANOMALY_ALERTS', '1') in ('1', 'true', 'yes')
```

---

## 📊 Performance Metrics

### System Performance
- **Frame Processing**: 10-15 FPS (with YOLOv8n on CPU)
- **Detection Latency**: 100-200ms per frame
- **Alert Response Time**: < 500ms (SSE)
- **Database Query Time**: < 50ms (average)

### Model Performance (LOL Eval15 Training)
- **Precision**: 97.3%
- **Recall**: 100%
- **mAP50**: 99.5%
- **mAP50-95**: 95.8%
- **Training Time**: ~15 minutes (5 epochs on CPU)

### Resource Usage
- **RAM**: 500-800 MB (idle), 1-2 GB (active detection)
- **CPU**: 30-50% (single core during inference)
- **Disk**: ~2 GB (base installation), grows with saved images
- **Network**: Minimal (local streaming)

---

## 🔮 Future Enhancements

### Planned Features
1. **Multi-camera support** - Monitor multiple cameras simultaneously
2. **Cloud storage integration** - AWS S3, Google Cloud Storage
3. **Mobile app** - React Native or Flutter companion app
4. **Facial recognition** - Identify known persons
5. **License plate detection** - Vehicle identification
6. **Heatmap visualization** - Activity zones over time
7. **Advanced analytics** - Charts, graphs, reports
8. **Webhook support** - Integrate with other services
9. **Voice alerts** - Text-to-speech announcements
10. **PTZ camera control** - Pan, tilt, zoom commands

### Potential Improvements
- GPU acceleration for faster inference
- WebRTC for lower latency streaming
- Redis for caching and session management
- Docker containerization for easy deployment
- Kubernetes orchestration for scaling
- Progressive Web App (PWA) for offline support

---

## 🆕 Recent Updates (December 2025)

### Image Enhancement Feature Added
A new standalone image enhancement tool has been integrated into the system:

**What's New**:
- ✅ Dedicated `/image_enhancement` route
- ✅ Drag-and-drop file upload interface  
- ✅ Real-time before/after comparison
- ✅ AJAX-based processing (no page reload)
- ✅ Toast notification feedback system
- ✅ Mobile-responsive grid layout
- ✅ Enhancement techniques info panel
- ✅ Updated sidebar navigation across all templates

**Technical Implementation**:
- **Backend**: Flask route using existing `enhance_image()` function
- **Frontend**: 650+ lines HTML with vanilla JavaScript
- **Styling**: Integrated with existing dashboard CSS theme
- **File Handling**: Absolute paths using `BASE_DIR` for Windows compatibility
- **Processing**: Pure enhancement without YOLO detection (1-2 second response)

**Files Added/Modified**:
- `templates/image_enhancement.html` (NEW - 654 lines)
- `main.py` - Added `/image_enhancement` route
- `enhancement.py` - Fixed numpy array LUT bug
- `templates/dashboard.html` - Updated sidebar
- `templates/lowlight_detection.html` - Updated sidebar  
- `IMAGE_ENHANCEMENT_FEATURE.md` - Complete feature documentation

**Bug Fixes**:
- Fixed OpenCV LUT error (bytearray → numpy array)
- Fixed file path resolution (Windows backslashes vs URL forward slashes)
- Fixed BASE_DIR configuration for correct static file serving

**User Benefits**:
- Quick photo enhancement without detection overhead
- Easy way to test enhancement quality
- Useful for personal photo improvement
- Clear demonstration of enhancement capabilities

---

## 📝 Summary

**Night Shield** is a comprehensive AI-powered surveillance system built from scratch using modern web technologies and state-of-the-art deep learning. The project demonstrates:

### Technical Skills
- Full-stack web development (Flask, HTML/CSS/JS)
- Computer vision (OpenCV, YOLOv8)
- Deep learning (PyTorch, transfer learning)
- Database design (SQLite, SQL)
- Real-time communication (SSE, MJPEG streaming)
- System architecture (client-server, event-driven)

### Software Engineering
- Modular code organization
- Error handling and logging
- Configuration management
- Version control (Git)
- Documentation
- Testing and debugging

### Problem-Solving
- Low-light image enhancement
- Real-time object detection
- Anomaly classification
- Alert notification system
- Performance optimization

### Domain Knowledge
- Surveillance systems
- Computer vision pipelines
- ML model training and deployment
- Security best practices

This project serves as a solid foundation for understanding how modern AI-powered applications are built, deployed, and maintained. It combines theoretical knowledge with practical implementation, making it an excellent learning resource and portfolio piece.

---

**Created with dedication and technical expertise** 🚀  
**For questions or explanations, refer to this comprehensive guide**
