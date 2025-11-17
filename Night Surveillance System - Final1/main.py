import cv2
from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, session
import sqlite3
import re, os
from ultralytics import YOLO
from dotenv import load_dotenv
import smtplib
from email.message import EmailMessage
from enhancement import enhance_image
import random
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import threading
import time

# Load environment variables from .env if present
load_dotenv()

# Initialize YOLOv8 model
model = YOLO("yolov8n.pt") # yolov8n, yolov8m, yolov8l, yolov8x

# Global variables for motion detection
prev_frame = None
motion_detected = False

# Initialize Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/videos'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# Function for motion detection
def motion_detection(frame1, frame2):
    global motion_detected
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    motion_detected = len(contours) > 0


def detect_objects_and_classify(frame, camera_id=1):
    # Perform object detection
    results = model(frame)

    # Define the classes you want to detect
    desired_classes = ["person", "car", "motorcycle", "bicycle", "bus", "truck"]

    # Iterate through detected objects
    for detection in results[0].boxes:
        x1, y1, x2, y2, conf, cls = int(detection.xyxy[0][0]), int(detection.xyxy[0][1]), int(detection.xyxy[0][2]), int(detection.xyxy[0][3]), float(detection.conf[0]), int(detection.cls[0])
        class_name = model.names[int(cls)]
        label = f"{class_name} {conf:.2f}"

        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Filter out classes not in desired_classes
        if class_name in desired_classes:
            
            # Save the detected frame as an image with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            detection_image_path = f'static/images/detection_{timestamp}_{class_name}.jpg'
            cv2.imwrite(detection_image_path, frame)
            
            # Save detection to dataset
            save_detection_to_dataset_db({
                'camera_id': camera_id,
                'object_class': class_name,
                'confidence': conf,
                'bbox_x': x1,
                'bbox_y': y1,
                'bbox_width': x2 - x1,
                'bbox_height': y2 - y1,
                'image_path': detection_image_path,
                'dataset_id': 1  # Person Detection Dataset ID
            })
            
            # Log surveillance event
            log_surveillance_event_db({
                'camera_id': camera_id,
                'event_type': f'{class_name}_detected',
                'severity': 'medium',
                'description': f'{class_name.title()} detected with {conf:.2f} confidence',
                'image_path': detection_image_path
            })

            # Keep the original detected frame for email
            att = 'static/images/detected_frame.jpg'
            cv2.imwrite(att, frame)

            # Send email alert in a separate thread
            subject = "Motion Detected!"
            to = os.getenv("ALERT_TO", "user.nightshield@gmail.com")
            body = f"Motion of {class_name} has been detected by the surveillance system."
            threading.Thread(target=send_email_alert, args=(subject, body, to, att)).start()

    return frame

# Helper function to save detection to database
def save_detection_to_dataset_db(detection_data):
    """Save detection result to database"""
    try:
        conn = get_db_connection()
        conn.execute('''
            INSERT INTO detection_results 
            (camera_id, object_class, confidence, bbox_x, bbox_y, bbox_width, bbox_height, image_path, dataset_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (detection_data['camera_id'], detection_data['object_class'], detection_data['confidence'], 
              detection_data['bbox_x'], detection_data['bbox_y'], detection_data['bbox_width'], 
              detection_data['bbox_height'], detection_data['image_path'], detection_data.get('dataset_id')))
        
        conn.commit()
        
        # Update sample count
        if detection_data.get('dataset_id'):
            conn.execute('''
                UPDATE datasets 
                SET total_samples = (SELECT COUNT(*) FROM detection_results WHERE dataset_id = datasets.dataset_id)
                WHERE dataset_id = ?
            ''', (detection_data['dataset_id'],))
            conn.commit()
        
        conn.close()
    except Exception as e:
        print(f"Error saving detection to dataset: {e}")

# Helper function to log surveillance events
def log_surveillance_event_db(event_data):
    """Log surveillance event to database"""
    try:
        conn = get_db_connection()
        conn.execute('''
            INSERT INTO surveillance_events 
            (camera_id, event_type, severity, description, image_path, video_path)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (event_data.get('camera_id'), event_data['event_type'], event_data.get('severity', 'medium'),
              event_data.get('description'), event_data.get('image_path'), event_data.get('video_path')))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error logging surveillance event: {e}")

last_alert_time = 0
# Function to send email alerts
def send_email_alert(subject, body, to, att):
    
    global last_alert_time
    
    # Check if at least 5 seconds have passed since the last alert
    current_time = time.time()
    if current_time - last_alert_time < 50:
        print("Email alert rate limit exceeded. Skipping this alert.")
        return
    
    msg = EmailMessage()
    msg.set_content(body)
    msg['subject'] = subject
    msg['to'] = to

    user = os.getenv("SMTP_USER", "owork7864@gmail.com")
    msg['from'] = user
    password = os.getenv("SMTP_PASSWORD", "gtuhwrtdcfizlkkz")

    with open(att, 'rb') as img_file:
        img_data = img_file.read()
        msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename='detected_frame.jpg')

    try:
        host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        port = int(os.getenv("SMTP_PORT", "587"))
        server = smtplib.SMTP(host, port)
        server.starttls()
        server.login(user, password)
        server.send_message(msg)
        server.quit()
        # Update last_alert_time
        last_alert_time = current_time

    except Exception as e:
        print(f"Error sending email alert: {e}")

# Video stream generator
def video_stream(source, stop_event):
    global prev_frame, motion_detected
    
    # Capture video from webcam/ uploaded video
    cap = cv2.VideoCapture(source)
    

    # Initialize previous frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read the initial frame.")
        return
    
    prev_frame = cv2.resize(prev_frame, (640, 480))

    while not stop_event.is_set() and cap.isOpened():
        # Skip 5 frames
        for _ in range(5):
            cap.read()

        ret, frame = cap.read()
        
        if not ret:
            break

        # Resize the frame for faster processing
        frame = cv2.resize(frame, (640, 480))
        
        frame_bytes = b''  # Define frame_bytes outside the if statement
        
        if prev_frame is not None:
            motion_detection(prev_frame, frame)
            
            if motion_detected:

                # Enhance the image before processing
                enhanced_frame = enhance_image(frame)
                
                detected_frame = detect_objects_and_classify(enhanced_frame)
                
                # Convert frame to JPEG
                _, jpeg = cv2.imencode('.jpg', detected_frame)
                frame_bytes = jpeg.tobytes()
        
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        if motion_detected:
            prev_frame = frame.copy()  # Update previous frame if motion is detected

    cap.release()

# Route for home page
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/upload')
def upload():
    video_path = request.args.get('video_path')
    return render_template('upload_video.html', video_path=video_path)

@app.route('/upload_video', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        if 'video' not in request.files:
            return 'No video file uploaded'
        video = request.files['video']
        if video.filename == '':
            return 'No video file selected'
        if video and allowed_file(video.filename):
            filename = video.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            video.save(file_path)
            return redirect(url_for('upload', video_path=file_path))
        return 'Invalid File Type'
    return render_template('upload_video.html')

#seperate route for uploaded video processing 
@app.route('/upload_video_feed')
def upload_video_feed():
    video_path = request.args.get('video_path')
    if video_path:
        stop_event = threading.Event()
        return Response(video_stream(video_path, stop_event), mimetype='multipart/x-mixed-replace; boundary=frame')
    return 'No video path provided'

app.secret_key = 'mysecretkey'

# SQLite database configuration
DATABASE = 'night_surveillance.db'

# Function to get database connection
def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

# Initialize database
def init_db():
    conn = get_db_connection()
    
    # Create users table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS user (
            sno INTEGER PRIMARY KEY AUTOINCREMENT,
            firstname TEXT NOT NULL,
            lastname TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    
    # Create cameras table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS cam (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camname TEXT UNIQUE NOT NULL,
            camurl TEXT NOT NULL,
            camfps TEXT NOT NULL
        )
    ''')
    
    # Create datasets table for managing surveillance datasets
    conn.execute('''
        CREATE TABLE IF NOT EXISTS datasets (
            dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_name TEXT UNIQUE NOT NULL,
            dataset_type TEXT NOT NULL,
            description TEXT,
            created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_samples INTEGER DEFAULT 0,
            file_path TEXT,
            is_active BOOLEAN DEFAULT 1
        )
    ''')
    
    # Create detection_results table for storing AI detection results
    conn.execute('''
        CREATE TABLE IF NOT EXISTS detection_results (
            result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            object_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            bbox_x REAL,
            bbox_y REAL,
            bbox_width REAL,
            bbox_height REAL,
            image_path TEXT,
            dataset_id INTEGER,
            FOREIGN KEY (camera_id) REFERENCES cam (id),
            FOREIGN KEY (dataset_id) REFERENCES datasets (dataset_id)
        )
    ''')
    
    # Create training_data table for machine learning datasets
    conn.execute('''
        CREATE TABLE IF NOT EXISTS training_data (
            sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER NOT NULL,
            image_path TEXT NOT NULL,
            annotation_path TEXT,
            label TEXT,
            category TEXT,
            is_validated BOOLEAN DEFAULT 0,
            added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (dataset_id) REFERENCES datasets (dataset_id)
        )
    ''')
    
    # Create surveillance_events table for logging security events
    conn.execute('''
        CREATE TABLE IF NOT EXISTS surveillance_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id INTEGER,
            event_type TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            severity TEXT DEFAULT 'medium',
            description TEXT,
            image_path TEXT,
            video_path TEXT,
            is_resolved BOOLEAN DEFAULT 0,
            FOREIGN KEY (camera_id) REFERENCES cam (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database when app starts
init_db()

# Signup API
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    mesage=''
    if request.method == 'POST' and 'firstname' in request.form and 'password' in request.form:
        lastname = request.form['lastname']
        firstName = request.form['firstname']
        password = request.form['password']
        email = request.form['email']

        print("Get data successfully")

        conn = get_db_connection()
        account = conn.execute('SELECT * FROM user WHERE email = ?', (email,)).fetchone()

        if account:
            mesage = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            mesage = 'Invalid email address!'
        elif not firstName or not password or not email:
            mesage = 'Please fill out the form!'
        else:
            conn.execute('INSERT INTO user (firstname, lastname, email, password) VALUES (?, ?, ?, ?)', 
                        (firstName, lastname, email, password))
            conn.commit()
            mesage = 'You have successfully registered!'
        
        conn.close()
    elif request.method == 'POST':
        mesage = 'Please fill out the form!'
    return render_template('home.html', mesage = mesage)


# Login API
@app.route('/login', methods=['GET', 'POST'])
def login():
    mesage = ''
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        raw_identifier = (request.form.get('email') or '').strip()
        password = (request.form.get('password') or '').strip()

        if not raw_identifier or not password:
            mesage = 'Please provide both email/username and password.'
            return render_template('home.html', mesage=mesage)

        conn = get_db_connection()
        try:
            # If input looks like an email, match on email; otherwise allow firstname as a simple username
            identifier = raw_identifier.lower()
            if '@' in identifier:
                user = conn.execute('SELECT * FROM user WHERE LOWER(email) = ? AND password = ?',
                                    (identifier, password)).fetchone()
            else:
                user = conn.execute('SELECT * FROM user WHERE LOWER(firstname) = ? AND password = ?',
                                    (identifier, password)).fetchone()

            if user:
                session['loggedin'] = True
                session['sno'] = user['sno']
                session['firstname'] = user['firstname']
                session['email'] = user['email']
                mesage = 'Logged in successfully!'
                return render_template('dashboard.html', mesage=mesage)
            else:
                mesage = 'Invalid credentials. Use your email or first name with the correct password.'
        finally:
            conn.close()
    return render_template('home.html', mesage=mesage)


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/addCamera')
def addCamera():
    return render_template('addCamera.html')

@app.route('/cam', methods=['GET', 'POST'])
def cam():
    mesage=''
    if request.method == 'POST' and 'camname' in request.form and 'camurl' in request.form:
        camname = request.form['camname']
        camurl = request.form['camurl']
        camfps = request.form['camfps']

        conn = get_db_connection()
        existing_cam = conn.execute('SELECT * FROM cam WHERE camname = ?', (camname,)).fetchone()

        if existing_cam:
            mesage = 'Camera with this name already exists!'
        elif not camfps or not camurl or not camname:
            mesage = 'Please fill out the form!'
        else:
            conn.execute('INSERT INTO cam (camname, camurl, camfps) VALUES (?, ?, ?)', 
                        (camname, camurl, camfps))
            conn.commit()
            mesage = 'You have successfully Added Camera!'
        
        conn.close()
    elif request.method == 'POST':
        mesage = 'Please fill out the form!'
    return render_template('dashboard.html', mesage = mesage)

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/night-shield-legal')
def night_shield_legal():
    return render_template('night-shield-legal.html')

@app.route('/services')
def services():
    return render_template('services.html')

# Route for video stream
@app.route('/video_feed')
def video_feed():
    stop_event = threading.Event()
    return Response(video_stream(0, stop_event), mimetype='multipart/x-mixed-replace; boundary=frame')

# Dataset Management Routes
@app.route('/datasets')
def datasets():
    """Display all datasets"""
    if 'loggedin' not in session:
        return redirect(url_for('index'))
    
    conn = get_db_connection()
    datasets = conn.execute('SELECT * FROM datasets ORDER BY created_date DESC').fetchall()
    conn.close()
    
    return render_template('datasets.html', datasets=datasets)

@app.route('/add_dataset', methods=['GET', 'POST'])
def add_dataset():
    """Add new dataset"""
    if 'loggedin' not in session:
        return redirect(url_for('index'))
    
    message = ''
    if request.method == 'POST':
        dataset_name = request.form['dataset_name']
        dataset_type = request.form['dataset_type']
        description = request.form.get('description', '')
        file_path = request.form.get('file_path', '')
        
        if not dataset_name or not dataset_type:
            message = 'Please fill out all required fields!'
        else:
            conn = get_db_connection()
            try:
                conn.execute('''
                    INSERT INTO datasets (dataset_name, dataset_type, description, file_path) 
                    VALUES (?, ?, ?, ?)
                ''', (dataset_name, dataset_type, description, file_path))
                conn.commit()
                message = 'Dataset added successfully!'
                
                # Create directory for dataset if specified
                if file_path:
                    os.makedirs(file_path, exist_ok=True)
                    
            except Exception as e:
                message = f'Error adding dataset: {str(e)}'
            finally:
                conn.close()
    
    return render_template('add_dataset.html', message=message)

@app.route('/dataset_details/<int:dataset_id>')
def dataset_details(dataset_id):
    """Display dataset details and samples"""
    if 'loggedin' not in session:
        return redirect(url_for('index'))
    
    conn = get_db_connection()
    
    # Get dataset info
    dataset = conn.execute('SELECT * FROM datasets WHERE dataset_id = ?', (dataset_id,)).fetchone()
    
    if not dataset:
        conn.close()
        return "Dataset not found", 404
    
    # Get training data samples
    samples = conn.execute('''
        SELECT * FROM training_data 
        WHERE dataset_id = ? 
        ORDER BY added_date DESC 
        LIMIT 100
    ''', (dataset_id,)).fetchall()
    
    # Get detection results if it's a detection dataset
    detections = conn.execute('''
        SELECT dr.*, c.camname 
        FROM detection_results dr
        LEFT JOIN cam c ON dr.camera_id = c.id
        WHERE dr.dataset_id = ? 
        ORDER BY dr.timestamp DESC 
        LIMIT 50
    ''', (dataset_id,)).fetchall()
    
    conn.close()
    
    return render_template('dataset_details.html', 
                         dataset=dataset, 
                         samples=samples, 
                         detections=detections)

@app.route('/save_detection_to_dataset', methods=['POST'])
def save_detection_to_dataset():
    """Save detection result to dataset"""
    data = request.get_json()
    
    conn = get_db_connection()
    try:
        conn.execute('''
            INSERT INTO detection_results 
            (camera_id, object_class, confidence, bbox_x, bbox_y, bbox_width, bbox_height, image_path, dataset_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (data['camera_id'], data['object_class'], data['confidence'], 
              data['bbox_x'], data['bbox_y'], data['bbox_width'], data['bbox_height'],
              data['image_path'], data.get('dataset_id')))
        
        conn.commit()
        
        # Update sample count
        conn.execute('''
            UPDATE datasets 
            SET total_samples = (SELECT COUNT(*) FROM detection_results WHERE dataset_id = datasets.dataset_id)
            WHERE dataset_id = ?
        ''', (data.get('dataset_id'),))
        conn.commit()
        
        return jsonify({'status': 'success', 'message': 'Detection saved to dataset'})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})
    finally:
        conn.close()

@app.route('/log_surveillance_event', methods=['POST'])
def log_surveillance_event():
    """Log surveillance event"""
    data = request.get_json()
    
    conn = get_db_connection()
    try:
        conn.execute('''
            INSERT INTO surveillance_events 
            (camera_id, event_type, severity, description, image_path, video_path)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (data.get('camera_id'), data['event_type'], data.get('severity', 'medium'),
              data.get('description'), data.get('image_path'), data.get('video_path')))
        
        conn.commit()
        return jsonify({'status': 'success', 'message': 'Event logged successfully'})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})
    finally:
        conn.close()

@app.route('/surveillance_events')
def surveillance_events():
    """Display surveillance events"""
    if 'loggedin' not in session:
        return redirect(url_for('index'))
    
    conn = get_db_connection()
    events = conn.execute('''
        SELECT se.*, c.camname 
        FROM surveillance_events se
        LEFT JOIN cam c ON se.camera_id = c.id
        ORDER BY se.timestamp DESC
        LIMIT 100
    ''').fetchall()
    conn.close()
    
    return render_template('surveillance_events.html', events=events)

@app.route('/dataset_analytics')
def dataset_analytics():
    """Display dataset analytics and statistics"""
    if 'loggedin' not in session:
        return redirect(url_for('index'))
    
    conn = get_db_connection()
    
    # Get dataset statistics
    dataset_stats = conn.execute('''
        SELECT 
            dataset_type,
            COUNT(*) as dataset_count,
            SUM(total_samples) as total_samples
        FROM datasets 
        GROUP BY dataset_type
    ''').fetchall()
    
    # Get detection statistics
    detection_stats = conn.execute('''
        SELECT 
            object_class,
            COUNT(*) as detection_count,
            AVG(confidence) as avg_confidence
        FROM detection_results 
        GROUP BY object_class
        ORDER BY detection_count DESC
    ''').fetchall()
    
    # Get recent activity
    recent_activity = conn.execute('''
        SELECT 
            'Detection' as type, 
            object_class as description, 
            timestamp,
            confidence as value
        FROM detection_results 
        UNION ALL
        SELECT 
            'Event' as type, 
            event_type as description, 
            timestamp,
            NULL as value
        FROM surveillance_events
        ORDER BY timestamp DESC
        LIMIT 20
    ''').fetchall()
    
    conn.close()
    
    return render_template('dataset_analytics.html', 
                         dataset_stats=dataset_stats,
                         detection_stats=detection_stats,
                         recent_activity=recent_activity)

if __name__ == '__main__':
    app.run(debug=True)
