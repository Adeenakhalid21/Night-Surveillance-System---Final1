import cv2
from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, session
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re, os
from ultralytics import YOLO
import smtplib
from email.message import EmailMessage
from enhancement import enhance_image
import mysql.connector
from mysql.connector import errorcode
import random
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import threading
import time

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


def detect_objects_and_classify(frame):
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
            
            # Save the detected frame as an image
            att = 'static/images/detected_frame.jpg'
            cv2.imwrite(att, frame)

            # Send email alert in a separate thread
            subject = "Motion Detected!"
            to = "user.nightshield@gmail.com"
            body = f"Motion of {class_name} has been detected by the surveillance system."
            threading.Thread(target=send_email_alert, args= (subject, body,  to ,  att)).start()

    return frame

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

    user = "owork7864@gmail.com"
    msg['from'] = user
    password = "gtuhwrtdcfizlkkz"

    with open(att, 'rb') as img_file:
        img_data = img_file.read()
        msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename='detected_frame.jpg')

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
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

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'nss'

mysql = MySQL(app)

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

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM user WHERE email = %s', (email,))
        account = cursor.fetchone()

        if account:
            mesage = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            mesage = 'Invalid email address!'
        elif not firstName or not password or not email:
            mesage = 'Please fill out the form!'
        else:
            cursor.execute('INSERT INTO user VALUES (NULL, %s, %s, %s, %s)', (firstName, lastname, email, password,))
            mysql.connection.commit()
            mesage = 'You have successfully registered!'
    elif request.method == 'POST':
        mesage = 'Please fill out the form!'
    return render_template('home.html', mesage = mesage)


# Login API
@app.route('/login', methods=['GET', 'POST'])
def login():
    mesage = ''
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email = request.form['email']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM user WHERE email = %s AND password = %s', (email, password,))
        user = cursor.fetchone()
        if user:
            session['loggedin'] = True
            session['sno'] = user['sno']
            session['firstname'] = user['firstname']
            session['email'] = user['email']
            mesage = 'Logged in successfully !'
            return render_template('dashboard.html', mesage = mesage)
        else:
            mesage = 'Please enter correct email id and password !'
    return render_template('home.html', mesage = mesage)


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

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM cam WHERE camname = %s', (camname,))
        account = cursor.fetchone()

        if account:
            mesage = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', camname):
            mesage = 'Invalid email address!'
        elif not camfps or not camurl or not camname:
            mesage = 'Please fill out the form!'
        else:
            cursor.execute('INSERT INTO user VALUES (NULL, %s, %s, %s, %s)', (camfps, camurl, camname,))
            mysql.connection.commit()
            mesage = 'You have successfully Added Camera!'
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

if __name__ == '__main__':
    app.run(debug=True)
