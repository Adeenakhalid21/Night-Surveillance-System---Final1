import cv2
from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, session
import sqlite3
import re, os
import json
from dotenv import load_dotenv
import smtplib
from email.message import EmailMessage
from enhancement import enhance_image, enhance_night_image, enhancement_stage_keys
import random
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import threading
import time
from datetime import timedelta
from collections import deque
import uuid
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from werkzeug.utils import secure_filename

# Load environment variables from .env if present
load_dotenv()

# Lazy-load YOLO to avoid import issues blocking app startup
def resolve_default_weights() -> str:
    env_weights = (os.getenv("YOLO_WEIGHTS") or "").strip()
    if env_weights:
        if os.path.exists(env_weights) or env_weights.lower().startswith("yolov8"):
            return env_weights
        print(f"[YOLO] configured weights not found: {env_weights}, using best available local checkpoint")

    candidates = [
        "runs/detect/gun_knife_finetune_v4/weights/best.pt",
        "runs/detect/gun_knife_hand_ft/weights/best.pt",
        "runs/detect/dataset2_weapon_detection_best.pt",
        "runs/detect/gun_knife_binary/weights/best.pt",
        "runs/detect/coco_sample_pseudo/weights/best.pt",
        "yolov8n.pt",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return "yolov8n.pt"


weights_path = resolve_default_weights()
model = None

def get_model():
    global model
    if model is not None:
        return model
    try:
        from ultralytics import YOLO
        print("[YOLO] Loading model...", flush=True)
        model = YOLO(weights_path)
        print("[YOLO] Model ready", flush=True)
        return model
    except KeyboardInterrupt:
        print("[YOLO] Load interrupted")
        return None
    except Exception as e:
        print(f"[YOLO] Failed to load: {e}")
        return None

# Global variables for motion detection
prev_frame = None
motion_detected = False

# Global variable for anomaly alerts
recent_anomalies = deque(maxlen=200)
anomalies_lock = threading.Lock()
anomaly_event_counter = 0
anomaly_last_emit = {}

# Runtime throttles for detection side effects (DB writes, snapshots, event logs)
detection_runtime_lock = threading.Lock()
detection_last_db_emit = {}
detection_last_alert_emit = {}
detection_last_snapshot_emit = {}

# Async image enhancement jobs
enhancement_jobs = {}
enhancement_jobs_lock = threading.Lock()

# Initialize Flask application
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'mysecretkey')
app.permanent_session_lifetime = timedelta(days=30)

REMEMBER_COOKIE_NAME = 'remember_token'
REMEMBER_COOKIE_MAX_AGE = 30 * 24 * 60 * 60
REMEMBER_COOKIE_SALT = 'nightwatch-remember'


def _remember_serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(app.secret_key, salt=REMEMBER_COOKIE_SALT)


def create_remember_token(username: str) -> str:
    return _remember_serializer().dumps({'username': username})


def parse_remember_token(token: str) -> str:
    try:
        payload = _remember_serializer().loads(token, max_age=REMEMBER_COOKIE_MAX_AGE)
        username = (payload or {}).get('username', '')
        return username if isinstance(username, str) else ''
    except (BadSignature, SignatureExpired):
        return ''
    except Exception:
        return ''


def get_remembered_username_from_request() -> str:
    token = request.cookies.get(REMEMBER_COOKIE_NAME, '')
    return parse_remember_token(token) if token else ''


def _restore_session_from_remember_cookie() -> bool:
    remembered_identifier = get_remembered_username_from_request()
    if not remembered_identifier:
        return False

    conn = get_db_connection()
    try:
        user = conn.execute('SELECT * FROM user WHERE LOWER(email) = ?', (remembered_identifier.lower(),)).fetchone()
        if not user:
            return False

        session['loggedin'] = True
        session['sno'] = user['sno']
        session['firstname'] = user['firstname']
        session['email'] = user['email']
        session.permanent = True
        return True
    except Exception as exc:
        print(f"[AUTH] Remember-me restore failed: {exc}")
        return False
    finally:
        conn.close()

# Get the directory where main.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'videos')
app.config['LOWLIGHT_FOLDER'] = os.path.join(BASE_DIR, 'static', 'lowlight_uploads')
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}
app.config['ALLOWED_IMAGES'] = {'jpg', 'jpeg', 'png', 'bmp'}


def _parse_label_set(raw_value: str) -> set[str]:
    return {s.strip().lower() for s in (raw_value or '').split(',') if s.strip()}


DISABLE_DETECTION_DB = str(os.getenv('DISABLE_DETECTION_DB', '1')).strip().lower() in ('1','true','yes','on')
IMPORTANT_CLASSES = _parse_label_set(os.getenv('IMPORTANT_CLASSES', 'person,backpack,knife,gun')) or {'person', 'backpack', 'knife', 'gun'}
ALERT_CONF_MIN = float(os.getenv('ALERT_CONF_MIN', '0.6'))

# Anomaly detection classes (abnormal behaviors/objects)
ANOMALY_CLASSES = _parse_label_set(os.getenv('ANOMALY_CLASSES', 'knife,gun,weapon,fire,fight,intruder')) or {'knife', 'gun', 'weapon', 'fire', 'fight', 'intruder'}
ANOMALY_MATCH_ALL = bool(ANOMALY_CLASSES.intersection({'*', 'all'}))
ANOMALY_CONF_MIN = float(os.getenv('ANOMALY_CONF_MIN', '0.5'))
ANOMALY_MIN_BOX_AREA = int(float(os.getenv('ANOMALY_MIN_BOX_AREA', '1200')))
ANOMALY_COOLDOWN_SEC = float(os.getenv('ANOMALY_COOLDOWN_SEC', '8'))
ENABLE_ANOMALY_ALERTS = str(os.getenv('ENABLE_ANOMALY_ALERTS', '1')).strip().lower() in ('1','true','yes','on')

# Stream/detection performance tuning
STREAM_FRAME_WIDTH = int(float(os.getenv('STREAM_FRAME_WIDTH', '640')))
STREAM_FRAME_HEIGHT = int(float(os.getenv('STREAM_FRAME_HEIGHT', '480')))
STREAM_GRAB_SKIP = max(0, int(float(os.getenv('STREAM_GRAB_SKIP', '1'))))
STREAM_JPEG_QUALITY = int(max(50, min(95, float(os.getenv('STREAM_JPEG_QUALITY', '75')))))

DETECTION_MIN_INTERVAL_SEC = float(os.getenv('DETECTION_MIN_INTERVAL_SEC', '0.35'))
DETECTION_DB_COOLDOWN_SEC = float(os.getenv('DETECTION_DB_COOLDOWN_SEC', '2.0'))
DETECTION_ALERT_COOLDOWN_SEC = float(os.getenv('DETECTION_ALERT_COOLDOWN_SEC', '12.0'))
DETECTION_SNAPSHOT_COOLDOWN_SEC = float(os.getenv('DETECTION_SNAPSHOT_COOLDOWN_SEC', '4.0'))
MOTION_MIN_CONTOUR_AREA = float(os.getenv('MOTION_MIN_CONTOUR_AREA', '850'))
DETECTION_CONF_MIN = float(os.getenv('DETECTION_CONF_MIN', '0.35'))
DETECTION_DISPLAY_CONF_MIN = float(os.getenv('DETECTION_DISPLAY_CONF_MIN', '0.22'))
TRACK_IOU_THRESHOLD = float(os.getenv('TRACK_IOU_THRESHOLD', '0.32'))
TRACK_BBOX_EMA_ALPHA = float(os.getenv('TRACK_BBOX_EMA_ALPHA', '0.56'))
TRACK_CONF_EMA_ALPHA = float(os.getenv('TRACK_CONF_EMA_ALPHA', '0.42'))
TRACK_MIN_HITS = max(1, int(float(os.getenv('TRACK_MIN_HITS', '2'))))
TRACK_STALE_SEC = float(os.getenv('TRACK_STALE_SEC', '1.2'))
TRACK_CONF_DECAY_PER_SEC = float(os.getenv('TRACK_CONF_DECAY_PER_SEC', '0.22'))

DESIRED_CLASSES = {
    'person', 'car', 'motorcycle', 'bicycle', 'bus', 'truck', 'knife', 'gun'
}


@app.before_request
def auto_login_from_remember_cookie():
    if session.get('loggedin'):
        return

    public_endpoints = {
        'index', 'login', 'signup', 'services', 'contact', 'about',
        'night_shield_legal', 'static',
    }
    endpoint = request.endpoint or ''
    if endpoint in public_endpoints:
        return

    _restore_session_from_remember_cookie()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def allowed_image(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_IMAGES']


def _normalize_label(label: str) -> str:
    return str(label or '').strip().lower()


def _is_anomaly_target(class_name: str) -> bool:
    normalized = _normalize_label(class_name)
    if not normalized:
        return False
    if ANOMALY_MATCH_ALL:
        return True
    return normalized in ANOMALY_CLASSES


def _enqueue_anomaly_event(class_name: str, confidence: float, camera_id: int = 1) -> bool:
    if not ENABLE_ANOMALY_ALERTS:
        return False

    if not _is_anomaly_target(class_name):
        return False

    now = time.time()
    normalized = _normalize_label(class_name)

    global anomaly_event_counter
    with anomalies_lock:
        last_seen = anomaly_last_emit.get(normalized, 0.0)
        if now - last_seen < ANOMALY_COOLDOWN_SEC:
            return False

        anomaly_last_emit[normalized] = now
        anomaly_event_counter += 1
        recent_anomalies.append({
            'id': anomaly_event_counter,
            'class': class_name,
            'confidence': float(confidence),
            'timestamp': now,
            'camera_id': camera_id,
        })
    return True


def _allow_with_cooldown(bucket: dict, key: str, cooldown_sec: float) -> bool:
    now = time.time()
    with detection_runtime_lock:
        last_time = bucket.get(key, 0.0)
        if now - last_time < cooldown_sec:
            return False
        bucket[key] = now
        return True


def _bbox_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def _blend_bbox(prev_bbox, new_bbox, alpha: float):
    a = max(0.0, min(1.0, float(alpha)))
    return tuple(((1.0 - a) * float(prev_bbox[i])) + (a * float(new_bbox[i])) for i in range(4))


class SimpleObjectTracker:
    def __init__(
        self,
        iou_threshold: float,
        bbox_alpha: float,
        conf_alpha: float,
        min_hits: int,
        stale_sec: float,
        display_conf_min: float,
        conf_decay_per_sec: float,
    ) -> None:
        self.iou_threshold = float(max(0.05, min(0.9, iou_threshold)))
        self.bbox_alpha = float(max(0.05, min(0.95, bbox_alpha)))
        self.conf_alpha = float(max(0.05, min(0.95, conf_alpha)))
        self.min_hits = int(max(1, min_hits))
        self.stale_sec = float(max(0.2, stale_sec))
        self.display_conf_min = float(max(0.01, min(0.95, display_conf_min)))
        self.conf_decay_per_sec = float(max(0.0, conf_decay_per_sec))
        self._next_track_id = 1
        self._tracks = {}

    def _prune_stale(self, now: float) -> None:
        stale_ids = [
            tid for tid, track in self._tracks.items()
            if (now - float(track.get('last_seen', 0.0))) > self.stale_sec
        ]
        for tid in stale_ids:
            self._tracks.pop(tid, None)

    def update(self, detections: list[dict], now: float | None = None) -> list[dict]:
        now = float(now if now is not None else time.time())
        self._prune_stale(now)

        normalized_detections = []
        for det in detections or []:
            try:
                bbox = tuple(float(v) for v in det['bbox'])
                conf = float(det.get('confidence', 0.0))
                class_name = str(det.get('class') or '')
            except Exception:
                continue
            if not class_name:
                continue
            normalized_detections.append({
                'class': class_name,
                'class_key': _normalize_label(class_name),
                'confidence': conf,
                'bbox': bbox,
                'bbox_area': max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]),
            })

        candidate_pairs = []
        for track_id, track in self._tracks.items():
            track_class = track.get('class_key', '')
            if not track_class:
                continue
            for det_idx, det in enumerate(normalized_detections):
                if det['class_key'] != track_class:
                    continue
                iou = _bbox_iou(track['bbox'], det['bbox'])
                if iou >= self.iou_threshold:
                    candidate_pairs.append((iou, track_id, det_idx))

        candidate_pairs.sort(key=lambda item: item[0], reverse=True)

        matched_tracks = set()
        matched_dets = set()
        for _, track_id, det_idx in candidate_pairs:
            if track_id in matched_tracks or det_idx in matched_dets:
                continue
            track = self._tracks.get(track_id)
            if not track:
                continue

            det = normalized_detections[det_idx]
            track['bbox'] = _blend_bbox(track['bbox'], det['bbox'], self.bbox_alpha)
            track['confidence'] = ((1.0 - self.conf_alpha) * float(track['confidence'])) + (self.conf_alpha * float(det['confidence']))
            track['class'] = det['class']
            track['class_key'] = det['class_key']
            track['bbox_area'] = det['bbox_area']
            track['last_seen'] = now
            track['hits'] = int(track.get('hits', 0)) + 1

            matched_tracks.add(track_id)
            matched_dets.add(det_idx)

        for det_idx, det in enumerate(normalized_detections):
            if det_idx in matched_dets:
                continue
            track_id = self._next_track_id
            self._next_track_id += 1
            self._tracks[track_id] = {
                'id': track_id,
                'class': det['class'],
                'class_key': det['class_key'],
                'confidence': det['confidence'],
                'bbox': det['bbox'],
                'bbox_area': det['bbox_area'],
                'created_at': now,
                'last_seen': now,
                'hits': 1,
            }

        return self.get_active_tracks(now)

    def get_active_tracks(self, now: float | None = None) -> list[dict]:
        now = float(now if now is not None else time.time())
        self._prune_stale(now)

        active = []
        for track_id, track in self._tracks.items():
            age_since_seen = max(0.0, now - float(track.get('last_seen', now)))
            decayed_conf = max(0.0, float(track.get('confidence', 0.0)) - (age_since_seen * self.conf_decay_per_sec))

            if decayed_conf < self.display_conf_min:
                continue

            if int(track.get('hits', 1)) < self.min_hits and age_since_seen > (DETECTION_MIN_INTERVAL_SEC * 0.75):
                continue

            x1, y1, x2, y2 = [int(round(v)) for v in track['bbox']]
            if x2 <= x1 or y2 <= y1:
                continue

            active.append({
                'track_id': track_id,
                'track_hits': int(track.get('hits', 1)),
                'track_age_sec': age_since_seen,
                'class': track.get('class', ''),
                'confidence': decayed_conf,
                'bbox': (x1, y1, x2, y2),
                'bbox_area': max(0, x2 - x1) * max(0, y2 - y1),
            })

        return active


def _run_detection_inference(frame):
    m = get_model()
    if m is None:
        return []

    results = m(frame, verbose=False)
    parsed = []
    for detection in results[0].boxes:
        x1 = int(detection.xyxy[0][0])
        y1 = int(detection.xyxy[0][1])
        x2 = int(detection.xyxy[0][2])
        y2 = int(detection.xyxy[0][3])
        conf = float(detection.conf[0])
        if conf < DETECTION_CONF_MIN:
            continue
        cls = int(detection.cls[0])
        class_name = m.names[int(cls)]
        class_key = _normalize_label(class_name)
        if class_key and class_key not in DESIRED_CLASSES and not _is_anomaly_target(class_name):
            continue
        parsed.append({
            'class': class_name,
            'confidence': conf,
            'bbox': (x1, y1, x2, y2),
            'bbox_area': max(0, x2 - x1) * max(0, y2 - y1),
        })
    return parsed


def _new_enhancement_job(job_id: str) -> dict:
    stages = [
        {'key': item['key'], 'label': item['label'], 'status': 'pending'}
        for item in enhancement_stage_keys()
    ]
    return {
        'job_id': job_id,
        'status': 'queued',
        'progress': 0,
        'stages': stages,
        'result': None,
        'error': None,
    }


def _compute_job_progress(stages: list[dict]) -> int:
    if not stages:
        return 0
    completed = sum(1 for stage in stages if stage.get('status') == 'done')
    return int((completed / len(stages)) * 100)


def _update_enhancement_stage(job_id: str, stage_key: str, stage_status: str) -> None:
    with enhancement_jobs_lock:
        job = enhancement_jobs.get(job_id)
        if not job:
            return
        for stage in job['stages']:
            if stage.get('key') == stage_key:
                stage['status'] = stage_status
                break
        if stage_status == 'running':
            job['status'] = 'processing'
        job['progress'] = _compute_job_progress(job['stages'])


def _mark_enhancement_failed(job_id: str, error_message: str) -> None:
    with enhancement_jobs_lock:
        job = enhancement_jobs.get(job_id)
        if not job:
            return
        for stage in job['stages']:
            if stage.get('status') == 'running':
                stage['status'] = 'failed'
        job['status'] = 'failed'
        job['error'] = error_message


def _mark_enhancement_done(job_id: str, result_payload: dict) -> None:
    with enhancement_jobs_lock:
        job = enhancement_jobs.get(job_id)
        if not job:
            return
        for stage in job['stages']:
            if stage.get('status') != 'done':
                stage['status'] = 'done'
        job['status'] = 'completed'
        job['progress'] = 100
        job['result'] = result_payload


def _process_enhancement_job(job_id: str, original_path: str, original_filename: str, settings: dict) -> None:
    try:
        def progress_callback(stage_key: str, stage_status: str) -> None:
            _update_enhancement_stage(job_id, stage_key, stage_status)

        outcome = enhance_night_image(original_path, settings=settings, progress_callback=progress_callback)

        enhanced_img = outcome['enhanced_image']
        comparison_img = outcome['comparison_image']
        stats = outcome['stats']
        upscale_meta = outcome.get('upscale_meta', {})

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        safe_name = secure_filename(original_filename) or f'image_{timestamp}.jpg'
        stem = os.path.splitext(safe_name)[0]

        enhanced_name = f'enhanced_{timestamp}_{stem}.jpg'
        comparison_name = f'comparison_{timestamp}_{stem}.jpg'

        enhanced_path = os.path.join(app.config['LOWLIGHT_FOLDER'], enhanced_name)
        comparison_path = os.path.join(app.config['LOWLIGHT_FOLDER'], comparison_name)

        enhanced_img.save(enhanced_path, format='JPEG', quality=95)
        comparison_img.save(comparison_path, format='JPEG', quality=95)

        size_bytes = os.path.getsize(enhanced_path)
        size_kb = round(size_bytes / 1024, 1)

        _mark_enhancement_done(
            job_id,
            {
                'original_url': f"/static/lowlight_uploads/{os.path.basename(original_path)}",
                'enhanced_url': f"/static/lowlight_uploads/{enhanced_name}",
                'comparison_url': f"/static/lowlight_uploads/{comparison_name}",
                'enhanced_size_kb': size_kb,
                'download_name': f'enhanced_{stem}.jpg',
                'stats': stats,
                'upscale_meta': upscale_meta,
            },
        )
    except Exception as exc:
        _mark_enhancement_failed(job_id, f'Enhancement failed: {exc}')

def detect_anomalies(frame, camera_id=1, detections=None):
    """Detect anomaly candidates in frame using YOLO."""
    if detections is None:
        detections = _run_detection_inference(frame)

    if not detections:
        return frame, []

    anomalies_detected = []

    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        conf = float(detection['confidence'])
        class_name = detection['class']
        bbox_area = int(detection.get('bbox_area', max(0, x2 - x1) * max(0, y2 - y1)))
        
        # Check if detected class is in anomaly list
        if conf >= ANOMALY_CONF_MIN and bbox_area >= ANOMALY_MIN_BOX_AREA and _is_anomaly_target(class_name):
            anomalies_detected.append({
                'class': class_name,
                'confidence': conf,
                'bbox': (x1, y1, x2, y2)
            })
            
            # Draw red bounding box for anomalies
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            label = f"ANOMALY: {class_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    return frame, anomalies_detected


# Function for motion detection
def motion_detection(frame1, frame2):
    global motion_detected
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    motion_detected = any(cv2.contourArea(c) >= MOTION_MIN_CONTOUR_AREA for c in contours)


def detect_objects_and_classify(frame, camera_id=1, detections=None):
    if detections is None:
        detections = _run_detection_inference(frame)

    if not detections:
        return frame

    # Save at most one frame snapshot per processed frame when needed.
    frame_snapshot_path = None
    snapshot_dir_ready = False

    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        conf = float(detection['confidence'])
        class_name = detection['class']
        class_key = _normalize_label(class_name)
        track_id = int(detection.get('track_id', 0))
        track_hits = int(detection.get('track_hits', 1))
        track_age_sec = float(detection.get('track_age_sec', 0.0))

        if class_key not in DESIRED_CLASSES:
            continue

        # Ignore weak stale tracks so visuals remain stable and meaningful.
        if conf < DETECTION_DISPLAY_CONF_MIN:
            continue

        if track_hits < TRACK_MIN_HITS and conf < max(DETECTION_CONF_MIN + 0.1, 0.55):
            continue

        track_suffix = f" #{track_id}" if track_id > 0 else ""
        label = f"{class_name}{track_suffix} {conf:.2f}"

        if track_age_sec > max(0.25, DETECTION_MIN_INTERVAL_SEC):
            box_color = (0, 205, 255)
        else:
            box_color = (0, 255, 0)

        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

        needs_db = not DISABLE_DETECTION_DB and _allow_with_cooldown(
            detection_last_db_emit,
            f"{camera_id}:{class_key}",
            DETECTION_DB_COOLDOWN_SEC,
        )
        needs_alert = class_key in IMPORTANT_CLASSES and conf >= ALERT_CONF_MIN and _allow_with_cooldown(
            detection_last_alert_emit,
            f"{camera_id}:{class_key}",
            DETECTION_ALERT_COOLDOWN_SEC,
        )

        needs_snapshot = needs_db or needs_alert
        detection_image_path = None

        if needs_snapshot:
            if frame_snapshot_path is None and _allow_with_cooldown(
                detection_last_snapshot_emit,
                f"{camera_id}:snapshot",
                DETECTION_SNAPSHOT_COOLDOWN_SEC,
            ):
                if not snapshot_dir_ready:
                    os.makedirs('static/images', exist_ok=True)
                    snapshot_dir_ready = True
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                frame_snapshot_path = f'static/images/detection_{timestamp}.jpg'
                try:
                    cv2.imwrite(frame_snapshot_path, frame)
                except Exception as e:
                    print(f"Failed to save detection image: {e}")
                    frame_snapshot_path = None

            detection_image_path = frame_snapshot_path

            # Optionally store detection rows (disabled by default)
            if needs_db:
                save_detection_to_dataset_db({
                    'camera_id': camera_id,
                    'object_class': class_name,
                    'confidence': conf,
                    'bbox_x': x1,
                    'bbox_y': y1,
                    'bbox_width': x2 - x1,
                    'bbox_height': y2 - y1,
                    'image_path': detection_image_path,
                    'dataset_id': 1
                })


            # Only log/alert important items
            if needs_alert:
                log_surveillance_event_db({
                    'camera_id': camera_id,
                    'event_type': f'{class_name}_detected',
                    'severity': 'medium',
                    'description': f'{class_name.title()} detected with {conf:.2f} confidence',
                    'image_path': detection_image_path
                })

                # Send email alert in a separate thread
                if detection_image_path:
                    subject = "Motion Detected!"
                    to = os.getenv("ALERT_TO", "user.nightshield@gmail.com")
                    body = f"Motion of {class_name} has been detected by the surveillance system."
                    threading.Thread(target=send_email_alert, args=(subject, body, to, detection_image_path)).start()

    return frame

# Helper function to save detection to database
def save_detection_to_dataset_db(detection_data):
    """Save detection result to database"""
    try:
        if DISABLE_DETECTION_DB:
            return
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
    
    # Capture video from webcam/uploaded video and keep driver buffer shallow.
    cap = cv2.VideoCapture(source)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    

    # Initialize previous frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read the initial frame.")
        return
    
    prev_frame = cv2.resize(prev_frame, (STREAM_FRAME_WIDTH, STREAM_FRAME_HEIGHT))
    last_detection_ts = 0.0
    tracker = SimpleObjectTracker(
        iou_threshold=TRACK_IOU_THRESHOLD,
        bbox_alpha=TRACK_BBOX_EMA_ALPHA,
        conf_alpha=TRACK_CONF_EMA_ALPHA,
        min_hits=TRACK_MIN_HITS,
        stale_sec=TRACK_STALE_SEC,
        display_conf_min=DETECTION_DISPLAY_CONF_MIN,
        conf_decay_per_sec=TRACK_CONF_DECAY_PER_SEC,
    )

    while not stop_event.is_set() and cap.isOpened():
        # Skip/grab frames without decoding all of them to reduce processing backlog.
        for _ in range(STREAM_GRAB_SKIP):
            cap.grab()

        ret, frame = cap.read()
        if not ret:
            break

        # Resize for consistent processing speed
        frame = cv2.resize(frame, (STREAM_FRAME_WIDTH, STREAM_FRAME_HEIGHT))

        # Default: send raw frame (ensures stream never blanks)
        output_frame = frame

        if prev_frame is not None:
            motion_detection(prev_frame, frame)
            try:
                now = time.time()
                should_run_detection = (now - last_detection_ts) >= DETECTION_MIN_INTERVAL_SEC

                if should_run_detection:
                    raw_detections = _run_detection_inference(frame)
                    tracked_detections = tracker.update(raw_detections, now=now)
                    last_detection_ts = now
                else:
                    tracked_detections = tracker.get_active_tracks(now=now)

                if tracked_detections:
                    detection_frame = frame.copy()
                    output_frame = detect_objects_and_classify(detection_frame, detections=tracked_detections)
                    output_frame, anomalies = detect_anomalies(output_frame, detections=tracked_detections)

                    emitted_anomalies = []
                    for anomaly in anomalies:
                        if _enqueue_anomaly_event(anomaly['class'], anomaly['confidence']):
                            emitted_anomalies.append(anomaly)

                    if emitted_anomalies:
                        print(f"[ANOMALY] Emitted {len(emitted_anomalies)} alert(s): {[a['class'] for a in emitted_anomalies]}")
                        os.makedirs('static/images/anomalies', exist_ok=True)

                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        anomaly_image_path = f'static/images/anomalies/anomaly_{timestamp}.jpg'
                        anomaly_image_saved = False

                        try:
                            anomaly_image_saved = bool(cv2.imwrite(anomaly_image_path, output_frame))
                        except Exception as e:
                            anomaly_image_saved = False
                            print(f"Failed to save anomaly image: {e}")

                        for anomaly in emitted_anomalies:
                            class_name = anomaly['class']
                            conf = anomaly['confidence']
                            try:
                                log_surveillance_event_db({
                                    'camera_id': 1,
                                    'event_type': 'anomaly_detected',
                                    'severity': 'high',
                                    'description': f'Anomaly detected: {class_name} with {conf:.2f} confidence',
                                    'image_path': anomaly_image_path if anomaly_image_saved else None,
                                    'video_path': None
                                })
                            except Exception as e:
                                print(f"Failed to log anomaly: {e}")

                        if ENABLE_ANOMALY_ALERTS and anomaly_image_saved:
                            summary = ", ".join(
                                f"{a['class']} ({a['confidence']:.2f})" for a in emitted_anomalies
                            )
                            subject = "⚠️ ANOMALY DETECTED!"
                            to = os.getenv("ALERT_TO", "user.nightshield@gmail.com")
                            body = (
                                "ANOMALY ALERT: "
                                f"{summary} at {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                                "Please check the surveillance system immediately."
                            )
                            threading.Thread(target=send_email_alert, args=(subject, body, to, anomaly_image_path)).start()
            except Exception as e:
                print(f"Detection error: {e}")
                output_frame = frame

        # Always encode a frame (prevents broken image tag showing alt text)
        try:
            _, jpeg = cv2.imencode(
                '.jpg',
                output_frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_JPEG_QUALITY],
            )
            frame_bytes = jpeg.tobytes()
        except Exception as e:
            print(f"JPEG encode failed: {e}")
            frame_bytes = b''

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Update previous frame every loop for motion comparison
        prev_frame = frame.copy()

    cap.release()

# Route for home page
@app.route('/')
def index():
    remembered_username = ''
    token = request.cookies.get(REMEMBER_COOKIE_NAME, '')
    if token:
        remembered_username = parse_remember_token(token)
    return render_template('home.html', remembered_username=remembered_username)

@app.route('/upload')
def upload():
    if 'loggedin' not in session:
        return redirect(url_for('index'))
    video_path = request.args.get('video_path')
    return render_template('upload_video.html', video_path=video_path)

@app.route('/upload_video', methods=['GET', 'POST'])
def upload_video():
    if 'loggedin' not in session:
        return redirect(url_for('index'))
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
    if 'loggedin' not in session:
        return redirect(url_for('index'))
    video_path = request.args.get('video_path')
    if video_path:
        stop_event = threading.Event()
        return Response(video_stream(video_path, stop_event), mimetype='multipart/x-mixed-replace; boundary=frame')
    return 'No video path provided'

# SQLite database configuration
DATABASE = 'night_surveillance.db'
USE_PG = bool(os.getenv('SUPABASE_DB_URL') or os.getenv('DATABASE_URL'))

class PGCursor:
    def __init__(self, cur):
        self._cur = cur
    def fetchone(self):
        return self._cur.fetchone()
    def fetchall(self):
        return self._cur.fetchall()

class PGConn:
    def __init__(self, conn):
        self._conn = conn
        self.row_factory = None
    def execute(self, sql, params=None):
        sql_ps = sql.replace('?', '%s')
        # Quote reserved table name user for Postgres using regex, covers start/end and joins
        def _quote_user_table(s: str) -> str:
            patterns = [
                (r'(?i)(\bfrom\s+)user(\b)', r'\1"user"\2'),
                (r'(?i)(\binsert\s+into\s+)user(\b)', r'\1"user"\2'),
                (r'(?i)(\bupdate\s+)user(\b)', r'\1"user"\2'),
                (r'(?i)(\bdelete\s+from\s+)user(\b)', r'\1"user"\2'),
                (r'(?i)(\bjoin\s+)user(\b)', r'\1"user"\2'),
            ]
            for p, rpl in patterns:
                s = re.sub(p, rpl, s)
            return s
        sql_ps = _quote_user_table(sql_ps)
        import psycopg2.extras
        cur = self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql_ps, params or ())
        return PGCursor(cur)
    def commit(self):
        self._conn.commit()
    def rollback(self):
        self._conn.rollback()
    def close(self):
        self._conn.close()

# Function to get database connection
def get_db_connection():
    if USE_PG:
        import psycopg2, psycopg2.extras
        dsn = os.getenv('SUPABASE_DB_URL') or os.getenv('DATABASE_URL')
        if not dsn:
            raise RuntimeError('DATABASE_URL/SUPABASE_DB_URL not set')
        if 'sslmode=' not in dsn:
            dsn = dsn + ('&' if '?' in dsn else '?') + 'sslmode=require'
        conn = psycopg2.connect(dsn, cursor_factory=psycopg2.extras.RealDictCursor)
        return PGConn(conn)
    else:
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
if USE_PG:
    print('[DB] Using Supabase Postgres connection.')
    # Simple connectivity test
    try:
        c = get_db_connection()
        test_row = c.execute('SELECT 1 as ok').fetchone()
        if test_row:
            print(f"[DB] Postgres connectivity OK: {test_row}")
            c.close()
    except Exception as e:
        print(f"[DB] Postgres connectivity test failed: {e}")
else:
    print('[DB] Using local SQLite database night_surveillance.db')
    init_db()

# Signup API
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    mesage=''
    remembered_username = get_remembered_username_from_request()
    if request.method == 'POST':
        lastname = (request.form.get('lastname') or '').strip()
        firstName = (request.form.get('firstname') or '').strip()
        password = request.form.get('password') or ''
        email = (request.form.get('email') or '').strip().lower()

        print("Get data successfully")

        if not firstName or not password or not email:
            mesage = 'Please fill out the form!'
            return render_template('home.html', mesage=mesage, remembered_username=remembered_username)
        if not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            mesage = 'Invalid email address!'
            return render_template('home.html', mesage=mesage, remembered_username=remembered_username)

        conn = get_db_connection()
        try:
            account = conn.execute('SELECT * FROM user WHERE LOWER(email) = ?', (email,)).fetchone()

            if account:
                mesage = 'Account already exists!'
            else:
                conn.execute('INSERT INTO user (firstname, lastname, email, password) VALUES (?, ?, ?, ?)', 
                            (firstName, lastname, email, password))
                conn.commit()
                mesage = 'You have successfully registered!'
        except Exception as e:
            print(f"[AUTH] Signup failed: {e}")
            mesage = 'Registration failed. Please try again.'
        finally:
            conn.close()
    return render_template('home.html', mesage=mesage, remembered_username=remembered_username)


# Login API
@app.route('/login', methods=['GET', 'POST'])
def login():
    mesage = ''
    remembered_username = get_remembered_username_from_request()
    if request.method == 'POST':
        raw_identifier = (request.form.get('email') or request.form.get('username') or '').strip()
        password = request.form.get('password') or ''
        remember_me = (request.form.get('remember_me') or '').strip().lower() in ('on', '1', 'true', 'yes')

        if not raw_identifier or not password.strip():
            mesage = 'Please provide both email/username and password.'
            return render_template('home.html', mesage=mesage, remembered_username=raw_identifier or remembered_username)

        conn = get_db_connection()
        try:
            identifier = raw_identifier.lower()
            user = None
            if '@' in identifier:
                # Email login
                user = conn.execute('SELECT * FROM user WHERE LOWER(email) = ?', (identifier,)).fetchone()
            else:
                # First or last name login fallback
                user = conn.execute('SELECT * FROM user WHERE LOWER(firstname) = ? OR LOWER(lastname) = ?', (identifier, identifier)).fetchone()

            if not user:
                mesage = 'User not found. Please register first or check spelling.'
            else:
                if user['password'] != password:
                    mesage = 'Password incorrect. Please try again.'
                    try:
                        log_surveillance_event_db({
                            'camera_id': None,
                            'event_type': 'login_failed',
                            'severity': 'low',
                            'description': f"Login failed (bad password) for '{identifier}'",
                            'image_path': None,
                            'video_path': None
                        })
                    except Exception:
                        pass
                else:
                    session['loggedin'] = True
                    session['sno'] = user['sno']
                    session['firstname'] = user['firstname']
                    session['email'] = user['email']
                    session.permanent = remember_me
                    mesage = 'Logged in successfully!'
                    try:
                        log_surveillance_event_db({
                            'camera_id': None,
                            'event_type': 'login_success',
                            'severity': 'low',
                            'description': f"Login success for {user['email']}",
                            'image_path': None,
                            'video_path': None
                        })
                    except Exception:
                        pass
                    response = redirect(url_for('dashboard'))
                    if remember_me:
                        remember_token = create_remember_token(user['email'])
                        response.set_cookie(
                            REMEMBER_COOKIE_NAME,
                            remember_token,
                            max_age=REMEMBER_COOKIE_MAX_AGE,
                            httponly=True,
                            secure=bool(request.is_secure),
                            samesite='Lax',
                        )
                    else:
                        response.delete_cookie(REMEMBER_COOKIE_NAME)
                    return response
        except Exception as e:
            print(f"[AUTH] Login failed: {e}")
            mesage = 'Login failed due to a server error. Please try again.'
        finally:
            conn.close()
    if request.method == 'POST' and raw_identifier:
        remembered_username = raw_identifier
    return render_template('home.html', mesage=mesage, remembered_username=remembered_username)


@app.route('/logout')
def logout():
    session.clear()
    response = redirect(url_for('index'))
    response.delete_cookie(REMEMBER_COOKIE_NAME)
    return response


@app.route('/lowlight_detection', methods=['GET', 'POST'])
def lowlight_detection():
    if 'loggedin' not in session:
        return redirect(url_for('index'))
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_image(file.filename):
            # Create upload directory
            os.makedirs(app.config['LOWLIGHT_FOLDER'], exist_ok=True)
            
            # Save uploaded file
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"lowlight_{timestamp}_{file.filename}"
            filepath = os.path.join(app.config['LOWLIGHT_FOLDER'], filename)
            file.save(filepath)
            
            # Read and enhance the image
            img = cv2.imread(filepath)
            if img is None:
                return jsonify({'error': 'Failed to read image'}), 500
            
            # Enhance low-light image
            try:
                enhanced_img = enhance_image(img)
            except Exception as e:
                print(f"Enhancement failed: {e}")
                enhanced_img = img
            
            # Detect objects in enhanced image
            m = get_model()
            if m is None:
                return jsonify({'error': 'Model not loaded'}), 500
            
            results = m(enhanced_img)
            detections = []
            
            # Draw detections on image
            for detection in results[0].boxes:
                x1, y1, x2, y2, conf, cls = int(detection.xyxy[0][0]), int(detection.xyxy[0][1]), int(detection.xyxy[0][2]), int(detection.xyxy[0][3]), float(detection.conf[0]), int(detection.cls[0])
                class_name = m.names[int(cls)]
                
                detections.append({
                    'class': class_name,
                    'confidence': float(conf),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
                
                # Draw bounding box
                cv2.rectangle(enhanced_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(enhanced_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Save result image
            result_filename = f"result_{timestamp}.jpg"
            result_path = os.path.join(app.config['LOWLIGHT_FOLDER'], result_filename)
            cv2.imwrite(result_path, enhanced_img)
            
            return jsonify({
                'success': True,
                'detections': detections,
                'result_image': f"/static/lowlight_uploads/{result_filename}",
                'original_image': f"/static/lowlight_uploads/{filename}"
            })
        else:
            return jsonify({'error': 'Invalid file type. Please upload JPG, PNG, or BMP'}), 400
    
    # GET request - render the upload page
    return render_template('lowlight_detection.html')

@app.route('/image_enhancement', methods=['GET', 'POST'])
def image_enhancement():
    if 'loggedin' not in session:
        return redirect(url_for('index'))
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_image(image_file.filename):
            return jsonify({'error': 'Invalid file type. Please upload JPG, PNG, or BMP'}), 400

        os.makedirs(app.config['LOWLIGHT_FOLDER'], exist_ok=True)

        safe_name = secure_filename(image_file.filename) or 'uploaded_image.jpg'
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        original_name = f'original_{timestamp}_{safe_name}'
        original_path = os.path.join(app.config['LOWLIGHT_FOLDER'], original_name)
        image_file.save(original_path)

        def _safe_float(name: str, default: float) -> float:
            try:
                return float(request.form.get(name, default))
            except (TypeError, ValueError):
                return default

        settings = {
            'brightness': _safe_float('brightness', 55),
            'contrast': _safe_float('contrast', 65),
            'sharpness': _safe_float('sharpness', 60),
            'denoise': _safe_float('denoise', 55),
            'upscale': _safe_float('upscale', 70),
            'upscale_scale': request.form.get('upscale_scale', '4'),
        }

        job_id = uuid.uuid4().hex
        with enhancement_jobs_lock:
            enhancement_jobs[job_id] = _new_enhancement_job(job_id)

        worker = threading.Thread(
            target=_process_enhancement_job,
            args=(job_id, original_path, image_file.filename, settings),
            daemon=True,
        )
        worker.start()

        return jsonify({'success': True, 'job_id': job_id}), 202

    return render_template('image_enhancement.html')


@app.route('/image_enhancement/progress/<job_id>')
def image_enhancement_progress(job_id):
    with enhancement_jobs_lock:
        job = enhancement_jobs.get(job_id)
        if not job:
            return jsonify({'error': 'Enhancement job not found'}), 404
        return jsonify(job)

@app.route('/dashboard')
def dashboard():
    if 'loggedin' not in session:
        return redirect(url_for('index'))
    cam_count = dataset_count = detections_count = events_count = 0
    try:
        conn = get_db_connection()
        if USE_PG:
            cam_count = conn.execute('SELECT COUNT(*) FROM cam').fetchone()[0]
            dataset_count = conn.execute('SELECT COUNT(*) FROM datasets').fetchone()[0]
            detections_count = conn.execute('SELECT COUNT(*) FROM detection_results').fetchone()[0]
            events_count = conn.execute('SELECT COUNT(*) FROM surveillance_events').fetchone()[0]
        else:
            cam_count = conn.execute('SELECT COUNT(*) FROM cam').fetchone()[0]
            dataset_count = conn.execute('SELECT COUNT(*) FROM datasets').fetchone()[0]
            detections_count = conn.execute('SELECT COUNT(*) FROM detection_results').fetchone()[0]
            events_count = conn.execute('SELECT COUNT(*) FROM surveillance_events').fetchone()[0]
    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return render_template('dashboard.html', cam_count=cam_count, dataset_count=dataset_count, detections_count=detections_count, events_count=events_count)

@app.route('/addCamera')
def addCamera():
    if 'loggedin' not in session:
        return redirect(url_for('index'))
    return render_template('addCamera.html')

@app.route('/cam', methods=['GET', 'POST'])
def cam():
    if 'loggedin' not in session:
        return redirect(url_for('index'))
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
    if 'loggedin' not in session:
        return redirect(url_for('index'))
    stop_event = threading.Event()
    return Response(video_stream(0, stop_event), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/anomaly_alerts')
def anomaly_alerts():
    if 'loggedin' not in session:
        return redirect(url_for('index'))
    """Server-Sent Events endpoint for real-time anomaly alerts"""
    def generate():
        with anomalies_lock:
            last_sent_id = anomaly_event_counter

        while True:
            events_to_send = []
            with anomalies_lock:
                events_to_send = [
                    anomaly for anomaly in recent_anomalies
                    if anomaly.get('id', 0) > last_sent_id
                ]

            for anomaly in events_to_send:
                payload = json.dumps({
                    'id': anomaly.get('id'),
                    'class': anomaly.get('class'),
                    'confidence': round(float(anomaly.get('confidence', 0.0)), 4),
                    'timestamp': anomaly.get('timestamp'),
                    'camera_id': anomaly.get('camera_id', 1),
                })
                yield f"data: {payload}\n\n"
                last_sent_id = anomaly.get('id', last_sent_id)

            # Keep-alive comment so browsers keep SSE open even when no events.
            yield ": keep-alive\n\n"
            time.sleep(1.0)

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )

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
    
    # Get training data samples (optional table in some deployments)
    try:
        samples = conn.execute('''
            SELECT * FROM training_data 
            WHERE dataset_id = ? 
            ORDER BY added_date DESC 
            LIMIT 100
        ''', (dataset_id,)).fetchall()
    except Exception as exc:
        print(f"[DATASET] Could not load training_data samples: {exc}")
        samples = []

    category_values = set()
    for sample in samples:
        value = None
        if isinstance(sample, dict):
            value = sample.get('category')
        else:
            try:
                value = sample['category']
            except Exception:
                value = getattr(sample, 'category', None)

        if value is not None and str(value).strip():
            category_values.add(str(value).strip())
    category_count = len(category_values)
    
    # Get detection results (optional table in some deployments)
    try:
        detections = conn.execute('''
            SELECT dr.*, c.camname 
            FROM detection_results dr
            LEFT JOIN cam c ON dr.camera_id = c.id
            WHERE dr.dataset_id = ? 
            ORDER BY dr.timestamp DESC 
            LIMIT 50
        ''', (dataset_id,)).fetchall()
    except Exception as exc:
        print(f"[DATASET] Could not load detection_results: {exc}")
        detections = []
    
    conn.close()
    
    return render_template('dataset_details.html', 
                         dataset=dataset, 
                         samples=samples, 
                         detections=detections,
                         category_count=category_count)

@app.route('/save_detection_to_dataset', methods=['POST'])
def save_detection_to_dataset():
    """Save detection result to dataset"""
    data = request.get_json()
    if DISABLE_DETECTION_DB:
        return jsonify({'status': 'skipped', 'message': 'Detection persistence disabled by server config'}), 200
    
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

@app.route('/auth_logs')
def auth_logs():
    """Display authentication logs (login successes and failures)."""
    if 'loggedin' not in session:
        return redirect(url_for('index'))
    conn = get_db_connection()
    rows = conn.execute('''
        SELECT * FROM surveillance_events 
        WHERE event_type IN ('login_success','login_failed')
        ORDER BY timestamp DESC
        LIMIT 200
    ''').fetchall()
    conn.close()
    # Reuse surveillance_events template for simplicity
    return render_template('surveillance_events.html', events=rows)

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
    
    # Get detection statistics (optional table in some deployments)
    try:
        detection_stats = conn.execute('''
            SELECT 
                object_class,
                COUNT(*) as detection_count,
                AVG(confidence) as avg_confidence
            FROM detection_results 
            GROUP BY object_class
            ORDER BY detection_count DESC
        ''').fetchall()
    except Exception as exc:
        print(f"[DATASET] Could not load detection stats: {exc}")
        detection_stats = []
        try:
            conn.rollback()
        except Exception:
            pass
    
    # Get recent activity with a graceful fallback when detection_results is unavailable
    try:
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
    except Exception as exc:
        print(f"[DATASET] Could not load combined activity feed: {exc}")
        try:
            conn.rollback()
        except Exception:
            pass
        try:
            recent_activity = conn.execute('''
                SELECT 
                    'Event' as type,
                    event_type as description,
                    timestamp,
                    NULL as value
                FROM surveillance_events
                ORDER BY timestamp DESC
                LIMIT 20
            ''').fetchall()
        except Exception as fallback_exc:
            print(f"[DATASET] Could not load event-only activity feed: {fallback_exc}")
            recent_activity = []
    
    conn.close()
    
    return render_template('dataset_analytics.html', 
                         dataset_stats=dataset_stats,
                         detection_stats=detection_stats,
                         recent_activity=recent_activity)

if __name__ == '__main__':
    app.run(debug=True)
