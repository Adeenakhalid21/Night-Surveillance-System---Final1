import cv2
import numpy as np
from PIL import Image
from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, session
import sqlite3
import re, os
import json
from dotenv import load_dotenv
import smtplib
from email.message import EmailMessage
from enhancement import (
    analyze_lowlight_improvement,
    enhance_image_with_meta,
    enhance_night_image,
    enhancement_stage_keys,
)
import random
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import threading
import time
from datetime import timedelta
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import uuid
import webbrowser
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from werkzeug.utils import secure_filename

# Base directory for stable relative path resolution.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load environment variables from .env if present
load_dotenv(dotenv_path=os.path.join(BASE_DIR, '.env'))


def _resolve_local_path(path_value: str) -> str:
    if not path_value:
        return path_value
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(BASE_DIR, path_value)

# Lazy-load YOLO to avoid import issues blocking app startup
def _resolve_weights_from_env_or_candidates(env_name: str, fallback_candidates: list[str]) -> str:
    env_value = (os.getenv(env_name) or "").strip()
    if env_value:
        if env_value.lower().startswith("yolov8"):
            return env_value
        resolved_env_value = _resolve_local_path(env_value)
        if os.path.exists(resolved_env_value):
            return resolved_env_value
        print(f"[YOLO] configured {env_name} not found: {env_value}")

    for candidate in fallback_candidates:
        if candidate.lower().startswith("yolov8"):
            return candidate
        resolved_candidate = _resolve_local_path(candidate)
        if os.path.exists(resolved_candidate):
            return resolved_candidate
    return ""


general_weights_path = _resolve_weights_from_env_or_candidates(
    "GENERAL_WEIGHTS",
    [
        "runs/detect/anomaly_eval15/weights/best.pt",
        "runs/detect/anomaly_lowlight3/weights/best.pt",
        "runs/detect/coco_sample_pseudo/weights/best.pt",
        "yolov8n.pt",
    ],
)

weapon_weights_path = _resolve_weights_from_env_or_candidates(
    "WEAPON_WEIGHTS",
    [
        (os.getenv("YOLO_WEIGHTS") or "").strip(),
        "runs/detect/gun_knife_finetune_v4/weights/best.pt",
        "runs/detect/gun_knife_binary/weights/best.pt",
        "runs/detect/dataset2_weapon_detection_best.pt",
    ],
)

if not general_weights_path:
    general_weights_path = "yolov8n.pt"

# Keep this legacy variable name for compatibility with existing scripts and diagnostics.
weights_path = general_weights_path

phone_weights_path = (os.getenv("PHONE_WEIGHTS") or "").strip()
if phone_weights_path and not phone_weights_path.lower().startswith("yolov8"):
    resolved_phone_weights = _resolve_local_path(phone_weights_path)
    if os.path.exists(resolved_phone_weights):
        phone_weights_path = resolved_phone_weights
    else:
        print(f"[YOLO] configured PHONE_WEIGHTS not found: {phone_weights_path}, phone-only detections disabled")
        phone_weights_path = ""
if not phone_weights_path:
    default_phone_weights = _resolve_local_path("runs/detect/phones.pt")
    if os.path.exists(default_phone_weights):
        phone_weights_path = default_phone_weights

model = None
weapon_model = None
phone_model = None
HF_BACKEND_ALIASES = {'hf', 'transformers', 'huggingface'}
DETECTION_BACKEND = (os.getenv('DETECTION_BACKEND', 'ultralytics') or 'ultralytics').strip().lower()
HF_OBJECT_DETECTION_MODEL_ID = (os.getenv('HF_OBJECT_DETECTION_MODEL', 'ciasimbaya/ObjectDetection') or 'ciasimbaya/ObjectDetection').strip()
hf_detector_pipe = None
hf_image_processor = None
hf_object_model = None
hf_backend_failed = False

def get_model():
    global model
    if model is not None:
        return model
    try:
        from ultralytics import YOLO
        print(f"[YOLO] Loading general model from {weights_path}...", flush=True)
        model = YOLO(weights_path)
        print("[YOLO] General model ready", flush=True)
        return model
    except KeyboardInterrupt:
        print("[YOLO] Load interrupted")
        return None
    except Exception as e:
        print(f"[YOLO] Failed to load: {e}")
        return None


def get_weapon_model():
    global weapon_model
    if not weapon_weights_path:
        return None
    if weapon_model is not None:
        return weapon_model
    try:
        from ultralytics import YOLO
        print(f"[YOLO] Loading weapon model from {weapon_weights_path}...", flush=True)
        weapon_model = YOLO(weapon_weights_path)
        print("[YOLO] Weapon model ready", flush=True)
        return weapon_model
    except KeyboardInterrupt:
        print("[YOLO] Weapon model load interrupted")
        return None
    except Exception as e:
        print(f"[YOLO] Weapon model unavailable: {e}")
        return None


def get_phone_model():
    global phone_model
    if not phone_weights_path:
        return None
    if phone_model is not None:
        return phone_model
    try:
        from ultralytics import YOLO
        print(f"[YOLO] Loading phone model from {phone_weights_path}...", flush=True)
        phone_model = YOLO(phone_weights_path)
        print("[YOLO] Phone model ready", flush=True)
        return phone_model
    except KeyboardInterrupt:
        print("[YOLO] Phone model load interrupted")
        return None
    except Exception as e:
        print(f"[YOLO] Phone model unavailable: {e}")
        return None


def get_hf_detector_pipe():
    global hf_detector_pipe, hf_image_processor, hf_object_model, hf_backend_failed

    if hf_backend_failed:
        return None
    if hf_detector_pipe is not None:
        return hf_detector_pipe

    try:
        import torch
        from transformers import pipeline, AutoImageProcessor, AutoModelForObjectDetection

        device = 0 if torch.cuda.is_available() else -1
        print(f"[HF] Loading object-detection model from {HF_OBJECT_DETECTION_MODEL_ID}...", flush=True)

        hf_image_processor = AutoImageProcessor.from_pretrained(HF_OBJECT_DETECTION_MODEL_ID)
        hf_object_model = AutoModelForObjectDetection.from_pretrained(HF_OBJECT_DETECTION_MODEL_ID)
        try:
            hf_detector_pipe = pipeline(
                'object-detection',
                model=hf_object_model,
                image_processor=hf_image_processor,
                device=device,
            )
        except TypeError:
            # Backward-compatible fallback for older transformers pipeline signatures.
            hf_detector_pipe = pipeline('object-detection', model=HF_OBJECT_DETECTION_MODEL_ID, device=device)

        print("[HF] Object-detection model ready", flush=True)
        return hf_detector_pipe
    except KeyboardInterrupt:
        print("[HF] Model load interrupted")
        hf_backend_failed = True
        return None
    except Exception as exc:
        print(f"[HF] Model unavailable: {exc}")
        hf_backend_failed = True
        return None


def _run_hf_detection_inference(frame):
    hf_pipe = get_hf_detector_pipe()
    if hf_pipe is None:
        return None

    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        raw = hf_pipe(pil_image, threshold=max(0.01, min(0.95, HF_DETECTION_THRESHOLD)))
    except Exception as exc:
        print(f"[HF] Inference failed: {exc}")
        return None

    parsed = []
    for detection in raw or []:
        score = float(detection.get('score', 0.0))
        if score < HF_POST_CONF_MIN:
            continue

        box = detection.get('box') or {}
        x1 = int(round(float(box.get('xmin', 0))))
        y1 = int(round(float(box.get('ymin', 0))))
        x2 = int(round(float(box.get('xmax', 0))))
        y2 = int(round(float(box.get('ymax', 0))))
        if x2 <= x1 or y2 <= y1:
            continue

        raw_label = str(detection.get('label', '')).strip()
        class_key = _normalize_label(raw_label)
        if class_key in {'pistol', 'rifle', 'shotgun', 'handgun', 'firearm'}:
            class_key = 'gun'

        if not DETECTION_MATCH_ALL and class_key and class_key not in DESIRED_CLASSES and not _is_anomaly_target(class_key):
            continue

        class_name = 'phone' if class_key == 'phone' else (class_key or raw_label.lower())
        parsed.append({
            'class': class_name,
            'confidence': score,
            'bbox': (x1, y1, x2, y2),
            'bbox_area': max(0, x2 - x1) * max(0, y2 - y1),
        })

    return parsed

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

# Uploaded video processing sessions and summary metrics
upload_video_sessions = {}
upload_video_sessions_lock = threading.Lock()

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
        session['lastname'] = user['lastname']
        session['email'] = user['email']
        session.permanent = True
        return True
    except Exception as exc:
        print(f"[AUTH] Remember-me restore failed: {exc}")
        return False
    finally:
        conn.close()

app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'videos')
app.config['LOWLIGHT_FOLDER'] = os.path.join(BASE_DIR, 'static', 'lowlight_uploads')
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}
app.config['ALLOWED_IMAGES'] = {'jpg', 'jpeg', 'png', 'bmp'}


def _parse_label_set(raw_value: str) -> set[str]:
    aliases = {
        'cell phone': 'phone',
        'cellphone': 'phone',
        'mobile phone': 'phone',
        'mobile': 'phone',
        'phones': 'phone',
        'smartphone': 'phone',
        'handgun': 'gun',
        'firearm': 'gun',
        'pistol': 'gun',
        'rifle': 'gun',
        'shotgun': 'gun',
        'automatic rifle': 'gun',
        'smg': 'gun',
        'sniper': 'gun',
        'bazooka': 'gun',
        'grenade launcher': 'gun',
        'grenade': 'gun',
        'launcher': 'gun',
        'sword': 'knife',
        'blade': 'knife',
        'dagger': 'knife',
        'machete': 'knife',
        'human': 'person',
        'man': 'person',
        'woman': 'person',
    }
    parsed = set()
    for s in (raw_value or '').split(','):
        normalized = str(s or '').strip().lower().replace('_', ' ').replace('-', ' ')
        normalized = ' '.join(normalized.split())
        if not normalized:
            continue
        parsed.add(aliases.get(normalized, normalized))
    return parsed


DISABLE_DETECTION_DB = str(os.getenv('DISABLE_DETECTION_DB', '1')).strip().lower() in ('1','true','yes','on')
IMPORTANT_CLASSES = _parse_label_set(os.getenv('IMPORTANT_CLASSES', 'person,backpack,knife,gun,phone')) or {'person', 'backpack', 'knife', 'gun', 'phone'}
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
STREAM_GRAPH_PANEL_WIDTH = int(float(os.getenv('STREAM_GRAPH_PANEL_WIDTH', '270')))
STREAM_METRIC_HISTORY = max(20, int(float(os.getenv('STREAM_METRIC_HISTORY', '90'))))

UPLOAD_VIDEO_GRAPH_POINTS = max(40, int(float(os.getenv('UPLOAD_VIDEO_GRAPH_POINTS', '120'))))
UPLOAD_VIDEO_CONTEXT_CLASS_LIMIT = max(3, int(float(os.getenv('UPLOAD_VIDEO_CONTEXT_CLASS_LIMIT', '5'))))
LOWLIGHT_MAX_SIDE = max(320, int(float(os.getenv('LOWLIGHT_MAX_SIDE', '960'))))

DETECTION_MIN_INTERVAL_SEC = float(os.getenv('DETECTION_MIN_INTERVAL_SEC', '0.35'))
DETECTION_DB_COOLDOWN_SEC = float(os.getenv('DETECTION_DB_COOLDOWN_SEC', '2.0'))
DETECTION_ALERT_COOLDOWN_SEC = float(os.getenv('DETECTION_ALERT_COOLDOWN_SEC', '12.0'))
DETECTION_SNAPSHOT_COOLDOWN_SEC = float(os.getenv('DETECTION_SNAPSHOT_COOLDOWN_SEC', '4.0'))
MOTION_MIN_CONTOUR_AREA = float(os.getenv('MOTION_MIN_CONTOUR_AREA', '850'))
DETECTION_CONF_MIN = float(os.getenv('DETECTION_CONF_MIN', '0.35'))
DETECTION_DISPLAY_CONF_MIN = float(os.getenv('DETECTION_DISPLAY_CONF_MIN', '0.22'))
WEAPON_CONF_MIN = float(os.getenv('WEAPON_CONF_MIN', str(min(DETECTION_CONF_MIN, 0.18))))
KNIFE_CONF_MIN = float(os.getenv('KNIFE_CONF_MIN', str(WEAPON_CONF_MIN)))
GUN_CONF_MIN = float(os.getenv('GUN_CONF_MIN', str(WEAPON_CONF_MIN)))
HF_DETECTION_THRESHOLD = float(os.getenv('HF_DETECTION_THRESHOLD', str(DETECTION_CONF_MIN)))
HF_POST_CONF_MIN = float(os.getenv('HF_POST_CONF_MIN', str(min(DETECTION_CONF_MIN, HF_DETECTION_THRESHOLD, 0.18))))
HF_AUGMENT_WITH_YOLO = str(os.getenv('HF_AUGMENT_WITH_YOLO', '1')).strip().lower() in ('1', 'true', 'yes', 'on')
HF_FUSION_INCLUDE_GENERAL = str(os.getenv('HF_FUSION_INCLUDE_GENERAL', '0')).strip().lower() in ('1', 'true', 'yes', 'on')
STREAM_DETECTION_BACKEND = (os.getenv('STREAM_DETECTION_BACKEND', 'ultralytics') or 'ultralytics').strip().lower()
TRACK_IOU_THRESHOLD = float(os.getenv('TRACK_IOU_THRESHOLD', '0.32'))
TRACK_BBOX_EMA_ALPHA = float(os.getenv('TRACK_BBOX_EMA_ALPHA', '0.56'))
TRACK_CONF_EMA_ALPHA = float(os.getenv('TRACK_CONF_EMA_ALPHA', '0.42'))
TRACK_MIN_HITS = max(1, int(float(os.getenv('TRACK_MIN_HITS', '1'))))
TRACK_STALE_SEC = float(os.getenv('TRACK_STALE_SEC', '1.2'))
TRACK_CONF_DECAY_PER_SEC = float(os.getenv('TRACK_CONF_DECAY_PER_SEC', '0.22'))
DETECTION_LANE_WORKERS = max(1, min(3, int(float(os.getenv('DETECTION_LANE_WORKERS', '3')))))
ENABLE_DETECTION_SIDE_EFFECTS = str(os.getenv('ENABLE_DETECTION_SIDE_EFFECTS', '0')).strip().lower() in ('1', 'true', 'yes', 'on')
SAVE_DETECTION_FRAMES = str(os.getenv('SAVE_DETECTION_FRAMES', '0')).strip().lower() in ('1', 'true', 'yes', 'on')
SAVE_ANOMALY_FRAMES = str(os.getenv('SAVE_ANOMALY_FRAMES', '0')).strip().lower() in ('1', 'true', 'yes', 'on')

DESIRED_CLASSES = _parse_label_set(os.getenv(
    'DESIRED_CLASSES',
    'person,car,motorcycle,bicycle,bus,truck,knife,gun,phone,backpack,book,bottle,laptop,mouse,remote,tv,chair,potted plant,dining table',
)) or {
    'person', 'car', 'motorcycle', 'bicycle', 'bus', 'truck', 'knife', 'gun', 'phone',
    'backpack', 'book', 'bottle', 'laptop', 'mouse', 'remote', 'tv', 'chair', 'potted plant', 'dining table'
}
DETECTION_MATCH_ALL = bool(DESIRED_CLASSES.intersection({'*', 'all'}))
PERSON_LANE_CLASSES = {'person'}
WEAPON_LANE_CLASSES = {'knife', 'gun', 'weapon', 'pistol', 'rifle'}
_lane_worker_count = max(1, DETECTION_LANE_WORKERS)
detection_lane_pool = ThreadPoolExecutor(max_workers=_lane_worker_count, thread_name_prefix='det-lane')


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
    normalized = str(label or '').strip().lower().replace('_', ' ').replace('-', ' ')
    normalized = ' '.join(normalized.split())
    aliases = {
        'cell phone': 'phone',
        'cellphone': 'phone',
        'mobile phone': 'phone',
        'mobile': 'phone',
        'phones': 'phone',
        'smartphone': 'phone',
        'handgun': 'gun',
        'firearm': 'gun',
        'pistol': 'gun',
        'rifle': 'gun',
        'shotgun': 'gun',
        'automatic rifle': 'gun',
        'smg': 'gun',
        'sniper': 'gun',
        'bazooka': 'gun',
        'grenade launcher': 'gun',
        'grenade': 'gun',
        'launcher': 'gun',
        'sword': 'knife',
        'blade': 'knife',
        'dagger': 'knife',
        'machete': 'knife',
        'human': 'person',
        'man': 'person',
        'woman': 'person',
    }
    return aliases.get(normalized, normalized)


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


def _trim_graph_series(series: list, limit: int = UPLOAD_VIDEO_GRAPH_POINTS) -> None:
    overflow = len(series) - max(1, int(limit))
    if overflow > 0:
        del series[:overflow]


def _top_counts(counter: dict, limit: int = UPLOAD_VIDEO_CONTEXT_CLASS_LIMIT) -> list[dict]:
    ordered = sorted(counter.items(), key=lambda item: int(item[1]), reverse=True)
    return [
        {'label': str(label), 'count': int(count)}
        for label, count in ordered[:max(1, int(limit))]
    ]


def _build_upload_video_context(session_data: dict) -> dict:
    frames_processed = int(session_data.get('frames_processed', 0))
    total_anomalies = int(session_data.get('total_anomalies', 0))
    anomaly_rate = (total_anomalies / frames_processed) if frames_processed else 0.0

    if total_anomalies >= 20 or anomaly_rate >= 0.35:
        risk_level = 'high'
    elif total_anomalies >= 8 or anomaly_rate >= 0.15:
        risk_level = 'medium'
    else:
        risk_level = 'low'

    if total_anomalies <= 0:
        narrative = 'No anomaly events were detected in this upload.'
    elif risk_level == 'high':
        narrative = 'High anomaly density detected. Review this video immediately.'
    elif risk_level == 'medium':
        narrative = 'Moderate anomaly activity detected. Manual review is recommended.'
    else:
        narrative = 'Low anomaly activity detected. Continue routine monitoring.'

    return {
        'risk_level': risk_level,
        'anomaly_rate_percent': round(anomaly_rate * 100.0, 2),
        'top_detected_classes': _top_counts(session_data.get('class_counts', {})),
        'top_anomaly_classes': _top_counts(session_data.get('anomaly_counts', {})),
        'narrative': narrative,
    }


def _register_upload_video_session(video_id: str, original_name: str, stored_name: str, absolute_path: str) -> None:
    with upload_video_sessions_lock:
        upload_video_sessions[video_id] = {
            'video_id': video_id,
            'video_name': original_name,
            'stored_name': stored_name,
            'absolute_video_path': absolute_path,
            'status': 'ready',
            'created_at': time.time(),
            'started_at': None,
            'completed_at': None,
            'updated_at': time.time(),
            'frames_processed': 0,
            'total_detections': 0,
            'total_anomalies': 0,
            'peak_detections': 0,
            'peak_anomalies': 0,
            'sum_inference_ms': 0.0,
            'sum_fps': 0.0,
            'max_confidence': 0.0,
            'class_counts': {},
            'anomaly_counts': {},
            'graph': {
                'labels': [],
                'detections': [],
                'anomalies': [],
                'persons': [],
                'weapons': [],
                'objects': [],
                'inference_ms': [],
                'fps': [],
            },
            'error': '',
        }


def _reset_upload_video_session_metrics(session_data: dict) -> None:
    session_data['frames_processed'] = 0
    session_data['total_detections'] = 0
    session_data['total_anomalies'] = 0
    session_data['peak_detections'] = 0
    session_data['peak_anomalies'] = 0
    session_data['sum_inference_ms'] = 0.0
    session_data['sum_fps'] = 0.0
    session_data['max_confidence'] = 0.0
    session_data['class_counts'] = {}
    session_data['anomaly_counts'] = {}
    session_data['graph'] = {
        'labels': [],
        'detections': [],
        'anomalies': [],
        'persons': [],
        'weapons': [],
        'objects': [],
        'inference_ms': [],
        'fps': [],
    }


def _mark_upload_video_session_started(video_id: str) -> None:
    with upload_video_sessions_lock:
        session_data = upload_video_sessions.get(video_id)
        if not session_data:
            return
        _reset_upload_video_session_metrics(session_data)
        session_data['status'] = 'processing'
        session_data['started_at'] = time.time()
        session_data['completed_at'] = None
        session_data['error'] = ''
        session_data['updated_at'] = time.time()


def _update_upload_video_session(
    video_id: str,
    metrics: dict,
    tracked_detections: list[dict],
    anomalies: list[dict],
) -> None:
    with upload_video_sessions_lock:
        session_data = upload_video_sessions.get(video_id)
        if not session_data:
            return

        frames_processed = int(session_data.get('frames_processed', 0)) + 1
        total_count = int(metrics.get('total_count', len(tracked_detections)))
        anomaly_count = int(metrics.get('anomaly_count', len(anomalies)))
        inference_ms = float(metrics.get('inference_ms', 0.0))
        stream_fps = float(metrics.get('fps', 0.0))

        session_data['frames_processed'] = frames_processed
        session_data['total_detections'] = int(session_data.get('total_detections', 0)) + total_count
        session_data['total_anomalies'] = int(session_data.get('total_anomalies', 0)) + anomaly_count
        session_data['peak_detections'] = max(int(session_data.get('peak_detections', 0)), total_count)
        session_data['peak_anomalies'] = max(int(session_data.get('peak_anomalies', 0)), anomaly_count)
        session_data['sum_inference_ms'] = float(session_data.get('sum_inference_ms', 0.0)) + inference_ms
        session_data['sum_fps'] = float(session_data.get('sum_fps', 0.0)) + stream_fps

        max_conf = max([float(det.get('confidence', 0.0)) for det in tracked_detections] + [0.0])
        session_data['max_confidence'] = max(float(session_data.get('max_confidence', 0.0)), max_conf)

        class_counts = session_data.setdefault('class_counts', {})
        for det in tracked_detections:
            label = _normalize_label(det.get('class', '')) or 'unknown'
            class_counts[label] = int(class_counts.get(label, 0)) + 1

        anomaly_counts = session_data.setdefault('anomaly_counts', {})
        for anomaly in anomalies:
            label = _normalize_label(anomaly.get('class', '')) or 'unknown'
            anomaly_counts[label] = int(anomaly_counts.get(label, 0)) + 1

        graph = session_data.setdefault('graph', {})
        labels = graph.setdefault('labels', [])
        detections_series = graph.setdefault('detections', [])
        anomalies_series = graph.setdefault('anomalies', [])
        persons_series = graph.setdefault('persons', [])
        weapons_series = graph.setdefault('weapons', [])
        objects_series = graph.setdefault('objects', [])
        inference_series = graph.setdefault('inference_ms', [])
        fps_series = graph.setdefault('fps', [])

        labels.append(str(frames_processed))
        detections_series.append(total_count)
        anomalies_series.append(anomaly_count)
        persons_series.append(int(metrics.get('person_count', 0)))
        weapons_series.append(int(metrics.get('weapon_count', 0)))
        objects_series.append(int(metrics.get('object_count', 0)))
        inference_series.append(round(inference_ms, 2))
        fps_series.append(round(stream_fps, 2))

        _trim_graph_series(labels)
        _trim_graph_series(detections_series)
        _trim_graph_series(anomalies_series)
        _trim_graph_series(persons_series)
        _trim_graph_series(weapons_series)
        _trim_graph_series(objects_series)
        _trim_graph_series(inference_series)
        _trim_graph_series(fps_series)

        session_data['updated_at'] = time.time()


def _finalize_upload_video_session(video_id: str, status: str = 'completed', error_message: str = '') -> None:
    with upload_video_sessions_lock:
        session_data = upload_video_sessions.get(video_id)
        if not session_data:
            return
        session_data['status'] = status
        session_data['completed_at'] = time.time()
        session_data['updated_at'] = time.time()
        if error_message:
            session_data['error'] = error_message


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


def _run_detection_inference(frame, backend_override: str | None = None):
    active_backend = str(backend_override or DETECTION_BACKEND or 'ultralytics').strip().lower()
    if active_backend in {'default', 'same', 'app'}:
        active_backend = DETECTION_BACKEND

    def _parse_boxes(active_model, active_results, phone_only: bool = False, weapon_only: bool = False):
        out = []
        if not active_model or not active_results:
            return out
        for detection in active_results[0].boxes:
            x1 = int(detection.xyxy[0][0])
            y1 = int(detection.xyxy[0][1])
            x2 = int(detection.xyxy[0][2])
            y2 = int(detection.xyxy[0][3])
            conf = float(detection.conf[0])
            cls = int(detection.cls[0])
            class_name = active_model.names[int(cls)]
            class_key = _normalize_label(class_name)

            required_conf = DETECTION_CONF_MIN
            if weapon_only:
                if class_key == 'knife':
                    required_conf = KNIFE_CONF_MIN
                elif class_key in {'gun', 'pistol', 'rifle', 'shotgun', 'handgun', 'firearm'}:
                    required_conf = GUN_CONF_MIN
                else:
                    required_conf = WEAPON_CONF_MIN

            if conf < required_conf:
                continue

            if phone_only and class_key != 'phone':
                continue

            if weapon_only and class_key not in WEAPON_LANE_CLASSES and class_key not in {'knife', 'gun'}:
                continue

            if not DETECTION_MATCH_ALL and class_key and class_key not in DESIRED_CLASSES and not _is_anomaly_target(class_name):
                continue

            if class_key in {'pistol', 'rifle', 'shotgun', 'handgun', 'firearm'}:
                class_name = 'gun'
            elif class_key == 'phone':
                class_name = 'phone'

            out.append({
                'class': class_name,
                'confidence': conf,
                'bbox': (x1, y1, x2, y2),
                'bbox_area': max(0, x2 - x1) * max(0, y2 - y1),
            })
        return out

    def _merge_detections(base: list[dict], incoming: list[dict]) -> list[dict]:
        merged = list(base)
        for new_det in incoming:
            replace_idx = None
            for idx, existing in enumerate(merged):
                if _normalize_label(existing.get('class', '')) != _normalize_label(new_det.get('class', '')):
                    continue
                if _bbox_iou(existing.get('bbox', (0, 0, 0, 0)), new_det.get('bbox', (0, 0, 0, 0))) >= 0.55:
                    replace_idx = idx
                    break

            if replace_idx is None:
                merged.append(new_det)
            elif float(new_det.get('confidence', 0.0)) > float(merged[replace_idx].get('confidence', 0.0)):
                merged[replace_idx] = new_det
        return merged

    parsed = []
    hf_active = False

    if active_backend in HF_BACKEND_ALIASES:
        hf_parsed = _run_hf_detection_inference(frame)
        if hf_parsed is not None:
            hf_active = True
            parsed = list(hf_parsed)
            if not HF_AUGMENT_WITH_YOLO:
                return parsed
        else:
            print('[HF] Falling back to Ultralytics ensemble backend.')

    include_general = (not hf_active) or HF_FUSION_INCLUDE_GENERAL
    general_m = get_model() if include_general else None
    if include_general and general_m is None and not parsed:
        return []

    general_results = None
    if general_m is not None:
        try:
            general_results = general_m(frame, verbose=False)
        except Exception as e:
            print(f"[YOLO] general model inference failed: {e}")

    weapon_m = get_weapon_model()
    weapon_results = None
    if weapon_m is not None and weapon_m is not general_m:
        try:
            weapon_results = weapon_m(frame, verbose=False)
        except Exception as e:
            print(f"[YOLO] weapon model inference failed: {e}")

    phone_m = get_phone_model()
    phone_results = None
    if phone_m is not None and phone_m is not general_m:
        try:
            phone_results = phone_m(frame, verbose=False)
        except Exception as e:
            print(f"[YOLO] phone model inference failed: {e}")

    parsed = _merge_detections(parsed, _parse_boxes(general_m, general_results, phone_only=False, weapon_only=False))
    parsed = _merge_detections(parsed, _parse_boxes(weapon_m, weapon_results, phone_only=False, weapon_only=True))
    parsed = _merge_detections(parsed, _parse_boxes(phone_m, phone_results, phone_only=True, weapon_only=False))

    return parsed


def _lane_name_for_class(class_name: str) -> str:
    class_key = _normalize_label(class_name)
    if class_key in PERSON_LANE_CLASSES:
        return 'person'
    if class_key in WEAPON_LANE_CLASSES:
        return 'weapon'
    return 'object'


def _collect_lane_detections(detections: list[dict], lane_name: str) -> list[dict]:
    lane_detections = []
    for detection in detections or []:
        if _lane_name_for_class(detection.get('class', '')) != lane_name:
            continue
        det_copy = dict(detection)
        det_copy['lane'] = lane_name
        lane_detections.append(det_copy)
    return lane_detections


def _run_detection_lanes(detections: list[dict]) -> dict[str, list[dict]]:
    if not detections:
        return {'object': [], 'person': [], 'weapon': []}

    if DETECTION_LANE_WORKERS <= 1:
        return {
            'object': _collect_lane_detections(detections, 'object'),
            'person': _collect_lane_detections(detections, 'person'),
            'weapon': _collect_lane_detections(detections, 'weapon'),
        }

    futures = {
        lane: detection_lane_pool.submit(_collect_lane_detections, detections, lane)
        for lane in ('object', 'person', 'weapon')
    }

    lane_results = {}
    for lane, future in futures.items():
        try:
            lane_results[lane] = future.result(timeout=1.2)
        except Exception:
            lane_results[lane] = _collect_lane_detections(detections, lane)

    return lane_results


def _count_detection_groups(detections: list[dict]) -> dict[str, int]:
    counts = {'object': 0, 'person': 0, 'weapon': 0}
    for detection in detections or []:
        lane = _lane_name_for_class(detection.get('class', ''))
        counts[lane] = counts.get(lane, 0) + 1
    return counts


def _collect_anomaly_detections(detections: list[dict]) -> list[dict]:
    anomalies_detected = []
    for detection in detections or []:
        x1, y1, x2, y2 = detection['bbox']
        conf = float(detection['confidence'])
        class_name = detection['class']
        bbox_area = int(detection.get('bbox_area', max(0, x2 - x1) * max(0, y2 - y1)))

        if conf >= ANOMALY_CONF_MIN and bbox_area >= ANOMALY_MIN_BOX_AREA and _is_anomaly_target(class_name):
            anomalies_detected.append({
                'class': class_name,
                'confidence': conf,
                'bbox': (x1, y1, x2, y2)
            })
    return anomalies_detected


def _draw_sparkline(panel: np.ndarray, values, x: int, y: int, width: int, height: int, color, title: str) -> None:
    cv2.rectangle(panel, (x, y), (x + width, y + height), (52, 66, 89), 1)
    cv2.putText(panel, title, (x + 2, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (186, 204, 232), 1)

    series = np.array(list(values), dtype=np.float32)
    if series.size < 2:
        return

    max_value = float(np.max(series))
    min_value = float(np.min(series))
    if max_value - min_value < 1e-6:
        max_value = min_value + 1.0

    points = []
    span = max(1, int(series.size - 1))
    for idx, value in enumerate(series):
        px = int(x + ((idx / span) * (width - 1)))
        normalized = (value - min_value) / (max_value - min_value)
        py = int(y + height - 1 - (normalized * (height - 1)))
        points.append((px, py))

    if len(points) >= 2:
        cv2.polylines(panel, [np.array(points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)


def _append_realtime_metrics_panel(frame: np.ndarray, metrics: dict, history: dict) -> np.ndarray:
    if frame is None or frame.size == 0:
        return frame

    height, width = frame.shape[:2]
    panel_width = max(220, STREAM_GRAPH_PANEL_WIDTH)
    canvas = np.zeros((height, width + panel_width, 3), dtype=np.uint8)
    canvas[:, :width] = frame

    panel = canvas[:, width:]
    panel[:] = (17, 21, 30)
    cv2.rectangle(panel, (0, 0), (panel_width - 1, height - 1), (51, 67, 95), 2)

    fps = float(metrics.get('fps', 0.0))
    infer_ms = float(metrics.get('inference_ms', 0.0))
    total_count = int(metrics.get('detections_total', 0))
    object_count = int(metrics.get('object_count', 0))
    person_count = int(metrics.get('person_count', 0))
    weapon_count = int(metrics.get('weapon_count', 0))
    motion_state = bool(metrics.get('motion_detected', False))

    cv2.putText(panel, 'Realtime Detection', (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 245, 255), 2)
    cv2.putText(panel, f'FPS: {fps:5.1f}', (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (126, 244, 176), 1)
    cv2.putText(panel, f'Infer: {infer_ms:5.1f} ms', (14, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (121, 194, 255), 1)
    cv2.putText(panel, f'Detections: {total_count}', (14, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (252, 226, 136), 1)
    cv2.putText(panel, f'Motion: {"Yes" if motion_state else "No"}', (14, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 168, 123), 1)

    bar_y = 188
    bar_max = 70
    bars = [
        ('Obj', object_count, (94, 158, 255)),
        ('Per', person_count, (120, 235, 148)),
        ('Wpn', weapon_count, (120, 120, 255)),
    ]
    for idx, (label, count, color) in enumerate(bars):
        x0 = 20 + (idx * 70)
        bar_h = min(bar_max, int(count * 14))
        cv2.rectangle(panel, (x0, bar_y - bar_h), (x0 + 32, bar_y), color, -1)
        cv2.rectangle(panel, (x0, bar_y - bar_max), (x0 + 32, bar_y), (66, 82, 112), 1)
        cv2.putText(panel, label, (x0, bar_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 215, 242), 1)
        cv2.putText(panel, str(count), (x0, bar_y - bar_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    spark_x = 14
    spark_w = panel_width - 28
    _draw_sparkline(panel, history.get('fps', []), spark_x, 236, spark_w, 58, (118, 245, 168), 'FPS')
    _draw_sparkline(panel, history.get('inference_ms', []), spark_x, 320, spark_w, 58, (114, 200, 255), 'Inference (ms)')

    return canvas


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

    anomalies_detected = _collect_anomaly_detections(detections)

    for anomaly in anomalies_detected:
        x1, y1, x2, y2 = anomaly['bbox']
        conf = float(anomaly['confidence'])
        class_name = anomaly['class']

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


def detect_objects_and_classify(frame, camera_id=1, detections=None, apply_side_effects=True):
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

        if not DETECTION_MATCH_ALL and class_key not in DESIRED_CLASSES and not _is_anomaly_target(class_name):
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

        if not apply_side_effects:
            continue

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

        needs_snapshot = SAVE_DETECTION_FRAMES and (needs_db or needs_alert)
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
def video_stream(source, stop_event, upload_video_id: str | None = None):
    global prev_frame, motion_detected

    # Capture video from webcam/uploaded video and keep driver buffer shallow.
    cap = cv2.VideoCapture(source)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    if not cap.isOpened():
        print("Error: Could not open camera/video source.")
        if upload_video_id:
            _finalize_upload_video_session(upload_video_id, status='failed', error_message='Could not open video source.')
        return

    frame_state = {'frame': None, 'seq': 0}
    frame_lock = threading.Lock()
    frame_event = threading.Event()
    capture_failed = threading.Event()

    detection_request = {'frame': None, 'seq': 0}
    detection_request_lock = threading.Lock()
    detection_event = threading.Event()

    detection_state = {
        'tracked': [],
        'anomalies': [],
        'meta': {
            'inference_ms': 0.0,
            'total_count': 0,
            'person_count': 0,
            'weapon_count': 0,
            'object_count': 0,
            'source_seq': 0,
            'updated_at': 0.0,
        },
    }
    detection_state_lock = threading.Lock()

    tracker = SimpleObjectTracker(
        iou_threshold=TRACK_IOU_THRESHOLD,
        bbox_alpha=TRACK_BBOX_EMA_ALPHA,
        conf_alpha=TRACK_CONF_EMA_ALPHA,
        min_hits=TRACK_MIN_HITS,
        stale_sec=TRACK_STALE_SEC,
        display_conf_min=DETECTION_DISPLAY_CONF_MIN,
        conf_decay_per_sec=TRACK_CONF_DECAY_PER_SEC,
    )

    def capture_worker() -> None:
        global prev_frame
        while not stop_event.is_set() and cap.isOpened():
            try:
                # Skip/grab frames without decoding all of them to reduce processing backlog.
                for _ in range(STREAM_GRAB_SKIP):
                    if not cap.grab():
                        break

                ret, frame = cap.read()
                if not ret:
                    capture_failed.set()
                    break

                frame = cv2.resize(frame, (STREAM_FRAME_WIDTH, STREAM_FRAME_HEIGHT))
                with frame_lock:
                    frame_state['frame'] = frame
                    frame_state['seq'] = int(frame_state['seq']) + 1
                frame_event.set()

                if prev_frame is None:
                    prev_frame = frame.copy()
            except Exception as exc:
                print(f"Capture error: {exc}")
                capture_failed.set()
                break

    def detection_worker() -> None:
        local_prev_frame = None
        last_detection_ts = 0.0

        while not stop_event.is_set() and not capture_failed.is_set():
            has_pending = detection_event.wait(timeout=0.2)
            if not has_pending:
                continue
            detection_event.clear()

            with detection_request_lock:
                pending_frame = None if detection_request['frame'] is None else detection_request['frame'].copy()
                source_seq = int(detection_request.get('seq', 0))

            if pending_frame is None:
                continue

            now = time.time()
            if local_prev_frame is not None:
                try:
                    motion_detection(local_prev_frame, pending_frame)
                except Exception:
                    pass
            local_prev_frame = pending_frame.copy()

            if (now - last_detection_ts) < DETECTION_MIN_INTERVAL_SEC:
                tracked_detections = tracker.get_active_tracks(now=now)
                anomalies = _collect_anomaly_detections(tracked_detections)
                counts = _count_detection_groups(tracked_detections)
                with detection_state_lock:
                    previous_meta = detection_state.get('meta', {})
                    detection_state['tracked'] = tracked_detections
                    detection_state['anomalies'] = anomalies
                    detection_state['meta'] = {
                        'inference_ms': float(previous_meta.get('inference_ms', 0.0)),
                        'total_count': len(tracked_detections),
                        'person_count': counts['person'],
                        'weapon_count': counts['weapon'],
                        'object_count': counts['object'],
                        'source_seq': source_seq,
                        'updated_at': now,
                    }
                continue

            infer_start = time.perf_counter()
            raw_detections = _run_detection_inference(
                pending_frame,
                backend_override=STREAM_DETECTION_BACKEND,
            )
            lane_results = _run_detection_lanes(raw_detections)
            merged_detections = lane_results['object'] + lane_results['person'] + lane_results['weapon']
            tracked_detections = tracker.update(merged_detections, now=now)
            anomalies = _collect_anomaly_detections(tracked_detections)
            inference_ms = (time.perf_counter() - infer_start) * 1000.0
            last_detection_ts = now

            emitted_anomalies = []
            for anomaly in anomalies:
                if _enqueue_anomaly_event(anomaly['class'], anomaly['confidence']):
                    emitted_anomalies.append(anomaly)

            if emitted_anomalies and ENABLE_DETECTION_SIDE_EFFECTS:
                for anomaly in emitted_anomalies:
                    class_name = anomaly['class']
                    conf = anomaly['confidence']
                    try:
                        log_surveillance_event_db({
                            'camera_id': 1,
                            'event_type': 'anomaly_detected',
                            'severity': 'high',
                            'description': f'Anomaly detected: {class_name} with {conf:.2f} confidence',
                            'image_path': None,
                            'video_path': None,
                        })
                    except Exception as exc:
                        print(f"Failed to log anomaly: {exc}")

                if SAVE_ANOMALY_FRAMES:
                    os.makedirs('static/images/anomalies', exist_ok=True)
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    anomaly_image_path = f'static/images/anomalies/anomaly_{timestamp}.jpg'
                    anomaly_frame = pending_frame.copy()
                    anomaly_frame, _ = detect_anomalies(anomaly_frame, detections=tracked_detections)
                    anomaly_image_saved = False
                    try:
                        anomaly_image_saved = bool(cv2.imwrite(anomaly_image_path, anomaly_frame))
                    except Exception as exc:
                        anomaly_image_saved = False
                        print(f"Failed to save anomaly image: {exc}")

                    if ENABLE_ANOMALY_ALERTS and anomaly_image_saved:
                        summary = ", ".join(
                            f"{a['class']} ({a['confidence']:.2f})" for a in emitted_anomalies
                        )
                        subject = "ANOMALY DETECTED"
                        to = os.getenv("ALERT_TO", "user.nightshield@gmail.com")
                        body = (
                            "ANOMALY ALERT: "
                            f"{summary} at {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                            "Please check the surveillance system immediately."
                        )
                        threading.Thread(target=send_email_alert, args=(subject, body, to, anomaly_image_path), daemon=True).start()

            counts = _count_detection_groups(tracked_detections)
            with detection_state_lock:
                detection_state['tracked'] = tracked_detections
                detection_state['anomalies'] = anomalies
                detection_state['meta'] = {
                    'inference_ms': inference_ms,
                    'total_count': len(tracked_detections),
                    'person_count': counts['person'],
                    'weapon_count': counts['weapon'],
                    'object_count': counts['object'],
                    'source_seq': source_seq,
                    'updated_at': now,
                }

    capture_thread = threading.Thread(target=capture_worker, name='capture-worker', daemon=True)
    detector_thread = threading.Thread(target=detection_worker, name='detection-worker', daemon=True)
    if upload_video_id:
        _mark_upload_video_session_started(upload_video_id)
    capture_thread.start()
    detector_thread.start()

    last_sent_seq = -1
    last_emit_perf = time.perf_counter()
    fps_history = deque(maxlen=STREAM_METRIC_HISTORY)
    infer_history = deque(maxlen=STREAM_METRIC_HISTORY)
    total_history = deque(maxlen=STREAM_METRIC_HISTORY)
    person_history = deque(maxlen=STREAM_METRIC_HISTORY)
    weapon_history = deque(maxlen=STREAM_METRIC_HISTORY)

    try:
        while not stop_event.is_set():
            has_frame = frame_event.wait(timeout=1.0)
            if not has_frame:
                if capture_failed.is_set():
                    break
                continue
            frame_event.clear()

            with frame_lock:
                frame = None if frame_state['frame'] is None else frame_state['frame'].copy()
                current_seq = int(frame_state['seq'])

            if frame is None:
                continue

            if current_seq == last_sent_seq:
                continue
            last_sent_seq = current_seq

            with detection_request_lock:
                detection_request['frame'] = frame
                detection_request['seq'] = current_seq
            detection_event.set()

            with detection_state_lock:
                tracked_detections = list(detection_state.get('tracked', []))
                detection_meta = dict(detection_state.get('meta', {}))

            output_frame = frame
            rendered_anomalies = []
            if tracked_detections:
                detection_frame = frame.copy()
                output_frame = detect_objects_and_classify(
                    detection_frame,
                    detections=tracked_detections,
                    apply_side_effects=ENABLE_DETECTION_SIDE_EFFECTS,
                )
                output_frame, rendered_anomalies = detect_anomalies(output_frame, detections=tracked_detections)

            now_perf = time.perf_counter()
            stream_fps = 1.0 / max(1e-6, now_perf - last_emit_perf)
            last_emit_perf = now_perf

            total_count = int(detection_meta.get('total_count', len(tracked_detections)))
            person_count = int(detection_meta.get('person_count', 0))
            weapon_count = int(detection_meta.get('weapon_count', 0))
            object_count = int(detection_meta.get('object_count', max(0, total_count - person_count - weapon_count)))
            inference_ms = float(detection_meta.get('inference_ms', 0.0))

            if upload_video_id:
                _update_upload_video_session(
                    upload_video_id,
                    {
                        'total_count': total_count,
                        'person_count': person_count,
                        'weapon_count': weapon_count,
                        'object_count': object_count,
                        'anomaly_count': len(rendered_anomalies),
                        'inference_ms': inference_ms,
                        'fps': stream_fps,
                    },
                    tracked_detections,
                    rendered_anomalies,
                )

            fps_history.append(stream_fps)
            infer_history.append(inference_ms)
            total_history.append(total_count)
            person_history.append(person_count)
            weapon_history.append(weapon_count)

            output_frame = _append_realtime_metrics_panel(
                output_frame,
                {
                    'fps': stream_fps,
                    'inference_ms': inference_ms,
                    'detections_total': total_count,
                    'person_count': person_count,
                    'weapon_count': weapon_count,
                    'object_count': object_count,
                    'motion_detected': motion_detected,
                },
                {
                    'fps': fps_history,
                    'inference_ms': infer_history,
                    'total': total_history,
                    'person': person_history,
                    'weapon': weapon_history,
                },
            )

            try:
                _, jpeg = cv2.imencode(
                    '.jpg',
                    output_frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_JPEG_QUALITY],
                )
                frame_bytes = jpeg.tobytes()
            except Exception as exc:
                print(f"JPEG encode failed: {exc}")
                frame_bytes = b''

            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            prev_frame = frame.copy()
    finally:
        stop_event.set()
        detection_event.set()
        frame_event.set()
        if capture_thread.is_alive():
            capture_thread.join(timeout=1.2)
        if detector_thread.is_alive():
            detector_thread.join(timeout=1.2)
        cap.release()

        if upload_video_id:
            with upload_video_sessions_lock:
                session_data = upload_video_sessions.get(upload_video_id, {})
                processed_frames = int(session_data.get('frames_processed', 0))
                already_failed = str(session_data.get('status', '')) == 'failed'

            if processed_frames <= 0:
                _finalize_upload_video_session(
                    upload_video_id,
                    status='failed',
                    error_message='No frames were processed from the uploaded video.',
                )
            elif not already_failed:
                _finalize_upload_video_session(upload_video_id, status='completed')

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
    video_id = (request.args.get('video_id') or '').strip()
    message = request.args.get('message', '')
    video_session = None

    if video_id:
        with upload_video_sessions_lock:
            existing = upload_video_sessions.get(video_id)
            if existing:
                video_session = deepcopy(existing)
            else:
                message = 'Video session not found. Please upload the file again.'
                video_id = ''

    return render_template(
        'upload_video.html',
        video_id=video_id,
        video=video_session,
        message=message,
    )

@app.route('/upload_video', methods=['GET', 'POST'])
def upload_video():
    if 'loggedin' not in session:
        return redirect(url_for('index'))
    if request.method == 'POST':
        if 'video' not in request.files:
            return redirect(url_for('upload', message='No video file uploaded.'))

        video = request.files['video']
        if video.filename == '':
            return redirect(url_for('upload', message='No video file selected.'))

        if not (video and allowed_file(video.filename)):
            return redirect(url_for('upload', message='Invalid file type. Please upload MP4, AVI, MOV, or MKV.'))

        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        safe_name = secure_filename(video.filename) or f'uploaded_{int(time.time())}.mp4'
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        stored_name = f"{timestamp}_{uuid.uuid4().hex[:8]}_{safe_name}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], stored_name)

        try:
            video.save(file_path)
        except Exception as exc:
            print(f"[UPLOAD] Failed to save uploaded video: {exc}")
            return redirect(url_for('upload', message='Failed to save video file. Please try again.'))

        video_id = uuid.uuid4().hex
        _register_upload_video_session(
            video_id=video_id,
            original_name=safe_name,
            stored_name=stored_name,
            absolute_path=file_path,
        )
        return redirect(url_for('upload', video_id=video_id))

    return redirect(url_for('upload'))

# Separate route for uploaded video processing
@app.route('/upload_video_feed')
def upload_video_feed():
    if 'loggedin' not in session:
        return redirect(url_for('index'))

    video_id = (request.args.get('video_id') or '').strip()
    if not video_id:
        return 'No video ID provided', 400

    with upload_video_sessions_lock:
        video_session = upload_video_sessions.get(video_id)
        if not video_session:
            return 'Video session not found', 404
        video_path = video_session.get('absolute_video_path', '')

    if not video_path or not os.path.exists(video_path):
        _finalize_upload_video_session(video_id, status='failed', error_message='Uploaded video file is missing.')
        return 'Uploaded video file not found', 404

    stop_event = threading.Event()
    return Response(
        video_stream(video_path, stop_event, upload_video_id=video_id),
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )


@app.route('/upload_video_summary')
def upload_video_summary():
    if 'loggedin' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    video_id = (request.args.get('video_id') or '').strip()
    if not video_id:
        return jsonify({'error': 'Missing video_id'}), 400

    with upload_video_sessions_lock:
        session_data = deepcopy(upload_video_sessions.get(video_id))

    if not session_data:
        return jsonify({'error': 'Video session not found'}), 404

    frames_processed = int(session_data.get('frames_processed', 0))
    total_detections = int(session_data.get('total_detections', 0))
    total_anomalies = int(session_data.get('total_anomalies', 0))
    started_at = float(session_data.get('started_at') or session_data.get('created_at') or time.time())
    ended_at = float(session_data.get('completed_at') or time.time())
    duration_sec = max(0.0, ended_at - started_at)

    avg_detections = (total_detections / frames_processed) if frames_processed else 0.0
    avg_inference_ms = (float(session_data.get('sum_inference_ms', 0.0)) / frames_processed) if frames_processed else 0.0
    avg_fps = (float(session_data.get('sum_fps', 0.0)) / frames_processed) if frames_processed else 0.0

    payload = {
        'success': True,
        'video_id': video_id,
        'status': str(session_data.get('status', 'ready')),
        'video_name': str(session_data.get('video_name', 'uploaded_video')),
        'error': str(session_data.get('error', '')),
        'summary': {
            'frames_processed': frames_processed,
            'total_detections': total_detections,
            'total_anomalies': total_anomalies,
            'peak_detections': int(session_data.get('peak_detections', 0)),
            'peak_anomalies': int(session_data.get('peak_anomalies', 0)),
            'avg_detections_per_frame': round(avg_detections, 3),
            'avg_inference_ms': round(avg_inference_ms, 2),
            'avg_fps': round(avg_fps, 2),
            'duration_sec': round(duration_sec, 2),
            'max_confidence': round(float(session_data.get('max_confidence', 0.0)), 4),
        },
        'context': _build_upload_video_context(session_data),
        'graph': session_data.get('graph', {}),
    }
    return jsonify(payload)

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
                    session['lastname'] = user['lastname']
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
            os.makedirs(app.config['LOWLIGHT_FOLDER'], exist_ok=True)

            safe_name = secure_filename(file.filename) or f'lowlight_{int(time.time())}.jpg'
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"lowlight_{timestamp}_{safe_name}"
            filepath = os.path.join(app.config['LOWLIGHT_FOLDER'], filename)
            file.save(filepath)

            img = cv2.imread(filepath)
            if img is None:
                return jsonify({'error': 'Failed to read image'}), 500

            img_h, img_w = img.shape[:2]
            scale_ratio = min(1.0, LOWLIGHT_MAX_SIDE / float(max(img_h, img_w)))
            if scale_ratio < 1.0:
                resized_w = max(64, int(img_w * scale_ratio))
                resized_h = max(64, int(img_h * scale_ratio))
                img_for_processing = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
            else:
                img_for_processing = img.copy()

            enhancement_settings = {
                'brightness': 58,
                'contrast': 62,
                'sharpness': 52,
                'denoise': 52,
                'upscale': 45,
                'upscale_scale': '2',
            }

            lowlight_analysis = {}
            upscale_meta = {}
            try:
                enhanced_img, _, upscale_meta = enhance_image_with_meta(
                    img_for_processing,
                    settings=enhancement_settings,
                )
            except Exception as e:
                print(f"Enhancement failed: {e}")
                # Fast fallback pipeline when the enhancement model stack is unavailable.
                ycrcb = cv2.cvtColor(img_for_processing, cv2.COLOR_BGR2YCrCb)
                y, cr, cb = cv2.split(ycrcb)
                y = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(y)
                enhanced_img = cv2.cvtColor(cv2.merge((y, cr, cb)), cv2.COLOR_YCrCb2BGR)
                fallback_h, fallback_w = enhanced_img.shape[:2]
                upscale_meta = {
                    'method': 'CLAHE Fallback',
                    'scale': 1,
                    'backend': 'opencv.clahe',
                    'device': 'cpu',
                    'passes': 1,
                    'output_width': int(fallback_w),
                    'output_height': int(fallback_h),
                    'models': [],
                    'model_names': [],
                    'fallback_used': True,
                    'fallback_reason': str(e),
                }

            try:
                lowlight_analysis = analyze_lowlight_improvement(img_for_processing, enhanced_img)
            except Exception as analysis_error:
                print(f"Low-light analysis failed: {analysis_error}")
                lowlight_analysis = {}

            if get_model() is None:
                return jsonify({'error': 'Model not loaded'}), 500

            detections = _run_detection_inference(enhanced_img)
            annotated = detect_objects_and_classify(
                enhanced_img.copy(),
                detections=detections,
                apply_side_effects=False,
            )
            annotated, anomalies = detect_anomalies(annotated, detections=detections)

            enhanced_filename = f"enhanced_{timestamp}.jpg"
            enhanced_path = os.path.join(app.config['LOWLIGHT_FOLDER'], enhanced_filename)
            cv2.imwrite(enhanced_path, enhanced_img)

            result_filename = f"result_{timestamp}.jpg"
            result_path = os.path.join(app.config['LOWLIGHT_FOLDER'], result_filename)
            cv2.imwrite(result_path, annotated)

            anomaly_counts = {}
            for anomaly in anomalies:
                anomaly_key = _normalize_label(anomaly.get('class', '')) or 'unknown'
                anomaly_counts[anomaly_key] = int(anomaly_counts.get(anomaly_key, 0)) + 1

            counts = _count_detection_groups(detections)
            top_anomaly = ''
            if anomaly_counts:
                top_anomaly = max(anomaly_counts.items(), key=lambda item: item[1])[0]

            return jsonify({
                'success': True,
                'detections': detections,
                'anomalies': anomalies,
                'counts': {
                    'total': len(detections),
                    'person': int(counts.get('person', 0)),
                    'weapon': int(counts.get('weapon', 0)),
                    'object': int(counts.get('object', 0)),
                    'anomaly': len(anomalies),
                },
                'anomaly_counts': anomaly_counts,
                'context': {
                    'enhancement': 'low_light_enhancement_plus_detection',
                    'anomaly_detected': bool(anomalies),
                    'top_anomaly': top_anomaly,
                    'superres_method': upscale_meta.get('method', 'unknown'),
                    'superres_backend': upscale_meta.get('backend', 'unknown'),
                },
                'upscale_meta': upscale_meta,
                'lowlight_analysis': lowlight_analysis,
                'result_image': f"/static/lowlight_uploads/{result_filename}",
                'enhanced_image': f"/static/lowlight_uploads/{enhanced_filename}",
                'original_image': f"/static/lowlight_uploads/{filename}",
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


def _value_from_row(row, key: str, index: int = 0):
    if row is None:
        return None
    if isinstance(row, dict):
        return row.get(key)
    try:
        return row[key]
    except Exception:
        try:
            return row[index]
        except Exception:
            return None


def _table_exists(conn, table_name: str) -> bool:
    try:
        if USE_PG:
            row = conn.execute(
                'SELECT to_regclass(?) AS table_name',
                (f'public.{table_name}',),
            ).fetchone()
            return bool(_value_from_row(row, 'table_name'))

        row = conn.execute(
            'SELECT name FROM sqlite_master WHERE type = ? AND name = ?',
            ('table', table_name),
        ).fetchone()
        return row is not None
    except Exception as exc:
        print(f"[DASHBOARD] Failed table existence check for {table_name}: {exc}")
        return False


def _safe_table_count(conn, table_name: str) -> int:
    if not _table_exists(conn, table_name):
        return 0
    try:
        row = conn.execute(f'SELECT COUNT(*) AS total FROM {table_name}').fetchone()
        value = _value_from_row(row, 'total', 0)
        return int(value or 0)
    except Exception as exc:
        print(f"[DASHBOARD] Failed to count rows in {table_name}: {exc}")
        return 0


def _collect_dashboard_stats() -> dict:
    stats = {
        'cam_count': 0,
        'dataset_count': 0,
        'detections_count': 0,
        'events_count': 0,
    }

    conn = None
    try:
        conn = get_db_connection()
        stats['cam_count'] = _safe_table_count(conn, 'cam')
        stats['dataset_count'] = _safe_table_count(conn, 'datasets')
        stats['detections_count'] = _safe_table_count(conn, 'detection_results')
        stats['events_count'] = _safe_table_count(conn, 'surveillance_events')
    except Exception as exc:
        print(f"[DASHBOARD] Failed to collect stats: {exc}")
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass

    return stats


def _resolve_dashboard_identity() -> dict:
    first_name = str(session.get('firstname', '') or '').strip()
    last_name = str(session.get('lastname', '') or '').strip()
    email = str(session.get('email', '') or '').strip()

    # Backfill missing session names for older sessions created before lastname was stored.
    if email and (not first_name or not last_name):
        conn = None
        try:
            conn = get_db_connection()
            row = conn.execute(
                'SELECT firstname, lastname FROM user WHERE LOWER(email) = ?',
                (email.lower(),),
            ).fetchone()
            if row:
                first_name = str(_value_from_row(row, 'firstname') or first_name).strip()
                last_name = str(_value_from_row(row, 'lastname') or last_name).strip()
                if first_name:
                    session['firstname'] = first_name
                if last_name:
                    session['lastname'] = last_name
        except Exception as exc:
            print(f"[DASHBOARD] Failed to resolve profile name: {exc}")
        finally:
            try:
                if conn is not None:
                    conn.close()
            except Exception:
                pass

    display_name = ' '.join(part for part in (first_name, last_name) if part).strip()
    if not display_name and first_name:
        display_name = first_name
    if not display_name and email:
        display_name = email.split('@', 1)[0]
    if not display_name:
        display_name = 'User'

    return {
        'display_name': display_name,
        'profile_role': 'Admin',
    }

@app.route('/dashboard')
def dashboard():
    if 'loggedin' not in session:
        return redirect(url_for('index'))
    stats = _collect_dashboard_stats()
    identity = _resolve_dashboard_identity()
    return render_template('dashboard.html', **stats, **identity)


@app.route('/api/dashboard_stats')
def dashboard_stats_api():
    if 'loggedin' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    payload = _collect_dashboard_stats()
    payload['server_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
    return jsonify(payload)

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
    host = (os.getenv('APP_HOST') or '127.0.0.1').strip() or '127.0.0.1'
    try:
        port = int(float(os.getenv('APP_PORT', '5000')))
    except Exception:
        port = 5000

    debug_enabled = str(os.getenv('FLASK_DEBUG', '1')).strip().lower() in ('1', 'true', 'yes', 'on')
    auto_open_browser = str(os.getenv('AUTO_OPEN_BROWSER', '1')).strip().lower() in ('1', 'true', 'yes', 'on')

    # Flask debug mode uses a reloader process. Open browser only once in the active child process.
    is_reloader_child = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'
    should_open_browser = auto_open_browser and (not debug_enabled or is_reloader_child)

    if should_open_browser:
        browser_host = '127.0.0.1' if host in ('0.0.0.0', '::') else host
        launch_url = f'http://{browser_host}:{port}/'

        def _launch_browser() -> None:
            try:
                webbrowser.open_new_tab(launch_url)
            except Exception as exc:
                print(f"[APP] Browser auto-open skipped: {exc}")

        threading.Timer(1.0, _launch_browser).start()

    app.run(host=host, port=port, debug=debug_enabled)
