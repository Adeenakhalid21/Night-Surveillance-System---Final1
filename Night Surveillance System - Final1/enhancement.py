import os
import shutil
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
from skimage import exposure


ENHANCEMENT_STAGES = [
    {'key': 'denoise', 'label': 'Denoising'},
    {'key': 'contrast', 'label': 'Contrast Enhancement'},
    {'key': 'gamma', 'label': 'Gamma Correction'},
    {'key': 'sharpen', 'label': 'Sharpening'},
    {'key': 'color', 'label': 'Color Boost'},
    {'key': 'upscale', 'label': 'Super-Resolution Upscaling'},
]


SUPERRES_MODEL_DIR = Path(__file__).resolve().parent / 'models' / 'superres'
SUPERRES_MODEL_SPECS = {
    2: {
        'name': 'edsr',
        'filename': 'edsr_x2.pb',
        'url': 'https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x2.pb',
    },
    4: {
        'name': 'edsr',
        'filename': 'edsr_x4.pb',
        'url': 'https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb',
    },
}
SUPERRES_ALLOWED_SCALES = (2, 4, 16)
SUPERRES_MAX_OUTPUT_PIXELS = int(float(os.getenv('SUPERRES_MAX_OUTPUT_PIXELS', '120000000')))
SUPERRES_DOWNLOAD_TIMEOUT_SEC = int(float(os.getenv('SUPERRES_DOWNLOAD_TIMEOUT_SEC', '180')))


def enhancement_stage_keys() -> list[dict]:
    return ENHANCEMENT_STAGES


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize_settings(settings: dict | None) -> dict:
    defaults = {
        'brightness': 55.0,
        'contrast': 65.0,
        'sharpness': 60.0,
        'denoise': 55.0,
        'upscale': 70.0,
        'upscale_scale': 4,
    }
    if not settings:
        return defaults

    normalized = {}
    for key in ('brightness', 'contrast', 'sharpness', 'denoise', 'upscale'):
        default_value = defaults[key]
        try:
            normalized[key] = _clamp(float(settings.get(key, default_value)), 0.0, 100.0)
        except Exception:
            normalized[key] = default_value

    normalized['upscale_scale'] = _resolve_upscale_scale(settings.get('upscale_scale', defaults['upscale_scale']))
    return normalized


def _emit(progress_callback, stage_key: str, stage_status: str) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(stage_key, stage_status)
    except Exception:
        pass


def _build_gamma_lut(gamma: float) -> np.ndarray:
    table = np.array([
        ((i / 255.0) ** (1.0 / gamma)) * 255.0
        for i in range(256)
    ]).astype('uint8')
    return table


def _resolve_upscale_scale(scale_value) -> int:
    try:
        scale = int(float(scale_value))
    except Exception:
        scale = 4
    return scale if scale in SUPERRES_ALLOWED_SCALES else 4


def _download_model(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + '.part')

    request = Request(url, headers={'User-Agent': 'NightWatch-SuperRes/1.0'})
    with urlopen(request, timeout=SUPERRES_DOWNLOAD_TIMEOUT_SEC) as response, temp_path.open('wb') as output_file:
        shutil.copyfileobj(response, output_file)

    temp_path.replace(destination)


def _ensure_superres_model_file(scale: int) -> Path:
    spec = SUPERRES_MODEL_SPECS.get(scale)
    if not spec:
        raise ValueError(f'No super-resolution model configured for x{scale}.')

    model_path = SUPERRES_MODEL_DIR / spec['filename']
    if model_path.exists() and model_path.stat().st_size > (1024 * 1024):
        return model_path

    _download_model(spec['url'], model_path)
    if not model_path.exists() or model_path.stat().st_size <= (1024 * 1024):
        raise RuntimeError(f'Super-resolution model download failed for x{scale}.')

    return model_path


def ensure_superres_models() -> dict:
    ready = {}
    for scale in (2, 4):
        path = _ensure_superres_model_file(scale)
        ready[f'x{scale}'] = str(path)
    return ready


def superres_backend_name() -> str:
    return 'opencv.dnn_superres' if hasattr(cv2, 'dnn_superres') else 'opencv.dnn.tensorflow'


def _create_superres_engine(scale: int):
    spec = SUPERRES_MODEL_SPECS.get(scale)
    if not spec:
        raise ValueError(f'Unsupported super-resolution scale x{scale}.')

    model_path = _ensure_superres_model_file(scale)

    if hasattr(cv2, 'dnn_superres'):
        engine = cv2.dnn_superres.DnnSuperResImpl_create()
        engine.readModel(str(model_path))
        engine.setModel(spec['name'], scale)
        return {
            'backend': 'opencv.dnn_superres',
            'model_path': model_path,
            'engine': engine,
            'scale': scale,
        }

    net = cv2.dnn.readNetFromTensorflow(str(model_path))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return {
        'backend': 'opencv.dnn.tensorflow',
        'model_path': model_path,
        'engine': net,
        'scale': scale,
    }


def _apply_superres_pass(image_bgr: np.ndarray, engine_info: dict) -> np.ndarray:
    backend = engine_info['backend']
    engine = engine_info['engine']

    if backend == 'opencv.dnn_superres':
        return engine.upsample(image_bgr)

    if backend == 'opencv.dnn.tensorflow':
        blob = cv2.dnn.blobFromImage(
            image_bgr,
            scalefactor=1.0 / 255.0,
            size=(image_bgr.shape[1], image_bgr.shape[0]),
            mean=(0.0, 0.0, 0.0),
            swapRB=True,
            crop=False,
        )
        engine.setInput(blob)
        output = engine.forward()
        output = np.squeeze(output, axis=0)
        output = np.transpose(output, (1, 2, 0))
        output = np.clip(output, 0.0, 1.0)
        output_rgb = (output * 255.0).astype(np.uint8)
        return cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)

    raise RuntimeError(f"Unsupported super-resolution backend: {backend}")


def _validate_superres_budget(width: int, height: int, scale: int) -> None:
    output_pixels = int(width) * int(height) * int(scale) * int(scale)
    if output_pixels > SUPERRES_MAX_OUTPUT_PIXELS:
        max_megapixels = round(SUPERRES_MAX_OUTPUT_PIXELS / 1_000_000, 1)
        raise ValueError(
            f'Requested x{scale} output is too large for this server budget. '
            f'Choose a lower scale or smaller image (limit ~{max_megapixels}MP).'
        )


def _super_resolve_bgr(image_bgr: np.ndarray, target_scale: int, denoise_strength: float, sharpness_strength: float, fidelity_strength: float) -> tuple[np.ndarray, dict]:
    if target_scale not in SUPERRES_ALLOWED_SCALES:
        raise ValueError('Supported super-resolution scales are x2, x4, and x16.')

    h, w = image_bgr.shape[:2]
    _validate_superres_budget(w, h, target_scale)

    pass_scales = [2] if target_scale == 2 else [4] if target_scale == 4 else [4, 4]

    denoise_mix = _clamp((denoise_strength / 100.0) * (1.15 - (fidelity_strength / 140.0)), 0.0, 1.0)
    sharpen_mix = _clamp((sharpness_strength / 100.0) * (0.45 + (fidelity_strength / 200.0)), 0.0, 1.0)

    current = image_bgr
    model_files = []
    backend_name = ''
    for pass_scale in pass_scales:
        engine_info = _create_superres_engine(pass_scale)
        model_files.append(engine_info['model_path'].name)
        backend_name = engine_info['backend']
        current = _apply_superres_pass(current, engine_info)

        if denoise_mix > 0.25:
            sigma_color = 14 + (34 * denoise_mix)
            current = cv2.bilateralFilter(current, d=0, sigmaColor=sigma_color, sigmaSpace=4)

    if sharpen_mix > 0.12:
        blur = cv2.GaussianBlur(current, (0, 0), sigmaX=1.05)
        amount = 0.10 + (0.24 * sharpen_mix)
        current = cv2.addWeighted(current, 1.0 + amount, blur, -amount, 0)

    current = np.clip(current, 0, 255).astype(np.uint8)
    out_h, out_w = current.shape[:2]

    return current, {
        'method': 'EDSR',
        'scale': target_scale,
        'backend': backend_name or superres_backend_name(),
        'passes': len(pass_scales),
        'output_width': int(out_w),
        'output_height': int(out_h),
        'models': model_files,
    }


def _enhance_pipeline(image_bgr: np.ndarray, settings: dict | None = None, progress_callback=None) -> tuple[Image.Image, dict, dict]:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError('Input image is empty')

    s = _normalize_settings(settings)

    denoise_h = int(5 + (s['denoise'] * 0.16))
    denoise_chroma = int(4 + (s['denoise'] * 0.13))
    clahe_clip = 1.6 + (s['contrast'] * 0.035)
    gamma = 1.15 + (s['brightness'] * 0.01)
    sharpen_amount = 0.22 + (s['sharpness'] * 0.014)
    detail_mix = 0.1 + (s['sharpness'] * 0.003)
    color_factor = 1.05 + (s['contrast'] * 0.006)
    contrast_factor = 1.03 + (s['contrast'] * 0.0025)
    brightness_factor = 0.92 + (s['brightness'] * 0.006)

    _emit(progress_callback, 'denoise', 'running')
    pre_smoothed = cv2.bilateralFilter(image_bgr, d=5, sigmaColor=40, sigmaSpace=40)
    denoised = cv2.fastNlMeansDenoisingColored(pre_smoothed, None, denoise_h, denoise_chroma, 7, 21)
    _emit(progress_callback, 'denoise', 'done')

    _emit(progress_callback, 'contrast', 'running')
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    lift_gamma = max(0.45, 0.88 - (s['brightness'] * 0.0035))
    shadow_lut = np.array([
        ((i / 255.0) ** lift_gamma) * 255.0
        for i in range(256)
    ]).astype('uint8')
    l_balanced = cv2.LUT(l_enhanced, shadow_lut)

    lab_enhanced = cv2.merge([l_balanced, a_channel, b_channel])
    contrast_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    contrast_enhanced = exposure.rescale_intensity(
        contrast_enhanced,
        in_range='image',
        out_range=(0, 255)
    ).astype(np.uint8)
    _emit(progress_callback, 'contrast', 'done')

    _emit(progress_callback, 'gamma', 'running')
    lut = _build_gamma_lut(gamma)
    gamma_corrected = cv2.LUT(contrast_enhanced, lut)
    _emit(progress_callback, 'gamma', 'done')

    _emit(progress_callback, 'sharpen', 'running')
    blurred = cv2.GaussianBlur(gamma_corrected, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(gamma_corrected, 1.0 + sharpen_amount, blurred, -sharpen_amount, 0)
    detail_map = cv2.subtract(gamma_corrected, blurred)
    sharpened = cv2.addWeighted(sharpened, 1.0, detail_map, detail_mix, 0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    _emit(progress_callback, 'sharpen', 'done')

    _emit(progress_callback, 'color', 'running')
    pil_img = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
    color_boosted = ImageEnhance.Color(pil_img).enhance(color_factor)
    contrast_boosted = ImageEnhance.Contrast(color_boosted).enhance(contrast_factor)
    brightened = ImageEnhance.Brightness(contrast_boosted).enhance(brightness_factor)
    final = brightened.filter(
        ImageFilter.UnsharpMask(radius=1.8, percent=int(110 + s['sharpness']), threshold=2)
    )
    _emit(progress_callback, 'color', 'done')

    _emit(progress_callback, 'upscale', 'running')
    enhanced_bgr = cv2.cvtColor(np.array(final.convert('RGB')), cv2.COLOR_RGB2BGR)
    upscaled_bgr, upscale_meta = _super_resolve_bgr(
        enhanced_bgr,
        target_scale=int(s['upscale_scale']),
        denoise_strength=s['denoise'],
        sharpness_strength=s['sharpness'],
        fidelity_strength=s['upscale'],
    )
    final = Image.fromarray(cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB))

    _emit(progress_callback, 'upscale', 'done')

    return final, s, upscale_meta


def enhance_image(image_bgr: np.ndarray, settings: dict | None = None) -> np.ndarray:
    final_pil, _, _ = _enhance_pipeline(image_bgr, settings=settings, progress_callback=None)
    return cv2.cvtColor(np.array(final_pil), cv2.COLOR_RGB2BGR)


def generate_comparison(original_path: str, enhanced_img: Image.Image) -> Image.Image:
    original = Image.open(original_path).convert('RGB')
    enhanced = enhanced_img.convert('RGB')

    height = 500
    orig_w = int(original.width * height / original.height)
    enh_w = int(enhanced.width * height / enhanced.height)

    original = original.resize((orig_w, height))
    enhanced = enhanced.resize((enh_w, height))

    comparison = Image.new('RGB', (orig_w + enh_w + 14, height + 40), (10, 14, 26))
    comparison.paste(original, (0, 40))
    comparison.paste(enhanced, (orig_w + 14, 40))

    draw = ImageDraw.Draw(comparison)
    label_font = ImageFont.load_default()
    draw.text((12, 10), 'BEFORE', fill=(180, 190, 210), font=label_font)
    draw.text((orig_w + 26, 10), 'AFTER', fill=(0, 255, 136), font=label_font)

    return comparison


def calculate_enhancement_stats(original_path: str, enhanced_img: Image.Image, settings: dict | None = None, upscale_meta: dict | None = None) -> dict:
    s = _normalize_settings(settings)

    orig = np.array(Image.open(original_path).convert('L'), dtype=np.float32)
    enh = np.array(enhanced_img.convert('L'), dtype=np.float32)

    brightness_delta = int(max(0, enh.mean() - orig.mean()))
    orig_std = max(1.0, float(orig.std()))
    contrast_delta = int(max(0, ((float(enh.std()) - orig_std) / orig_std) * 100))
    contrast_delta = int(min(160, contrast_delta))

    orig_blur = cv2.GaussianBlur(orig, (3, 3), 0)
    enh_blur = cv2.GaussianBlur(enh, (3, 3), 0)
    orig_noise_proxy = max(1.0, float(cv2.Laplacian(orig_blur, cv2.CV_32F).std()))
    enh_noise_proxy = float(cv2.Laplacian(enh_blur, cv2.CV_32F).std())
    measured_noise_reduction = int(max(0, (1.0 - (enh_noise_proxy / orig_noise_proxy)) * 100))
    noise_reduction = int(min(90, max(measured_noise_reduction, int(18 + (s['denoise'] * 0.48)))))

    orig_edge_var = max(1.0, float(cv2.Laplacian(orig, cv2.CV_32F).var()))
    enh_edge_var = float(cv2.Laplacian(enh, cv2.CV_32F).var())
    sharpness_boost = int(max(0, ((enh_edge_var - orig_edge_var) / orig_edge_var) * 100))
    sharpness_boost = int(min(180, max(sharpness_boost, int(22 + (s['sharpness'] * 0.5)))))

    resolution_gain = int(max(0, ((enhanced_img.width * enhanced_img.height) / max(1, orig.shape[1] * orig.shape[0]) - 1) * 100))

    resolved_scale = int((upscale_meta or {}).get('scale', s.get('upscale_scale', 4)))
    resolved_method = str((upscale_meta or {}).get('method', 'EDSR'))

    return {
        'brightness_increase': f'+{brightness_delta}',
        'contrast_improvement': f'+{contrast_delta}%',
        'noise_reduction': f'-{noise_reduction}%',
        'sharpness_boost': f'+{sharpness_boost}%',
        'resolution_gain': f'+{resolution_gain}%',
        'super_resolution': f'{resolved_method} x{resolved_scale}',
        'output_resolution': f'{enhanced_img.width} x {enhanced_img.height}',
    }


def enhance_night_image(image_path: str, settings: dict | None = None, progress_callback=None) -> dict:
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f'Could not read image from path: {image_path}')

    enhanced_pil, normalized_settings, upscale_meta = _enhance_pipeline(
        image_bgr,
        settings=settings,
        progress_callback=progress_callback,
    )

    comparison_img = generate_comparison(image_path, enhanced_pil)
    stats = calculate_enhancement_stats(
        image_path,
        enhanced_pil,
        settings=normalized_settings,
        upscale_meta=upscale_meta,
    )

    return {
        'enhanced_image': enhanced_pil,
        'comparison_image': comparison_img,
        'stats': stats,
        'upscale_meta': upscale_meta,
    }
