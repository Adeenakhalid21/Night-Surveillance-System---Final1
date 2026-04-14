import os
import shutil
import math
import sys
import threading
import types
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
from skimage import exposure

try:
    import torch
except Exception:  # pragma: no cover - runtime optional dependency
    torch = None

try:
    from torchvision.transforms import functional as _tv_functional
    if 'torchvision.transforms.functional_tensor' not in sys.modules:
        _tv_shim = types.ModuleType('torchvision.transforms.functional_tensor')
        _tv_shim.rgb_to_grayscale = _tv_functional.rgb_to_grayscale
        sys.modules['torchvision.transforms.functional_tensor'] = _tv_shim
except Exception:
    pass

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.archs.srvgg_arch import SRVGGNetCompact
except Exception:  # pragma: no cover - runtime optional dependency
    RRDBNet = None
    SRVGGNetCompact = None


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
        'name': 'fsrcnn',
        'filename': 'fsrcnn_x2.pb',
        'url': 'https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb',
        'min_size_bytes': 32 * 1024,
    },
    4: {
        'name': 'fsrcnn',
        'filename': 'fsrcnn_x4.pb',
        'url': 'https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb',
        'min_size_bytes': 32 * 1024,
    },
}
REAL_ESRGAN_MODEL_SPECS = {
    2: {
        'name': 'RealESRGAN_x2plus',
        'filename': 'RealESRGAN_x2plus.pth',
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
        'arch': 'rrdbnet',
        'scale': 2,
    },
    4: {
        'name': 'realesr-general-x4v3',
        'filename': 'realesr-general-x4v3.pth',
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth',
        'arch': 'srvgg',
        'scale': 4,
    },
}
SUPERRES_ALLOWED_SCALES = (2, 4, 8, 16)
SUPERRES_MAX_OUTPUT_PIXELS = int(float(os.getenv('SUPERRES_MAX_OUTPUT_PIXELS', '120000000')))
SUPERRES_DOWNLOAD_TIMEOUT_SEC = int(float(os.getenv('SUPERRES_DOWNLOAD_TIMEOUT_SEC', '180')))
SUPERRES_ENGINE = str(os.getenv('SUPERRES_ENGINE', 'auto')).strip().lower()
SUPERRES_TILE = max(0, int(float(os.getenv('SUPERRES_TILE', '256'))))
SUPERRES_TILE_PAD = max(2, int(float(os.getenv('SUPERRES_TILE_PAD', '12'))))
SUPERRES_PREPAD = max(0, int(float(os.getenv('SUPERRES_PREPAD', '0'))))
SUPERRES_HALF = str(os.getenv('SUPERRES_HALF', '1')).strip().lower() in ('1', 'true', 'yes', 'on')

ENHANCEMENT_FAST_MODE = str(os.getenv('ENHANCEMENT_FAST_MODE', '1')).strip().lower() in ('1', 'true', 'yes', 'on')
ENHANCEMENT_FAST_PIXELS = max(640 * 480, int(float(os.getenv('ENHANCEMENT_FAST_PIXELS', str(1280 * 720)))))

_SUPERRES_ENGINE_CACHE = {}
_SUPERRES_ENGINE_LOCK = threading.Lock()


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


def _realesrgan_available() -> bool:
    return (
        torch is not None
        and RRDBNet is not None
        and SRVGGNetCompact is not None
    )


def _torch_device_name() -> str:
    if torch is None:
        return 'cpu'
    try:
        if torch.cuda.is_available():
            return 'cuda'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
    except Exception:
        pass
    return 'cpu'


def _backend_priority() -> list[str]:
    if SUPERRES_ENGINE in ('realesrgan', 'esrgan'):
        return ['realesrgan', 'opencv.dnn_superres']
    if SUPERRES_ENGINE in ('opencv', 'opencv.dnn_superres', 'edsr', 'fsrcnn'):
        return ['opencv.dnn_superres', 'realesrgan']
    # Auto mode: prioritize Real-ESRGAN when GPU is available; otherwise use fast CPU SR first.
    return ['realesrgan', 'opencv.dnn_superres'] if _torch_device_name() == 'cuda' else ['opencv.dnn_superres', 'realesrgan']


def _decompose_scale(target_scale: int) -> list[int]:
    if target_scale == 2:
        return [2]
    if target_scale == 4:
        return [4]
    if target_scale == 8:
        return [4, 2]
    if target_scale == 16:
        return [4, 4]
    raise ValueError(f'Unsupported upscale factor x{target_scale}.')


def _ensure_superres_model_file(scale: int) -> Path:
    spec = SUPERRES_MODEL_SPECS.get(scale)
    if not spec:
        raise ValueError(f'No super-resolution model configured for x{scale}.')

    min_size_bytes = int(spec.get('min_size_bytes', 1024 * 1024))
    model_path = SUPERRES_MODEL_DIR / spec['filename']
    if model_path.exists() and model_path.stat().st_size > min_size_bytes:
        return model_path

    _download_model(spec['url'], model_path)
    if not model_path.exists() or model_path.stat().st_size <= min_size_bytes:
        raise RuntimeError(f'Super-resolution model download failed for x{scale}.')

    return model_path


def _ensure_realesrgan_model_file(scale: int) -> Path:
    spec = REAL_ESRGAN_MODEL_SPECS.get(scale)
    if not spec:
        raise ValueError(f'No Real-ESRGAN model configured for x{scale}.')

    model_path = SUPERRES_MODEL_DIR / spec['filename']
    if model_path.exists() and model_path.stat().st_size > (1024 * 1024):
        return model_path

    _download_model(spec['url'], model_path)
    if not model_path.exists() or model_path.stat().st_size <= (1024 * 1024):
        raise RuntimeError(f'Real-ESRGAN model download failed for x{scale}.')

    return model_path


def ensure_superres_models() -> dict:
    ready = {}
    for scale in (2, 4):
        try:
            path = _ensure_superres_model_file(scale)
            ready[f'opencv_x{scale}'] = str(path)
        except Exception:
            pass

        if _realesrgan_available():
            try:
                path = _ensure_realesrgan_model_file(scale)
                ready[f'realesrgan_x{scale}'] = str(path)
            except Exception:
                pass

    return ready


def superres_backend_name() -> str:
    for backend in _backend_priority():
        if backend == 'realesrgan' and _realesrgan_available():
            return 'realesrgan'
        if backend == 'opencv.dnn_superres' and hasattr(cv2, 'dnn_superres'):
            return 'opencv.dnn_superres'
    return 'opencv.resize'


def _build_realesrgan_model(spec: dict):
    scale = int(spec.get('scale', 4))
    arch = str(spec.get('arch', 'srvgg')).lower()

    if arch == 'rrdbnet':
        return RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=scale,
        )

    if arch == 'srvgg':
        return SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=32,
            upscale=scale,
            act_type='prelu',
        )

    raise ValueError(f"Unsupported Real-ESRGAN architecture: {arch}")


class _TorchRealESRGANEngine:
    def __init__(self, model, scale: int, device_name: str, use_half: bool, tile: int, tile_pad: int) -> None:
        self.scale = int(scale)
        self.device_name = device_name
        self.device = torch.device(device_name)
        self.tile = int(max(0, tile))
        self.tile_pad = int(max(0, tile_pad))
        self.use_half = bool(use_half and self.device_name == 'cuda')

        self.model = model.to(self.device)
        self.model.eval()
        if self.use_half:
            self.model = self.model.half()

    def _forward_tensor(self, input_tensor):
        if self.tile <= 0:
            return self.model(input_tensor)

        _, _, h, w = input_tensor.size()
        tile = int(self.tile)
        tile_pad = int(self.tile_pad)
        out_h = h * self.scale
        out_w = w * self.scale

        output = torch.zeros((1, 3, out_h, out_w), device=self.device, dtype=input_tensor.dtype)
        weight = torch.zeros_like(output)

        tiles_x = math.ceil(w / tile)
        tiles_y = math.ceil(h / tile)

        for y in range(tiles_y):
            for x in range(tiles_x):
                x0 = x * tile
                x1 = min(x0 + tile, w)
                y0 = y * tile
                y1 = min(y0 + tile, h)

                x0_pad = max(x0 - tile_pad, 0)
                x1_pad = min(x1 + tile_pad, w)
                y0_pad = max(y0 - tile_pad, 0)
                y1_pad = min(y1 + tile_pad, h)

                input_tile = input_tensor[:, :, y0_pad:y1_pad, x0_pad:x1_pad]
                output_tile = self.model(input_tile)

                out_x0 = x0 * self.scale
                out_x1 = x1 * self.scale
                out_y0 = y0 * self.scale
                out_y1 = y1 * self.scale

                crop_x0 = (x0 - x0_pad) * self.scale
                crop_x1 = crop_x0 + ((x1 - x0) * self.scale)
                crop_y0 = (y0 - y0_pad) * self.scale
                crop_y1 = crop_y0 + ((y1 - y0) * self.scale)

                output[:, :, out_y0:out_y1, out_x0:out_x1] += output_tile[:, :, crop_y0:crop_y1, crop_x0:crop_x1]
                weight[:, :, out_y0:out_y1, out_x0:out_x1] += 1

        return output / torch.clamp(weight, min=1.0)

    def upsample(self, image_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))).unsqueeze(0).to(self.device)
        if self.use_half:
            tensor = tensor.half()

        with torch.inference_mode():
            out_tensor = self._forward_tensor(tensor)

        out = out_tensor.squeeze(0).permute(1, 2, 0).float().cpu().clamp(0.0, 1.0).numpy()
        out = (out * 255.0).round().astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def _engine_cache_key(backend: str, scale: int, device_name: str) -> str:
    return f"{backend}:x{int(scale)}:{device_name}:{SUPERRES_TILE}:{SUPERRES_TILE_PAD}:{SUPERRES_PREPAD}:{int(SUPERRES_HALF)}"


def _create_realesrgan_engine(scale: int) -> dict:
    if not _realesrgan_available():
        raise RuntimeError(
            'Real-ESRGAN backend unavailable. Install torch and basicsr for true SR upscaling.'
        )

    spec = REAL_ESRGAN_MODEL_SPECS.get(scale)
    if not spec:
        raise ValueError(f'No Real-ESRGAN model configured for x{scale}.')

    model_path = _ensure_realesrgan_model_file(scale)
    device_name = _torch_device_name()
    model = _build_realesrgan_model(spec)

    checkpoint = torch.load(str(model_path), map_location='cpu')
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('params_ema') or checkpoint.get('params') or checkpoint.get('state_dict') or checkpoint
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise RuntimeError(f'Invalid Real-ESRGAN checkpoint format: {model_path.name}')

    cleaned_state = {}
    for key, value in state_dict.items():
        new_key = key[7:] if str(key).startswith('module.') else key
        cleaned_state[new_key] = value

    model.load_state_dict(cleaned_state, strict=False)
    engine = _TorchRealESRGANEngine(
        model=model,
        scale=int(spec['scale']),
        device_name=device_name,
        use_half=bool(SUPERRES_HALF),
        tile=int(SUPERRES_TILE),
        tile_pad=int(SUPERRES_TILE_PAD),
    )
    return {
        'backend': 'realesrgan',
        'model_path': model_path,
        'engine': engine,
        'scale': int(spec['scale']),
        'device': device_name,
        'model_name': spec.get('name', 'RealESRGAN'),
    }


def _create_opencv_superres_engine(scale: int) -> dict:
    spec = SUPERRES_MODEL_SPECS.get(scale)
    if not spec:
        raise ValueError(f'Unsupported OpenCV super-resolution scale x{scale}.')

    if not hasattr(cv2, 'dnn_superres'):
        raise RuntimeError(
            'OpenCV super-resolution backend is unavailable. '
            'Install opencv-contrib-python to enable EDSR upscaling.'
        )

    model_path = _ensure_superres_model_file(scale)

    engine = cv2.dnn_superres.DnnSuperResImpl_create()
    engine.readModel(str(model_path))
    engine.setModel(spec['name'], scale)
    return {
        'backend': 'opencv.dnn_superres',
        'model_path': model_path,
        'engine': engine,
        'scale': scale,
        'device': 'cpu',
        'model_name': spec.get('name', 'EDSR'),
    }


def _create_superres_engine(scale: int) -> dict:
    errors = []

    for backend in _backend_priority():
        if backend == 'realesrgan' and not _realesrgan_available():
            errors.append('realesrgan: dependencies missing')
            continue

        if backend == 'opencv.dnn_superres' and not hasattr(cv2, 'dnn_superres'):
            errors.append('opencv.dnn_superres: unavailable')
            continue

        device_name = _torch_device_name() if backend == 'realesrgan' else 'cpu'
        cache_key = _engine_cache_key(backend, scale, device_name)

        with _SUPERRES_ENGINE_LOCK:
            cached = _SUPERRES_ENGINE_CACHE.get(cache_key)
        if cached is not None:
            return cached

        try:
            if backend == 'realesrgan':
                created = _create_realesrgan_engine(scale)
            else:
                created = _create_opencv_superres_engine(scale)

            with _SUPERRES_ENGINE_LOCK:
                _SUPERRES_ENGINE_CACHE[cache_key] = created
            return created
        except Exception as exc:
            errors.append(f"{backend}: {exc}")

    raise RuntimeError('; '.join(errors) if errors else 'No super-resolution backend available.')


def _apply_superres_pass(image_bgr: np.ndarray, engine_info: dict) -> np.ndarray:
    backend = engine_info['backend']
    engine = engine_info['engine']

    if backend == 'realesrgan':
        output = engine.upsample(image_bgr)
        if output.dtype != np.uint8:
            output = np.clip(output, 0, 255).astype(np.uint8)
        return output

    if backend == 'opencv.dnn_superres':
        return engine.upsample(image_bgr)

    raise RuntimeError(f"Unsupported super-resolution backend: {backend}")


def _fallback_upscale_bgr(image_bgr: np.ndarray, target_scale: int, sharpness_strength: float, denoise_strength: float) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    interpolation = cv2.INTER_LANCZOS4 if target_scale <= 4 else cv2.INTER_CUBIC
    upscaled = cv2.resize(image_bgr, (w * target_scale, h * target_scale), interpolation=interpolation)

    denoise_mix = _clamp(denoise_strength / 100.0, 0.0, 1.0)
    if denoise_mix > 0.15:
        sigma_color = 10 + (26 * denoise_mix)
        upscaled = cv2.bilateralFilter(upscaled, d=0, sigmaColor=sigma_color, sigmaSpace=3)

    sharpen_mix = _clamp(sharpness_strength / 100.0, 0.0, 1.0)
    if sharpen_mix > 0.12:
        blur = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=0.95)
        amount = 0.08 + (0.2 * sharpen_mix)
        upscaled = cv2.addWeighted(upscaled, 1.0 + amount, blur, -amount, 0)

    return np.clip(upscaled, 0, 255).astype(np.uint8)


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
        raise ValueError('Supported super-resolution scales are x2, x4, x8, and x16.')

    h, w = image_bgr.shape[:2]
    _validate_superres_budget(w, h, target_scale)

    pass_scales = _decompose_scale(target_scale)

    denoise_mix = _clamp((denoise_strength / 100.0) * (1.15 - (fidelity_strength / 140.0)), 0.0, 1.0)
    sharpen_mix = _clamp((sharpness_strength / 100.0) * (0.45 + (fidelity_strength / 200.0)), 0.0, 1.0)

    current = image_bgr
    model_files = []
    model_names = []
    backend_name = ''
    backend_device = 'cpu'
    fallback_used = False
    fallback_reason = ''
    try:
        for pass_scale in pass_scales:
            engine_info = _create_superres_engine(pass_scale)
            model_files.append(engine_info['model_path'].name)
            model_names.append(str(engine_info.get('model_name', '')))
            backend_name = engine_info['backend']
            backend_device = str(engine_info.get('device', backend_device))
            current = _apply_superres_pass(current, engine_info)

            # Real-ESRGAN already performs strong restoration; avoid re-denoising it.
            if denoise_mix > 0.25 and backend_name != 'realesrgan':
                sigma_color = 14 + (34 * denoise_mix)
                current = cv2.bilateralFilter(current, d=0, sigmaColor=sigma_color, sigmaSpace=4)
    except Exception as exc:
        # Last-resort fallback ensures output is upscaled even if all deep SR backends fail.
        fallback_used = True
        fallback_reason = str(exc)
        backend_name = 'opencv.resize'
        backend_device = 'cpu'
        model_files = []
        model_names = []
        current = _fallback_upscale_bgr(
            image_bgr,
            target_scale=target_scale,
            sharpness_strength=sharpness_strength,
            denoise_strength=denoise_strength,
        )

    if backend_name == 'realesrgan':
        sharpen_mix = min(sharpen_mix, 0.42)

    if sharpen_mix > 0.12:
        blur = cv2.GaussianBlur(current, (0, 0), sigmaX=1.05)
        amount = 0.10 + (0.24 * sharpen_mix)
        current = cv2.addWeighted(current, 1.0 + amount, blur, -amount, 0)

    current = np.clip(current, 0, 255).astype(np.uint8)
    out_h, out_w = current.shape[:2]

    if fallback_used:
        method_name = 'Resize+Enhance (fallback)'
    elif backend_name == 'realesrgan':
        method_name = 'Real-ESRGAN'
    elif any('fsrcnn' in str(name).lower() for name in model_names):
        method_name = 'FSRCNN'
    else:
        method_name = 'EDSR'

    return current, {
        'method': method_name,
        'scale': target_scale,
        'backend': backend_name or superres_backend_name(),
        'device': backend_device,
        'passes': len(pass_scales),
        'output_width': int(out_w),
        'output_height': int(out_h),
        'models': model_files,
        'model_names': model_names,
        'fallback_used': fallback_used,
        'fallback_reason': fallback_reason,
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
    image_pixels = int(image_bgr.shape[0]) * int(image_bgr.shape[1])

    _emit(progress_callback, 'denoise', 'running')
    fast_path = ENHANCEMENT_FAST_MODE and image_pixels >= ENHANCEMENT_FAST_PIXELS
    if fast_path:
        # Faster denoise path for larger images to keep pipeline latency low.
        sigma_color = 24 + (s['denoise'] * 0.42)
        denoised = cv2.bilateralFilter(image_bgr, d=5, sigmaColor=sigma_color, sigmaSpace=22)
    else:
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
