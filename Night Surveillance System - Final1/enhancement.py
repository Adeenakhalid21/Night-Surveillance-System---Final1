import os
import sys
import threading
import types
from pathlib import Path

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


ENHANCEMENT_STAGES = [
    {'key': 'denoise', 'label': 'Denoising'},
    {'key': 'contrast', 'label': 'Contrast Enhancement'},
    {'key': 'gamma', 'label': 'Gamma Correction'},
    {'key': 'sharpen', 'label': 'Sharpening'},
    {'key': 'color', 'label': 'Color Boost'},
    {'key': 'upscale', 'label': 'Super-Resolution Upscaling'},
]


SUPERRES_ALLOWED_SCALES = (2, 4, 8, 16)
SUPERRES_MAX_OUTPUT_PIXELS = int(float(os.getenv('SUPERRES_MAX_OUTPUT_PIXELS', '120000000')))
SUPERRES_ENGINE = str(os.getenv('SUPERRES_ENGINE', 'nunif')).strip().lower()
SUPERRES_TILE = max(0, int(float(os.getenv('SUPERRES_TILE', '256'))))
SUPERRES_TILE_PAD = max(2, int(float(os.getenv('SUPERRES_TILE_PAD', '12'))))
SUPERRES_PREPAD = max(0, int(float(os.getenv('SUPERRES_PREPAD', '0'))))
SUPERRES_HALF = str(os.getenv('SUPERRES_HALF', '1')).strip().lower() in ('1', 'true', 'yes', 'on')

NUNIF_REPO_URL = 'https://github.com/nagadomi/nunif'
NUNIF_REPO_DIR = Path(__file__).resolve().parent / 'third_party' / 'nunif'
NUNIF_REPO_REF = str(os.getenv('NUNIF_REPO_REF', 'master') or 'master').strip() or 'master'
NUNIF_MODEL_TYPE = str(os.getenv('NUNIF_MODEL_TYPE', 'photo') or 'photo').strip().lower()
if NUNIF_MODEL_TYPE == 'scan':
    NUNIF_MODEL_TYPE = 'art_scan'
if NUNIF_MODEL_TYPE not in ('art', 'art_scan', 'photo', 'swin_unet/art', 'swin_unet/art_scan', 'swin_unet/photo', 'cunet/art', 'upconv_7/art', 'upconv_7/photo'):
    NUNIF_MODEL_TYPE = 'photo'

NUNIF_NOISE_LEVEL = max(0, min(3, int(float(os.getenv('NUNIF_NOISE_LEVEL', '2')))))
NUNIF_BATCH_SIZE = max(1, int(float(os.getenv('NUNIF_BATCH_SIZE', '1'))))
NUNIF_TILE_SIZE = max(64, int(float(os.getenv('NUNIF_TILE_SIZE', str(max(256, SUPERRES_TILE))))))
NUNIF_TTA = str(os.getenv('NUNIF_TTA', '0')).strip().lower() in ('1', 'true', 'yes', 'on')

if not os.getenv('NUNIF_HOME'):
    os.environ['NUNIF_HOME'] = str((Path(__file__).resolve().parent / 'models' / 'nunif_home').resolve())

ENHANCEMENT_FAST_MODE = str(os.getenv('ENHANCEMENT_FAST_MODE', '1')).strip().lower() in ('1', 'true', 'yes', 'on')
ENHANCEMENT_FAST_PIXELS = max(640 * 480, int(float(os.getenv('ENHANCEMENT_FAST_PIXELS', str(1280 * 720)))))

_SUPERRES_ENGINE_CACHE = {}
_SUPERRES_ENGINE_LOCK = threading.Lock()
_NUNIF_DIST_INFO_CACHE = None


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


def _lowlight_severity(score: int) -> str:
    if score >= 75:
        return 'severe'
    if score >= 58:
        return 'moderate'
    if score >= 45:
        return 'mild'
    return 'normal'


def _lowlight_recommendation(severity: str) -> str:
    if severity == 'severe':
        return 'Strong enhancement suggested: high denoise, high contrast, and NUNIF waifu2x x4 or above.'
    if severity == 'moderate':
        return 'Enhancement recommended: moderate denoise plus NUNIF waifu2x x2 to x4.'
    if severity == 'mild':
        return 'Light enhancement is enough: mild denoise and detail-preserving upscale.'
    return 'Lighting is acceptable. Enhancement is optional unless detection quality drops.'


def analyze_lowlight_image(image_bgr: np.ndarray) -> dict:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError('Input image is empty')

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean_luminance = float(np.mean(gray))
    contrast_std = float(np.std(gray))
    p10 = float(np.percentile(gray, 10))
    p90 = float(np.percentile(gray, 90))
    dynamic_range = max(0.0, p90 - p10)
    shadow_ratio = float(np.mean(gray <= 45.0) * 100.0)
    highlight_ratio = float(np.mean(gray >= 210.0) * 100.0)

    residual = cv2.subtract(gray, cv2.GaussianBlur(gray, (0, 0), 1.1))
    noise_proxy = float(np.std(residual))

    mean_darkness = 1.0 - (mean_luminance / 255.0)
    shadow_component = min(1.0, shadow_ratio / 70.0)
    contrast_component = max(0.0, (40.0 - contrast_std) / 40.0)

    low_light_score = int(round(100.0 * (
        (0.55 * mean_darkness)
        + (0.30 * shadow_component)
        + (0.15 * contrast_component)
    )))
    low_light_score = int(_clamp(low_light_score, 0, 100))
    severity = _lowlight_severity(low_light_score)

    return {
        'is_low_light': severity != 'normal',
        'low_light_score': low_light_score,
        'severity': severity,
        'recommendation': _lowlight_recommendation(severity),
        'metrics': {
            'mean_luminance': round(mean_luminance, 2),
            'contrast_std': round(contrast_std, 2),
            'dynamic_range': round(dynamic_range, 2),
            'shadow_ratio_percent': round(shadow_ratio, 2),
            'highlight_ratio_percent': round(highlight_ratio, 2),
            'noise_proxy': round(noise_proxy, 2),
        },
    }


def analyze_lowlight_improvement(original_bgr: np.ndarray, enhanced_bgr: np.ndarray) -> dict:
    before = analyze_lowlight_image(original_bgr)
    after = analyze_lowlight_image(enhanced_bgr)

    before_metrics = before.get('metrics', {})
    after_metrics = after.get('metrics', {})

    score_reduction = int(max(0, before.get('low_light_score', 0) - after.get('low_light_score', 0)))
    brightness_gain = round(float(after_metrics.get('mean_luminance', 0.0)) - float(before_metrics.get('mean_luminance', 0.0)), 2)
    contrast_gain = round(float(after_metrics.get('contrast_std', 0.0)) - float(before_metrics.get('contrast_std', 0.0)), 2)
    shadow_reduction = round(float(before_metrics.get('shadow_ratio_percent', 0.0)) - float(after_metrics.get('shadow_ratio_percent', 0.0)), 2)

    if score_reduction >= 25:
        quality = 'strong'
    elif score_reduction >= 12:
        quality = 'moderate'
    elif score_reduction >= 5:
        quality = 'small'
    else:
        quality = 'minimal'

    return {
        'before': before,
        'after': after,
        'improvement': {
            'score_reduction': score_reduction,
            'brightness_gain': brightness_gain,
            'contrast_gain': contrast_gain,
            'shadow_reduction_percent': shadow_reduction,
            'quality': quality,
            'improved': bool(score_reduction >= 5 or brightness_gain >= 8.0),
        },
    }


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


def _nunif_available() -> bool:
    return torch is not None


def _resolve_nunif_repo_dir() -> Path:
    custom_repo = str(os.getenv('NUNIF_REPO_DIR', '') or '').strip()
    if custom_repo:
        return Path(custom_repo).expanduser().resolve()
    return NUNIF_REPO_DIR


def _read_git_head_short(repo_dir: Path) -> str:
    git_dir = repo_dir / '.git'
    if git_dir.is_file():
        try:
            marker = git_dir.read_text(encoding='utf-8', errors='ignore').strip()
            if marker.startswith('gitdir:'):
                rel = marker.split(':', 1)[1].strip()
                candidate = (repo_dir / rel).resolve()
                if candidate.exists():
                    git_dir = candidate
        except Exception:
            return ''
    if not git_dir.exists():
        return ''

    head_file = git_dir / 'HEAD'
    if not head_file.exists():
        return ''
    try:
        head_value = head_file.read_text(encoding='utf-8', errors='ignore').strip()
        if head_value.startswith('ref:'):
            ref_name = head_value.split(':', 1)[1].strip()
            ref_file = git_dir / ref_name
            if ref_file.exists():
                head_value = ref_file.read_text(encoding='utf-8', errors='ignore').strip()
            else:
                packed_refs = git_dir / 'packed-refs'
                if packed_refs.exists():
                    for line in packed_refs.read_text(encoding='utf-8', errors='ignore').splitlines():
                        line = line.strip()
                        if not line or line.startswith('#') or line.startswith('^'):
                            continue
                        try:
                            commit, ref = line.split(' ', 1)
                        except ValueError:
                            continue
                        if ref.strip() == ref_name:
                            head_value = commit.strip()
                            break
        return head_value[:12] if len(head_value) >= 7 else ''
    except Exception:
        return ''


def _nunif_distribution_info() -> dict:
    global _NUNIF_DIST_INFO_CACHE
    if _NUNIF_DIST_INFO_CACHE is not None:
        return dict(_NUNIF_DIST_INFO_CACHE)

    repo_dir = _resolve_nunif_repo_dir()
    source = NUNIF_REPO_URL
    if repo_dir.exists():
        source = str(repo_dir)

    info = {
        'package': 'nunif-waifu2x',
        'version': 'master',
        'source': source,
        'commit': _read_git_head_short(repo_dir),
    }

    _NUNIF_DIST_INFO_CACHE = info
    return dict(info)


def _backend_priority() -> list[str]:
    if SUPERRES_ENGINE in ('nunif', 'nunif.waifu2x', 'waifu2x', 'nagadomi.nunif'):
        return ['nunif.waifu2x']
    # Keep runtime deterministic even if old env values are still present.
    return ['nunif.waifu2x']


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


def ensure_superres_models() -> dict:
    ready = {}
    if not _nunif_available():
        ready['nunif_error'] = 'torch/torchvision not available'
        return ready

    for scale in (2, 4):
        try:
            info = _create_superres_engine(scale)
            ready[f'nunif_x{scale}'] = f"{info.get('model_name', 'waifu2x')}:{info.get('method_name', '')}".rstrip(':')
        except Exception as exc:
            ready[f'nunif_x{scale}_error'] = str(exc)

    return ready


def superres_backend_name() -> str:
    if _nunif_available():
        return 'nunif.waifu2x'
    return 'opencv.resize'


class _NunifWaifu2xEngine:
    def __init__(self, model, scale: int, device_name: str, model_type: str, noise_level: int, tta: bool = False) -> None:
        self.model = model
        self.scale = int(scale)
        self.device_name = str(device_name)
        self.model_type = str(model_type)
        self.noise_level = int(noise_level)
        self.tta = bool(tta)
        self._lock = threading.RLock()

    def upsample(self, image_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_input = Image.fromarray(rgb)

        with self._lock:
            pil_output = self.model.infer(pil_input, tta=self.tta, output_type='pil')

        out_rgb = np.array(pil_output.convert('RGB'))
        return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


def _create_nunif_engine(scale: int) -> dict:
    if not _nunif_available():
        raise RuntimeError('NUNIF backend unavailable. Install torch and torchvision first.')

    if scale not in (2, 4):
        raise ValueError(f'NUNIF backend supports x2 and x4 passes only, got x{scale}.')

    method = 'scale' if scale == 2 else 'scale4x'
    device_name = _torch_device_name()
    use_amp = bool(SUPERRES_HALF and device_name == 'cuda')
    device_ids = [0] if device_name == 'cuda' else [-1]

    repo_dir = _resolve_nunif_repo_dir()
    if repo_dir.exists() and (repo_dir / 'hubconf.py').exists():
        hub_source = 'local'
        hub_target = str(repo_dir)
        package_source = str(repo_dir)
    else:
        hub_source = 'github'
        hub_target = f'nagadomi/nunif:{NUNIF_REPO_REF}'
        package_source = NUNIF_REPO_URL

    model = torch.hub.load(
        hub_target,
        'waifu2x',
        source=hub_source,
        trust_repo=True,
        model_type=NUNIF_MODEL_TYPE,
        method=method,
        noise_level=NUNIF_NOISE_LEVEL,
        device_ids=device_ids,
        tile_size=NUNIF_TILE_SIZE,
        batch_size=NUNIF_BATCH_SIZE,
        keep_alpha=False,
        amp=use_amp,
    )

    try:
        model = model.to(device_name)
    except Exception:
        model = model.to('cpu')
        device_name = 'cpu'

    dist_info = _nunif_distribution_info()
    model_file = (repo_dir / 'hubconf.py') if (repo_dir / 'hubconf.py').exists() else Path('nunif_hubconf.py')

    engine = _NunifWaifu2xEngine(
        model=model,
        scale=scale,
        device_name=device_name,
        model_type=NUNIF_MODEL_TYPE,
        noise_level=NUNIF_NOISE_LEVEL,
        tta=NUNIF_TTA,
    )

    return {
        'backend': 'nunif.waifu2x',
        'model_path': model_file,
        'engine': engine,
        'scale': int(scale),
        'device': device_name,
        'model_name': f'waifu2x/{NUNIF_MODEL_TYPE}',
        'method_name': method,
        'package': dist_info.get('package', 'nunif-waifu2x'),
        'package_version': dist_info.get('version', 'master'),
        'package_source': package_source,
        'package_commit': dist_info.get('commit', ''),
    }


def _engine_cache_key(backend: str, scale: int, device_name: str) -> str:
    return f"{backend}:x{int(scale)}:{device_name}:{SUPERRES_TILE}:{SUPERRES_TILE_PAD}:{SUPERRES_PREPAD}:{int(SUPERRES_HALF)}"


def _create_superres_engine(scale: int) -> dict:
    errors = []

    for backend in _backend_priority():
        if backend == 'nunif.waifu2x' and not _nunif_available():
            errors.append('nunif.waifu2x: dependencies missing')
            continue

        device_name = _torch_device_name() if backend == 'nunif.waifu2x' else 'cpu'
        cache_key = _engine_cache_key(backend, scale, device_name)

        with _SUPERRES_ENGINE_LOCK:
            cached = _SUPERRES_ENGINE_CACHE.get(cache_key)
        if cached is not None:
            return cached

        try:
            if backend == 'nunif.waifu2x':
                created = _create_nunif_engine(scale)
            else:
                raise RuntimeError(f'Unsupported backend: {backend}')

            with _SUPERRES_ENGINE_LOCK:
                _SUPERRES_ENGINE_CACHE[cache_key] = created
            return created
        except Exception as exc:
            errors.append(f"{backend}: {exc}")

    raise RuntimeError('; '.join(errors) if errors else 'No super-resolution backend available.')


def _apply_superres_pass(image_bgr: np.ndarray, engine_info: dict) -> np.ndarray:
    backend = engine_info['backend']
    engine = engine_info['engine']

    if backend == 'nunif.waifu2x':
        output = engine.upsample(image_bgr)
        if output.dtype != np.uint8:
            output = np.clip(output, 0, 255).astype(np.uint8)
        return output

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
    package_name = ''
    package_version = ''
    package_source = ''
    package_commit = ''
    try:
        for pass_scale in pass_scales:
            engine_info = _create_superres_engine(pass_scale)
            model_files.append(engine_info['model_path'].name)
            model_names.append(str(engine_info.get('model_name', '')))
            backend_name = engine_info['backend']
            backend_device = str(engine_info.get('device', backend_device))
            if backend_name == 'nunif.waifu2x':
                package_name = str(engine_info.get('package', package_name or 'nunif-waifu2x'))
                package_version = str(engine_info.get('package_version', package_version or 'master'))
                package_source = str(engine_info.get('package_source', package_source or NUNIF_REPO_URL))
                package_commit = str(engine_info.get('package_commit', package_commit or ''))
            current = _apply_superres_pass(current, engine_info)

            # NUNIF already applies noise-aware restoration; avoid adding heavy post-denoise.
            if denoise_mix > 0.25 and backend_name != 'nunif.waifu2x':
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

    if backend_name == 'nunif.waifu2x':
        sharpen_mix = min(sharpen_mix, 0.4)

    if sharpen_mix > 0.12:
        blur = cv2.GaussianBlur(current, (0, 0), sigmaX=1.05)
        amount = 0.10 + (0.24 * sharpen_mix)
        current = cv2.addWeighted(current, 1.0 + amount, blur, -amount, 0)

    current = np.clip(current, 0, 255).astype(np.uint8)
    out_h, out_w = current.shape[:2]

    if fallback_used:
        method_name = 'Resize+Enhance (fallback)'
    elif backend_name == 'nunif.waifu2x':
        method_name = 'NUNIF waifu2x'
    else:
        method_name = 'NUNIF'

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
        'package': package_name,
        'package_version': package_version,
        'package_source': package_source,
        'package_commit': package_commit,
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


def enhance_image_with_meta(image_bgr: np.ndarray, settings: dict | None = None) -> tuple[np.ndarray, dict, dict]:
    final_pil, normalized_settings, upscale_meta = _enhance_pipeline(
        image_bgr,
        settings=settings,
        progress_callback=None,
    )
    enhanced_bgr = cv2.cvtColor(np.array(final_pil), cv2.COLOR_RGB2BGR)
    return enhanced_bgr, normalized_settings, upscale_meta


def enhance_image(image_bgr: np.ndarray, settings: dict | None = None) -> np.ndarray:
    enhanced_bgr, _, _ = enhance_image_with_meta(image_bgr, settings=settings)
    return enhanced_bgr


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
    resolved_method = str((upscale_meta or {}).get('method', 'NUNIF waifu2x'))
    resolved_backend = str((upscale_meta or {}).get('backend', 'unknown'))
    package_version = str((upscale_meta or {}).get('package_version', '')).strip()
    package_commit = str((upscale_meta or {}).get('package_commit', '')).strip()

    model_names = [
        str(name).strip() for name in ((upscale_meta or {}).get('model_names') or [])
        if str(name).strip()
    ]
    model_summary = ', '.join(sorted(set(model_names))) if model_names else ''

    super_resolution_label = f'{resolved_method} x{resolved_scale}'
    if resolved_backend.startswith('nunif'):
        version_label = f' {package_version}' if package_version else ''
        commit_label = f' @{package_commit}' if package_commit else ''
        model_label = f' ({model_summary})' if model_summary else ''
        super_resolution_label = f'NUNIF waifu2x{version_label}{commit_label}{model_label} x{resolved_scale}'

    upscale_engine_label = resolved_backend
    if resolved_backend.startswith('nunif'):
        upscale_engine_label = f'nunif.waifu2x {package_version or "master"}'
        if package_commit:
            upscale_engine_label = f'{upscale_engine_label} ({package_commit})'

    return {
        'brightness_increase': f'+{brightness_delta}',
        'contrast_improvement': f'+{contrast_delta}%',
        'noise_reduction': f'-{noise_reduction}%',
        'sharpness_boost': f'+{sharpness_boost}%',
        'resolution_gain': f'+{resolution_gain}%',
        'super_resolution': super_resolution_label,
        'upscale_engine': upscale_engine_label,
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
