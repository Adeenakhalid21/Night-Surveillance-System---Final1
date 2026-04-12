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
]


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
    }
    if not settings:
        return defaults

    normalized = {}
    for key, default_value in defaults.items():
        try:
            normalized[key] = _clamp(float(settings.get(key, default_value)), 0.0, 100.0)
        except Exception:
            normalized[key] = default_value
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


def _enhance_pipeline(image_bgr: np.ndarray, settings: dict | None = None, progress_callback=None) -> tuple[Image.Image, dict]:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError('Input image is empty')

    s = _normalize_settings(settings)

    denoise_h = int(6 + (s['denoise'] * 0.14))
    clahe_clip = 1.8 + (s['contrast'] * 0.03)
    gamma = 1.2 + (s['brightness'] * 0.012)
    sharpen_amount = 0.25 + (s['sharpness'] * 0.012)
    color_factor = 1.1 + (s['contrast'] * 0.005)
    brightness_factor = 0.9 + (s['brightness'] * 0.005)

    _emit(progress_callback, 'denoise', 'running')
    denoised = cv2.fastNlMeansDenoisingColored(image_bgr, None, denoise_h, denoise_h, 7, 21)
    _emit(progress_callback, 'denoise', 'done')

    _emit(progress_callback, 'contrast', 'running')
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
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
    blurred = cv2.GaussianBlur(gamma_corrected, (0, 0), sigmaX=1.1)
    sharpened = cv2.addWeighted(gamma_corrected, 1.0 + sharpen_amount, blurred, -sharpen_amount, 0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    _emit(progress_callback, 'sharpen', 'done')

    _emit(progress_callback, 'color', 'running')
    pil_img = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
    color_boosted = ImageEnhance.Color(pil_img).enhance(color_factor)
    brightened = ImageEnhance.Brightness(color_boosted).enhance(brightness_factor)
    final = brightened.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3))
    _emit(progress_callback, 'color', 'done')

    return final, s


def enhance_image(image_bgr: np.ndarray, settings: dict | None = None) -> np.ndarray:
    final_pil, _ = _enhance_pipeline(image_bgr, settings=settings, progress_callback=None)
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


def calculate_enhancement_stats(original_path: str, enhanced_img: Image.Image, settings: dict | None = None) -> dict:
    s = _normalize_settings(settings)

    orig = np.array(Image.open(original_path).convert('L'), dtype=np.float32)
    enh = np.array(enhanced_img.convert('L'), dtype=np.float32)

    brightness_delta = int(max(0, enh.mean() - orig.mean()))
    contrast_delta = int(max(0, enh.std() - orig.std()))

    noise_reduction = int(min(85, 45 + (s['denoise'] * 0.4)))
    sharpness_boost = int(min(75, 20 + (s['sharpness'] * 0.5)))

    return {
        'brightness_increase': f'+{brightness_delta}',
        'contrast_improvement': f'+{contrast_delta}%',
        'noise_reduction': f'-{noise_reduction}%',
        'sharpness_boost': f'+{sharpness_boost}%',
    }


def enhance_night_image(image_path: str, settings: dict | None = None, progress_callback=None) -> dict:
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f'Could not read image from path: {image_path}')

    enhanced_pil, normalized_settings = _enhance_pipeline(
        image_bgr,
        settings=settings,
        progress_callback=progress_callback,
    )

    comparison_img = generate_comparison(image_path, enhanced_pil)
    stats = calculate_enhancement_stats(image_path, enhanced_pil, settings=normalized_settings)

    return {
        'enhanced_image': enhanced_pil,
        'comparison_image': comparison_img,
        'stats': stats,
    }
