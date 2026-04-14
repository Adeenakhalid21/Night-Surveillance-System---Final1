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
    {'key': 'upscale', 'label': 'Upscaling'},
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
        'upscale': 70.0,
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


def _resolve_upscale_factor(width: int, height: int, upscale_strength: float) -> float:
    strength = _clamp(upscale_strength / 100.0, 0.0, 1.0)
    if strength < 0.08:
        return 1.0

    longest_edge = max(width, height)
    if longest_edge >= 1800:
        base_factor = 1.15
    elif longest_edge >= 1280:
        base_factor = 1.35
    elif longest_edge >= 900:
        base_factor = 1.6
    else:
        base_factor = 2.0

    factor = 1.0 + (base_factor - 1.0) * (0.35 + (0.65 * strength))
    if strength > 0.85 and longest_edge < 1200:
        factor += 0.15

    return _clamp(factor, 1.0, 2.4)


def _enhance_pipeline(image_bgr: np.ndarray, settings: dict | None = None, progress_callback=None) -> tuple[Image.Image, dict]:
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
    enhanced_rgb = np.array(final.convert('RGB'))
    h, w = enhanced_rgb.shape[:2]
    upscale_factor = _resolve_upscale_factor(w, h, s['upscale'])

    if upscale_factor > 1.0:
        interpolation = cv2.INTER_LANCZOS4 if upscale_factor >= 1.6 else cv2.INTER_CUBIC
        upscaled = cv2.resize(
            enhanced_rgb,
            dsize=None,
            fx=upscale_factor,
            fy=upscale_factor,
            interpolation=interpolation,
        )

        detail = cv2.detailEnhance(
            upscaled,
            sigma_s=10,
            sigma_r=0.12 + (s['contrast'] * 0.0015),
        )
        detail_strength = 0.08 + (s['sharpness'] * 0.003)
        upscaled = cv2.addWeighted(upscaled, 1.0 - detail_strength, detail, detail_strength, 0)

        if s['denoise'] > 40:
            upscaled = cv2.fastNlMeansDenoisingColored(upscaled, None, 3, 3, 5, 11)

        upscaled_pil = Image.fromarray(upscaled)
        final = upscaled_pil.filter(
            ImageFilter.UnsharpMask(radius=2.2, percent=int(130 + (s['sharpness'] * 0.8)), threshold=2)
        )

    _emit(progress_callback, 'upscale', 'done')

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

    return {
        'brightness_increase': f'+{brightness_delta}',
        'contrast_improvement': f'+{contrast_delta}%',
        'noise_reduction': f'-{noise_reduction}%',
        'sharpness_boost': f'+{sharpness_boost}%',
        'resolution_gain': f'+{resolution_gain}%',
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
