import cv2

def enhance_image(image):
    """Light enhancement while preserving color.

    Steps:
      1. Convert to LAB and apply CLAHE on L channel for contrast.
      2. Median blur L slightly for noise suppression.
      3. Optional mild gamma correction on luminance LUT.
      4. Recombine with original A/B for true color output.
    """
    if image is None or image.size == 0:
        return image

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)

    l_eq = cv2.medianBlur(l_eq, 3)

    # Gamma correction (mild) using LUT for speed
    gamma = 1.15
    inv_gamma = 1.0 / gamma
    lut = ( ( (i / 255.0) ** inv_gamma ) * 255 for i in range(256) )
    lut = bytearray(int(v) for v in lut)
    l_gamma = cv2.LUT(l_eq, lut)

    lab_enhanced = cv2.merge((l_gamma, a, b))
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return enhanced
