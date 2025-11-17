import cv2
from scipy import ndimage

def enhance_image(image):
    # Convert image to gray scale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Non-Linear filter for noise removal
    deNoised = ndimage.median_filter(gray_image, 3)

    # Histogram Equalizer
    # High pass filter for improving the contrast of the image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    highPass = clahe.apply(deNoised)

    # Gamma Transformation
    # Prevent bleaching or darkening of images
    gamma = highPass / 255.0
    gammaFilter = cv2.pow(gamma, 1.5)
    gammaFilter = gammaFilter * 255

    # Convert enhanced image back to BGR
    enhanced_frame = cv2.cvtColor(gammaFilter.astype('uint8'), cv2.COLOR_GRAY2BGR)

    return enhanced_frame
