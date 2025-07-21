import cv2
import numpy as np
from PIL import Image
import pytesseract
import logging

logger = logging.getLogger(__name__)

def deskew_image(image):
    """Deskew an image to correct rotation."""
    # Convert PIL image to OpenCV format
    img_array = np.array(image)
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Edge detection to find text alignment
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines using Hough Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    angle = 0
    if lines is not None:
        for rho, theta in lines[0]:
            angle = (theta * 180 / np.pi) - 90
            break

    logger.debug(f"{angle=}")
    
    # Rotate image to deskew
    if abs(angle) > 0.5:  # Only rotate if skew is significant
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    # Convert back to PIL image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def clean_image(image):
    """Remove punch holes, borders, and artifacts from an image."""
    # Convert PIL image to OpenCV format
    img_array = np.array(image)
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast for better OCR and artifact removal
    gray = cv2.equalizeHist(gray)
    
    # Remove punch holes (detect circles and fill them)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                              param1=50, param2=30, minRadius=5, maxRadius=20)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(img, (x, y), r, (255, 255, 255), -1)
    
    # Remove dark borders and shadows with adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    mask = cv2.bitwise_not(thresh)
    cleaned = cv2.bitwise_and(img, img, mask=mask)
    
    # Apply morphological operation to remove small noise
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    # Convert back to PIL image
    cleaned_rgb = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cleaned_rgb)

def detect_double_page(image):
    """Detect if an image contains two pages using OCR."""
    width, height = image.size
    aspect_ratio = width / height
    
    # Only check landscape pages (width > height)
    if aspect_ratio <= 1.0:
        return False
    
    # Split image into left and right halves
    left_half = image.crop((0, 0, width // 2, height))
    right_half = image.crop((width // 2, 0, width, height))
    
    # Perform OCR on both halves with improved settings
    config = '--psm 3'  # Default mode for better handling of skewed text
    left_text = pytesseract.image_to_string(left_half, config=config).strip()
    logger.debug(f"{left_text=}")
    right_text = pytesseract.image_to_string(right_half, config=config).strip()
    logger.debug(f"{right_text=}")
    
    # Consider it double-page if both halves have significant text (>10 characters)
    return len(left_text) > 10 and len(right_text) > 10