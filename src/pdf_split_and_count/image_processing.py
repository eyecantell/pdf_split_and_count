import cv2
import numpy as np
from PIL import Image
import pytesseract
import logging

logger = logging.getLogger(__name__)

def clean_image(image):
    """Remove punch holes, borders, and artifacts from an image."""
    img_array = np.array(image)
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                              param1=50, param2=30, minRadius=5, maxRadius=20)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(img, (x, y), r, (255, 255, 255), -1)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 21, 5)
    mask = cv2.bitwise_not(thresh)
    cleaned = cv2.bitwise_and(img, img, mask=mask)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    cleaned_rgb = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cleaned_rgb)

def detect_double_page(image):
    """Detect if an image contains two pages using OCR, handling both horizontal and vertical layouts."""
    width, height = image.size
    page_aspect_ratio = width / height
    logger.debug(f"Page dimensions: width={width}, height={height}, aspect_ratio={page_aspect_ratio}")
    img_array = np.array(image)
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 21, 5)
    preprocessed = Image.fromarray(thresh)
    config = '--psm 3'
    data = pytesseract.image_to_data(preprocessed, config=config, output_type=pytesseract.Output.DICT)
    blocks = [(data['left'][i], data['top'][i], data['width'][i], data['height'][i])
              for i in range(len(data['text'])) if data['conf'][i] > 0]
    if not blocks:
        logger.debug("No text blocks detected")
        return False, None
    if page_aspect_ratio > 1.5:
        midpoint = width // 2
        split_type = "horizontal"
    elif page_aspect_ratio < 0.67:
        midpoint = height // 2
        split_type = "vertical"
    else:
        logger.debug("Aspect ratio suggests single page, no split")
        return False, None
    tolerance = max(10, width // 50)
    if split_type == "horizontal":
        left_edges = [left for left, _, _, _ in blocks]
        right_edges = [left + width for left, _, width, _ in blocks]
        crossings = sum(1 for left, right in zip(left_edges, right_edges)
                       if (left < midpoint < right) or (abs(left - midpoint) < tolerance) or (abs(right - midpoint) < tolerance))
    else:
        top_edges = [top for _, top, _, _ in blocks]
        bottom_edges = [top + height for _, top, _, height in blocks]
        crossings = sum(1 for top, bottom in zip(top_edges, bottom_edges)
                       if (top < midpoint < bottom) or (abs(top - midpoint) < tolerance) or (abs(bottom - midpoint) < tolerance))
    logger.debug(f"Midpoint crossings: {crossings}, Total blocks: {len(blocks)}")
    if crossings / len(blocks) < 0.2 and len(blocks) > 5:
        if split_type == "horizontal":
            left_half = preprocessed.crop((0, 0, width // 2, height))
            right_half = preprocessed.crop((width // 2, 0, width, height))
        else:
            left_half = preprocessed.crop((0, 0, width, height // 2))
            right_half = preprocessed.crop((0, height // 2, width, height))
        left_text = pytesseract.image_to_string(left_half, config=config).strip()
        right_text = pytesseract.image_to_string(right_half, config=config).strip()
        logger.debug(f"Left text: {left_text}, Right text: {right_text}")
        if len(left_text) > 5 and len(right_text) > 5:
            logger.debug(f"Natural break detected, splitting {split_type}")
            return True, split_type
        else:
            logger.debug("Insufficient content in one half, no split")
            return False, None
    else:
        logger.debug("No natural break detected near midpoint")
        return False, None