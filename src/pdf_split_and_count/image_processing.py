import cv2
import numpy as np
from PIL import Image
import pytesseract
import logging

logger = logging.getLogger(__name__)


def deskew_image(image):
    """Deskew an image by detecting its orientation using Tesseract OSD."""
    # Convert PIL image to OpenCV format
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Run Tesseract OSD to detect orientation
    osd = pytesseract.image_to_osd(gray, config='--psm 0')
    angle = 0.0
    confidence = 0.0
    text_length = 0
    script_conf = 0.0
    
    for line in osd.split('\n'):
        if 'Rotate' in line:
            angle = float(line.split(': ')[1])
        if 'Confidence' in line:
            confidence = float(line.split(': ')[1])
        if 'Text' in line:
            text_length = len(line.split(': ')[1])
        if 'Script confidence' in line:
            script_conf = float(line.split(': ')[1])
    
    logger.debug(f"Tesseract OSD angle: {angle}, script confidence: {script_conf}")
    
    # Normalize angle to [0, 360)
    angle = angle % 360
    logger.debug(f"Normalized OSD angle: {angle}")
    
    # Verify OSD reliability
    if text_length > 10 and confidence > 50.0 and script_conf > 0.5:
        logger.debug(f"OSD angle {angle} verified (text length: {text_length}, confidence: {confidence:.2f}, script confidence: {script_conf:.2f})")
    else:
        logger.warning(f"OSD angle {angle} unreliable (text length: {text_length}, confidence: {confidence:.2f}, script confidence: {script_conf:.2f}), trying all orientations.")
        max_confidence = 0.0
        best_angle = 0
        best_text = ""
        best_top_text = []
        for test_angle in [0, 90, 180, 270]:
            rotated = image.rotate(test_angle, expand=True)
            data = pytesseract.image_to_data(rotated, config='--psm 6 -c textord_tabfind_find_tables=0', output_type=pytesseract.Output.DICT)
            total_conf = sum(float(c) for c in data['conf'] if float(c) > 0)
            text = data['text']
            top_texts = [text[i] for i, top in enumerate(data['top']) if text[i].strip() and top < rotated.height // 4]
            logger.debug(f"Angle {test_angle}: Total confidence = {total_conf:.2f}, Text length = {len(''.join(text))}, Top text count = {len(top_texts)}")
            if total_conf > max_confidence:
                max_confidence = total_conf
                best_angle = test_angle
                best_text = ''.join(text)
                best_top_text = top_texts
        angle = best_angle
        logger.debug(f"Selected angle from confidence: {angle} with confidence: {max_confidence:.2f}")
        
        # Check if text at top is right-side-up (for 90 vs. 270)
        if angle in [90, 270] and best_top_text:
            rotated = image.rotate(angle, expand=True)
            data = pytesseract.image_to_data(rotated, config='--psm 6 -c textord_tabfind_find_tables=0', output_type=pytesseract.Output.DICT)
            top_conf = sum(float(c) for i, c in enumerate(data['conf']) if data['text'][i].strip() and data['top'][i] < rotated.height // 4 and float(c) > 0)
            if top_conf < 100 or len(best_top_text) < 2 or script_conf < 0.5:
                angle = (angle + 180) % 360  # Switch 90 to 270 or vice versa
                logger.debug(f"Switched to angle {angle} to ensure right-side-up text (top conf: {top_conf:.2f}, top text count: {len(best_top_text)})")
    
    # Apply rotation
    if abs(angle) > 0.1:  # Ensure rotation for non-zero angles
        image = image.rotate(angle, expand=True)
        logger.debug(f"Applied rotation of {angle} degrees")
    else:
        logger.debug("No rotation applied (angle = 0)")
    
    # Fallback to contour-based deskewing if needed
    if abs(angle) < 0.1 and text_length < 10:
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        if lines is not None:
            for rho, theta in lines[0]:
                angle = (theta * 180 / np.pi) - 90
                angle = angle % 360  # Normalize to [0, 360)
                if abs(angle) > 10:
                    image = image.rotate(angle, expand=True)
                    logger.debug(f"Fallback detected angle: {angle}")
                    break
    
    return image

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