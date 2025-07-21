import cv2
import numpy as np
from PIL import Image
import pytesseract
import logging

logger = logging.getLogger(__name__)

def deskew_image(image):
    """Deskew an image to correct rotation and orientation."""
    # Convert PIL image to OpenCV format
    img_array = np.array(image)
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast and apply preprocessing for better OSD
    gray = cv2.equalizeHist(gray)
    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # Use adaptive thresholding for better text detection
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 15, 5)
    
    # Use Tesseract OSD to detect orientation
    try:
        osd = pytesseract.image_to_osd(gray, config='--psm 0')
        angle = float(osd.split('Rotate: ')[1].split('\n')[0])
        confidence = float(osd.split('Confidence: ')[1].split('\n')[0])
        logger.debug(f"Tesseract OSD angle: {angle}, confidence: {confidence}")
    except pytesseract.pytesseract.TesseractError as e:
        logger.warning(f"Tesseract OSD failed: {e}. Attempting fallback.")
        angle = 0
        confidence = 0
    
    # If Tesseract confidence is low (< 50) or angle is suspicious, use fallback
    if confidence < 50 or abs(angle) in [90, 180, 270]:
        logger.warning(f"Low confidence ({confidence}) or suspicious angle ({angle}). Using fallback.")
        try:
            # Binarize for contour detection
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour (assuming it contains text)
                largest_contour = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(largest_contour)
                angle_fallback = rect[-1]
                # Convert angle to range [-90, 90]
                if angle_fallback < -45.0:
                    angle_fallback = 90 + angle_fallback
                elif angle_fallback > 45.0:
                    angle_fallback = -(90 - angle_fallback)
                logger.debug(f"Fallback detected angle: {angle_fallback} from contour")
                angle = angle_fallback
            else:
                logger.warning("No contours detected for fallback")
                angle = 0
        except Exception as e:
            logger.warning(f"Fallback failed: {e}. Using default angle 0.")
            angle = 0
    
    # Apply rotation only if angle is significant (avoid over-rotating small tilts)
    if abs(angle) > 1.0:  # Increased threshold to avoid minor corrections
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        logger.debug(f"Applied rotation: {angle} degrees")
    
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
                                   cv2.THRESH_BINARY, 21, 5)
    mask = cv2.bitwise_not(thresh)
    cleaned = cv2.bitwise_and(img, img, mask=mask)
    
    # Apply morphological operation to remove small noise
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    # Convert back to PIL image
    cleaned_rgb = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cleaned_rgb)

def detect_double_page(image):
    """Detect if an image contains two pages using OCR, handling both horizontal and vertical layouts."""
    width, height = image.size
    page_aspect_ratio = width / height
    
    logger.debug(f"Page dimensions: width={width}, height={height}, aspect_ratio={page_aspect_ratio}")
    
    # Preprocess image for better OCR
    img_array = np.array(image)
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Denoising
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 21, 5)
    preprocessed = Image.fromarray(thresh)
    
    # Determine split direction based on aspect ratio
    if page_aspect_ratio > 1.5:
        midpoint = width // 2
        split_type = "horizontal"
    elif page_aspect_ratio < 0.67:
        midpoint = height // 2
        split_type = "vertical"
    else:
        logger.debug("Aspect ratio suggests single page, no split")
        return False, None
    
    tolerance = max(10, width // 50)  # 10 pixels or 2% of width
    
    # Perform OCR to get text block data
    config = '--psm 3'
    data = pytesseract.image_to_data(preprocessed, config=config, output_type=pytesseract.Output.DICT)
    
    # Filter valid text blocks (confidence > 0)
    blocks = [(data['left'][i], data['top'][i], data['width'][i], data['height'][i])
              for i in range(len(data['text'])) if data['conf'][i] > 0]
    
    if not blocks:
        logger.debug("No text blocks detected")
        return False, None
    
    # Analyze distribution based on split type
    if split_type == "horizontal":
        left_edges = [left for left, _, _, _ in blocks]
        right_edges = [left + width for left, _, width, _ in blocks]
        crossings = sum(1 for left, right in zip(left_edges, right_edges)
                       if (left < midpoint < right) or (abs(left - midpoint) < tolerance) or (abs(right - midpoint) < tolerance))
    else:  # vertical
        top_edges = [top for _, top, _, _ in blocks]
        bottom_edges = [top + height for _, top, _, height in blocks]
        crossings = sum(1 for top, bottom in zip(top_edges, bottom_edges)
                       if (top < midpoint < bottom) or (abs(top - midpoint) < tolerance) or (abs(bottom - midpoint) < tolerance))
    
    logger.debug(f"Midpoint crossings: {crossings}, Total blocks: {len(blocks)}")
    
    # If few crossings (e.g., < 20% of blocks), assume a natural break
    if crossings / len(blocks) < 0.2 and len(blocks) > 5:  # Minimum content threshold
        # Perform split
        if split_type == "horizontal":
            left_half = preprocessed.crop((0, 0, width // 2, height))
            right_half = preprocessed.crop((width // 2, 0, width, height))
        else:  # vertical
            left_half = preprocessed.crop((0, 0, width, height // 2))
            right_half = preprocessed.crop((0, height // 2, width, height))
        
        # Verify each half has content
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