import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
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

def preprocess_for_ocr(image):
    """Preprocess image to enhance text visibility for OCR, similar to pdf_orientation_corrector."""
    # Convert to grayscale
    image = image.convert('L')
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)
    # Apply median filter to reduce noise
    image = image.filter(ImageFilter.MedianFilter())
    # Apply binary thresholding
    threshold = 120
    image = image.point(lambda p: p > threshold and 255)
    return image

def detect_double_page(image):
    """Detect if an image contains two pages by analyzing text block distribution.
    
    Args:
        image: PIL.Image object of the PDF page.
    
    Returns:
        tuple: (is_double_page: bool, split_type: str or None)
        - is_double_page: True if the image contains two pages, False otherwise.
        - split_type: 'horizontal' if two pages side-by-side, None otherwise.
    """
    width, height = image.size
    logger.debug(f"Image dimensions: width={width}, height={height}")

    # Step 1: Preprocess image for OCR
    preprocessed = preprocess_for_ocr(image)

    # Step 2: Perform OCR to get text block data
    config = '--psm 4'  # Assume column-based layout for better double-page detection
    data = pytesseract.image_to_data(preprocessed, config=config, output_type=pytesseract.Output.DICT)
    blocks = [
        (data['left'][i], data['top'][i], data['width'][i], data['height'][i], data['text'][i], data['conf'][i])
        for i in range(len(data['text']))
        if data['conf'][i] > 30 and data['text'][i].strip()  # Lower confidence threshold
    ]
    logger.debug(f"Detected {len(blocks)} text blocks with confidence > 30")
    if blocks:
        logger.debug(f"Sample text blocks: {[(b[4], b[5]) for b in blocks[:3]]}")

    # Step 3: Handle case with insufficient text blocks
    if len(blocks) < 2:
        logger.debug("Too few text blocks detected, assuming single page")
        return False, None

    # Step 4: Analyze text block distribution for a natural break
    # Calculate left and right edges of text blocks
    left_edges = [left for left, _, width, _, _, _ in blocks]
    right_edges = [left + width for left, _, width, _, _, _ in blocks]
    midpoint = width // 2
    tolerance = max(10, width // 50)  # Tolerance for blocks near midpoint

    # Count blocks that cross the midpoint or are close to it
    crossings = sum(
        1 for left, right in zip(left_edges, right_edges)
        if (left < midpoint < right) or
           (abs(left - midpoint) < tolerance) or
           (abs(right - midpoint) < tolerance)
    )
    logger.debug(f"Midpoint crossings: {crossings}, Total blocks: {len(blocks)}")

    # Step 5: Check for a natural break (gap in the middle)
    # Sort blocks by left edge and find gaps
    sorted_left_edges = sorted(left_edges)
    sorted_right_edges = sorted(right_edges)
    mid_range = (midpoint - tolerance, midpoint + tolerance)
    blocks_left_side = [b for b in blocks if b[0] + b[2] <= mid_range[1]]  # Blocks ending before mid_range
    blocks_right_side = [b for b in blocks if b[0] >= mid_range[0]]  # Blocks starting after mid_range

    # Calculate gap by finding max right edge on left side and min left edge on right side
    max_right_left_side = max((b[0] + b[2] for b in blocks_left_side), default=0)
    min_left_right_side = min((b[0] for b in blocks_right_side), default=width)
    gap_size = min_left_right_side - max_right_left_side
    logger.debug(f"Gap size in middle: {gap_size}px, Left side blocks: {len(blocks_left_side)}, Right side blocks: {len(blocks_right_side)}")

    # Step 6: Validate split with text content
    min_gap_size = max(100, width // 20)  # Allow smaller gutters
    if gap_size > min_gap_size and len(blocks_left_side) > 1 and len(blocks_right_side) > 1 and crossings < len(blocks) * 0.05:
        # Verify text content in both halves
        left_half = preprocessed.crop((0, 0, midpoint, height))
        right_half = preprocessed.crop((midpoint, 0, width, height))
        left_text = pytesseract.image_to_string(left_half, config=config).strip()
        right_text = pytesseract.image_to_string(right_half, config=config).strip()
        logger.debug(f"Left half text length: {len(left_text)}, Right half text length: {len(right_text)}")

        if len(left_text) > 10 and len(right_text) > 10:
            logger.debug("Natural break detected with sufficient text, splitting horizontally")
            return True, "horizontal"
        else:
            logger.debug("Insufficient text in one half, assuming single page")
            return False, None
    else:
        logger.debug(f"No significant gap (gap_size={gap_size}px, required>{min_gap_size}px) or too many crossings ({crossings}), assuming single page")
        return False, None