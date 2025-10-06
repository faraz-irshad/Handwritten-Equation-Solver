import cv2
import numpy as np
try:
    from scipy import ndimage
    from skimage.morphology import skeletonize
except ImportError:
    ndimage = None
    skeletonize = None

def stroke_width_transform(binary_img):
    """Stroke Width Transform for better text detection"""
    # Distance transform
    dist = cv2.distanceTransform(binary_img, cv2.DIST_L2, 5)
    
    # Skeleton
    skeleton = skeletonize(binary_img > 0)
    
    # Stroke width map
    stroke_width = np.zeros_like(dist)
    skeleton_points = np.where(skeleton)
    
    for y, x in zip(skeleton_points[0], skeleton_points[1]):
        stroke_width[y, x] = dist[y, x] * 2
    
    return stroke_width

def segment_image(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Preprocessing
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Multi-scale thresholding for robustness
    thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 3)
    _, thresh3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Combine thresholds
    thresh = cv2.bitwise_or(thresh1, cv2.bitwise_or(thresh2, thresh3))
    
    # Stroke width filtering
    try:
        stroke_map = stroke_width_transform(thresh)
        # Filter by stroke width consistency
        mean_stroke = np.mean(stroke_map[stroke_map > 0])
        stroke_mask = (stroke_map > mean_stroke * 0.3) & (stroke_map < mean_stroke * 3)
        thresh = cv2.bitwise_and(thresh, thresh, mask=stroke_mask.astype(np.uint8) * 255)
    except:
        pass  # Fallback to original thresh
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Connected component analysis (more robust than contours)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    
    # Convert to contour format for compatibility
    contours = []
    for i in range(1, num_labels):  # Skip background
        mask = (labels == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            contours.extend(cnts)
    
    # Get bounding boxes with better filtering
    boxes = []
    img_area = gray.shape[0] * gray.shape[1]
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        
        # Advanced filtering based on text characteristics
        aspect_ratio = w / h
        solidity = area / (w * h)  # How filled the bounding box is
        
        if (area > 100 and area < img_area * 0.3 and 
            w > 8 and h > 8 and 
            0.1 < aspect_ratio < 8.0 and  # More flexible aspect ratio
            solidity > 0.3):  # Must be reasonably filled
            boxes.append((x, y, w, h, area, solidity))
    
    # Sort by x-coordinate (left to right) with y-coordinate tie-breaking
    boxes = sorted(boxes, key=lambda b: (b[1] // 20, b[0]))  # Group by rows first
    
    # Smart merging based on text characteristics
    merged_boxes = []
    for box in boxes:
        x, y, w, h, area, solidity = box
        merged = False
        
        for i, (mx, my, mw, mh) in enumerate(merged_boxes):
            # Check for overlap with better heuristics
            x_overlap = max(0, min(x + w, mx + mw) - max(x, mx))
            y_overlap = max(0, min(y + h, my + mh) - max(y, my))
            
            # Merge if significant overlap or very close horizontally
            if ((x_overlap > 0 and y_overlap > min(h, mh) * 0.3) or 
                (abs(y - my) < max(h, mh) * 0.5 and abs(x - (mx + mw)) < w * 0.3)):
                
                new_x = min(x, mx)
                new_y = min(y, my)
                new_w = max(x + w, mx + mw) - new_x
                new_h = max(y + h, my + mh) - new_y
                merged_boxes[i] = (new_x, new_y, new_w, new_h)
                merged = True
                break
        
        if not merged:
            merged_boxes.append((x, y, w, h))
    
    # Extract symbols
    symbols = []
    for x, y, w, h in merged_boxes:
        # Add padding around the symbol
        pad = 5
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(gray.shape[1], x + w + pad)
        y2 = min(gray.shape[0], y + h + pad)
        
        crop = gray[y1:y2, x1:x2]
        
        # Smart inversion based on text vs background
        hist = cv2.calcHist([crop], [0], None, [256], [0, 256])
        # If more dark pixels, likely inverted
        if np.sum(hist[:128]) > np.sum(hist[128:]):
            crop = 255 - crop
        
        # Enhance contrast
        crop = cv2.equalizeHist(crop)
        
        # Aspect-ratio preserving resize with padding
        target_size = 64
        h, w = crop.shape
        
        # Scale to fit in target size while preserving aspect ratio
        scale = min(target_size / w, target_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        if new_w > 0 and new_h > 0:
            resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized = crop
        
        # Create square with padding
        square = np.full((target_size, target_size), 255, dtype=np.uint8)
        
        # Center the resized image
        start_y = (target_size - resized.shape[0]) // 2
        start_x = (target_size - resized.shape[1]) // 2
        end_y = start_y + resized.shape[0]
        end_x = start_x + resized.shape[1]
        
        square[start_y:end_y, start_x:end_x] = resized
        
        symbols.append(square)
    
    return symbols