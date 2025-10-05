"""
Symbol segmentation using OpenCV.
"""
import cv2
import numpy as np


def segment_image(image, min_area=100):
    """Takes a grayscale image (numpy array) and returns list of symbol images (cropped arrays) sorted left-to-right.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    # Use adaptive thresholding for variable lighting/ink
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 25, 10)
    # Morph to join strokes and remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w*h < min_area:
            continue
        boxes.append((x,y,w,h))
    boxes = sorted(boxes, key=lambda b: b[0])
    # Merge boxes that are close horizontally or clearly part of the same symbol
    merged = []
    for box in boxes:
        if not merged:
            merged.append(box)
            continue
        x,y,w,h = box
        px,py,pw,ph = merged[-1]
        # compute vertical overlap
        y0 = max(y, py)
        y1 = min(y+h, py+ph)
        vert_overlap = max(0, y1 - y0)
        vert_frac = vert_overlap / min(h, ph) if min(h, ph) > 0 else 0
        gap = x - (px + pw)
        # merge if boxes touch/near or have strong vertical overlap (parts of same glyph)
        if gap <= max(6, int(0.15 * pw)) or vert_frac > 0.45:
            nx = min(px, x)
            ny = min(py, y)
            nw = max(px+pw, x+w) - nx
            nh = max(py+ph, y+h) - ny
            # Prevent merging into a box that covers most of the line width
            img_w = th.shape[1]
            if nw < 0.60 * img_w and nw < (pw * 3):
                merged[-1] = (nx, ny, nw, nh)
            else:
                # skip merging if it would create an overly large box
                merged.append(box)
        else:
            merged.append(box)
    boxes = merged
    # If boxes are too few, try to split very wide boxes by vertical projection
    widths = [w for (_,_,w,_) in boxes] if boxes else [0]
    med_w = int(np.median(widths)) if widths else 0
    final_boxes = []
    for (x,y,w,h) in boxes:
        if med_w>0 and w > 1.4 * med_w and w > 2*med_w and w>40:
            # attempt to split by vertical projection of this crop
            crop = th[y:y+h, x:x+w]
            proj = np.sum(crop==0, axis=0)  # black pixel count per column
            # find candidate split positions where proj is small (mostly white)
            white_cols = np.where(proj < (0.08 * h))[0]
            # find largest gaps in white_cols to split
            if white_cols.size > 0:
                splits = []
                groups = np.split(white_cols, np.where(np.diff(white_cols) != 1)[0]+1)
                for g in groups:
                    if g.size>0:
                        splits.append(int(g[g.size//2]))
                # create sub-boxes between splits
                xs = [0] + splits + [w-1]
                for i in range(len(xs)-1):
                    sx = xs[i]
                    ex = xs[i+1]
                    sw = ex - sx
                    if sw>10:
                        final_boxes.append((x+sx, y, sw, h))
                continue
            # fallback: split at the largest local minima in projection
            proj_smooth = np.convolve(proj.astype(float), np.ones(3)/3, mode='same')
            # find minima positions
            minima = (np.r_[True, proj_smooth[1:] < proj_smooth[:-1]] & np.r_[proj_smooth[:-1] < proj_smooth[1:], True]).nonzero()[0]
            if minima.size>0:
                # sort minima by proj value (ascending) and take up to 2 splits
                mins = sorted(list(minima), key=lambda i: proj_smooth[i])
                k = min(2, max(1, int(round(w/med_w))-1))
                chosen = sorted(mins[:k])
                xs = [0] + chosen + [w-1]
                for i in range(len(xs)-1):
                    sx = xs[i]
                    ex = xs[i+1]
                    sw = ex - sx
                    if sw>10:
                        final_boxes.append((x+sx, y, sw, h))
                continue
        final_boxes.append((x,y,w,h))
    boxes = final_boxes
    # Fallback: if too few boxes (likely merged), try vertical projection over whole line
    if len(boxes) <= 2:
        proj_full = np.sum(th == 0, axis=0)
        groups = []
        in_group = False
        start = 0
        for i, v in enumerate(proj_full):
            if not in_group and v > 0:
                in_group = True
                start = i
            elif in_group and v == 0:
                in_group = False
                groups.append((start, i-1))
        if in_group:
            groups.append((start, len(proj_full)-1))
        # filter tiny groups
        groups = [g for g in groups if (g[1]-g[0]) > 6]
        if len(groups) >= 2 and len(groups) <= 8:
            boxes = []
            h = th.shape[0]
            for (sx, ex) in groups:
                sw = ex - sx + 1
                boxes.append((sx, 0, sw, h))
        else:
            # Peak-based fallback: find local maxima in projection and make centered boxes
            proj = proj_full
            maxv = proj.max() if proj.size>0 else 0
            if maxv > 0:
                thresh = max(4, int(0.12 * maxv))
                peaks = []
                for i in range(1, len(proj)-1):
                    if proj[i] > proj[i-1] and proj[i] >= proj[i+1] and proj[i] >= thresh:
                        peaks.append(i)
                # remove peaks that are too close
                filtered = []
                last = -999
                for p in peaks:
                    if p - last > 12:
                        filtered.append(p)
                        last = p
                peaks = filtered
                if len(peaks) >= 2 and len(peaks) <= 12:
                    boxes = []
                    h = th.shape[0]
                    # median width guess
                    guess_w = int(np.median([w for (_,_,w,_) in boxes]) if boxes else max(20, th.shape[1]//8))
                    for p in peaks:
                        half = max(12, guess_w//2)
                        sx = max(0, p-half)
                        ex = min(th.shape[1]-1, p+half)
                        boxes.append((sx, 0, ex-sx+1, h))
    symbols = []
    for (x,y,w,h) in boxes:
        crop = gray[y:y+h, x:x+w]
        # pad to square
        size = max(w,h)
        pad_x = (size - w) // 2
        pad_y = (size - h) // 2
        square = 255 * np.ones((size,size), dtype=np.uint8)
        square[pad_y:pad_y+h, pad_x:pad_x+w] = crop
        symbols.append(square)
    return symbols
