#!/usr/bin/env python3
"""Cross-check skew on a single PNG using three independent methods."""
import sys
import numpy as np
import cv2
from deskew import determine_skew

def method_deskew(gray):
    return determine_skew(gray)

def method_deskew_hires(gray):
    # No downscale.
    return determine_skew(gray)

def method_hough(gray):
    """Hough-based angle estimation. Uses minarea-rect over text components."""
    # Invert + threshold so text is white.
    _, bw = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # Dilate horizontally so words merge into lines.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
    dil = cv2.dilate(bw, kernel, iterations=1)
    coords = np.column_stack(np.where(dil > 0))
    if len(coords) < 100:
        return None
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    # cv2 minAreaRect returns angle in [-90, 0). Convert to a small skew angle.
    if angle < -45:
        angle = 90 + angle
    return -angle

def method_hough_lines(gray):
    """Estimate dominant line angle via HoughLinesP on text baselines."""
    _, bw = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    bw = cv2.dilate(bw, kernel, iterations=1)
    edges = cv2.Canny(bw, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 1800, threshold=200,
                            minLineLength=100, maxLineGap=10)
    if lines is None:
        return None
    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        if x2 == x1:
            continue
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Only near-horizontal lines count as baselines
        if -20 < a < 20:
            angles.append(a)
    if not angles:
        return None
    return float(np.median(angles))

def main():
    for path in sys.argv[1:]:
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"{path}: unreadable")
            continue
        # Downscaled copy for deskew (matches pipeline behavior).
        h, w = gray.shape
        max_dim = 1500
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            gray_small = cv2.resize(gray, (int(w*scale), int(h*scale)),
                                    interpolation=cv2.INTER_AREA)
        else:
            gray_small = gray
        a1 = method_deskew(gray_small)
        a2 = method_deskew_hires(gray)
        a3 = method_hough(gray)
        a4 = method_hough_lines(gray)
        print(f"{path}")
        print(f"  deskew (downscaled 1500px): {a1}")
        print(f"  deskew (full {w}x{h}):      {a2}")
        print(f"  Hough minAreaRect:          {a3}")
        print(f"  Hough lines median:         {a4}")
        print()

if __name__ == "__main__":
    main()
