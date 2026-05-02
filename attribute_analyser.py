import cv2
import numpy as np
from query_parser import COLOR_NAMES

def get_region_crop(frame, bbox, region):
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    regions = {
        "head":  (x1, y1,            x2, y1 + int(h * 0.20)),
        "upper": (x1, y1 + int(h * 0.20), x2, y1 + int(h * 0.55)),
        "lower": (x1, y1 + int(h * 0.55), x2, y1 + int(h * 0.85)),
        "feet":  (x1, y1 + int(h * 0.85), x2, y2),
    }
    rx1, ry1, rx2, ry2 = regions.get(region, regions["upper"])
    crop = frame[ry1:ry2, rx1:rx2]
    return crop if crop.size > 0 else None

def dominant_color(crop):
    if crop is None or crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    best_color, best_count = None, 0
    for color_name, (lo, hi) in COLOR_NAMES.items():
        mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
        count = cv2.countNonZero(mask)
        if count > best_count:
            best_count = count
            best_color = color_name
    return best_color

def analyse_person(frame, bbox):
    return {
        region: dominant_color(get_region_crop(frame, bbox, region))
        for region in ("head", "upper", "lower", "feet")
    }
