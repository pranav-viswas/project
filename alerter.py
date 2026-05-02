import cv2
import os
import base64
from datetime import datetime

os.makedirs("reports", exist_ok=True)

def encode_frame(frame):
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf).decode('utf-8')

def draw_alert_box(frame, bbox, score, details):
    x1, y1, x2, y2 = bbox
    color = (0, 120, 255)
    label = f"MATCH {score*100:.0f}%"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 15), (x1 + tw + 10, y1), color, -1)
    cv2.putText(frame, label, (x1 + 5, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Warning overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 50), (0, 0, 200), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, "TARGET DETECTED",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return frame

class EventLogger:
    def __init__(self):
        self.events = []

    def log(self, event):
        self.events.append(event)

    def get_all(self):
        return self.events
