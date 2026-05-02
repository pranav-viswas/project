from ultralytics import YOLO
import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = os.getenv("SMARTSEARCH_DETECTOR_MODEL", "yolov8s.pt")
inference_size = int(os.getenv("SMARTSEARCH_IMAGE_SIZE", "640"))
model = YOLO(model_name)

def _track(frame, classes=None):
    kwargs = {
        "persist": True,
        "tracker": "bytetrack.yaml",
        "verbose": False,
        "imgsz": inference_size,
        "device": device,
    }
    if classes is not None:
        kwargs["classes"] = classes
    return model.track(frame, **kwargs)

def detect_persons(frame):
    results = _track(frame, classes=[0])
    persons = []
    for r in results:
        if r.boxes.id is not None:
            for box, tid, conf in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.id.int().cpu().tolist(),
                r.boxes.conf.cpu().tolist()
            ):
                if conf > 0.4:
                    x1, y1, x2, y2 = map(int, box)
                    persons.append({"bbox": (x1, y1, x2, y2), "conf": conf, "id": tid})
    return persons

def detect_vehicles(frame):
    results = _track(frame, classes=[1, 2, 3, 5, 7])
    vehicles = []
    class_names = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    for r in results:
        if r.boxes.id is not None:
            for box, tid, conf, cls_id in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.id.int().cpu().tolist(),
                r.boxes.conf.cpu().tolist(),
                r.boxes.cls.int().cpu().tolist()
            ):
                if conf > 0.3:
                    x1, y1, x2, y2 = map(int, box)
                    vehicles.append({
                        "bbox": (x1, y1, x2, y2), "conf": conf, "id": tid,
                        "type": class_names.get(cls_id, "vehicle")
                    })
    return vehicles

def detect_all(frame):
    results = _track(frame)
    persons, vehicles, others = [], [], []
    class_names = model.names
    for r in results:
        if r.boxes.id is not None:
            for box, tid, conf, cls_id in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.id.int().cpu().tolist(),
                r.boxes.conf.cpu().tolist(),
                r.boxes.cls.int().cpu().tolist()
            ):
                x1, y1, x2, y2 = map(int, box)
                obj_type = class_names[cls_id]
                obj = {"bbox": (x1, y1, x2, y2), "conf": conf, "id": tid, "type": obj_type}
                if obj_type == "person" and conf > 0.4:
                    persons.append(obj)
                elif obj_type in ["car", "motorcycle", "bus", "truck", "bicycle"] and conf > 0.3:
                    vehicles.append(obj)
                elif conf > 0.25:
                    others.append(obj)
    return persons, vehicles, others
