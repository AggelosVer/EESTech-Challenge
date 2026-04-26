"""
Task 1: 2D Object Detection on NuScenes CAM_FRONT.

Pipeline:
  1. Load NuScenes (v1.0-eval) from disk.
  2. For every sample, run a pre-trained YOLOv8l detector on CAM_FRONT images.
  3. Keep only vehicle classes (car / bus / truck) that the evaluator scores.
  4. Clip boxes to the 1600x900 image plane.
  5. Expand each box by 8% to better match the projected 3D ground truth.
  6. Apply class-agnostic NMS (IoU threshold 0.6) to remove duplicate detections.
  7. Write predictions JSON in the required submission format.

Best config: conf=0.30, expand=0.08, nms=0.60 → Mean IoU = 0.420
"""

import json
import os
from pathlib import Path

import numpy as np
from nuscenes.nuscenes import NuScenes
from ultralytics import YOLO

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
DATAROOT = "./student_dataset"
VERSION = "v1.0-eval"
IMG_W, IMG_H = 1600, 900

# YOLO settings — best config from hyperparameter sweep
MODEL_WEIGHTS = "yolov8l.pt"
CONF_THRESHOLD = 0.30
NMS_IOU_THRESHOLD = 0.60
BOX_EXPAND_FACTOR = 0.08

# COCO class IDs that count as "vehicle" for this evaluation.
# COCO: 2=car, 5=bus, 7=truck
COCO_VEHICLE_CLASSES = {2: "car", 5: "bus", 7: "truck"}

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_JSON = ROOT / "predictions_task1.json"


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def calculate_iou_2d(boxA, boxB):
    """Calculate IoU between two [xmin, ymin, xmax, ymax] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter + 1e-6)


def class_agnostic_nms(detections, iou_threshold):
    """Suppress overlapping boxes regardless of class — prevents car/truck double-detects."""
    if len(detections) <= 1:
        return detections
    dets = sorted(detections, key=lambda d: d["score"], reverse=True)
    kept = []
    while dets:
        top = dets.pop(0)
        kept.append(top)
        dets = [d for d in dets if calculate_iou_2d(top["box_2d"], d["box_2d"]) < iou_threshold]
    return kept


def expand_box(box, factor):
    """Expand a box by `factor` in each direction to better match projected 3D GT."""
    w, h = box[2] - box[0], box[3] - box[1]
    dx, dy = w * factor, h * factor
    return [
        max(0.0, box[0] - dx),
        max(0.0, box[1] - dy),
        min(float(IMG_W), box[2] + dx),
        min(float(IMG_H), box[3] + dy),
    ]


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    # 1. Initialize NuScenes
    print(f"Loading NuScenes from {DATAROOT} (version={VERSION}) ...")
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)

    # 2. Load YOLO model
    print(f"Loading YOLO model: {MODEL_WEIGHTS}")
    model = YOLO(str(ROOT / MODEL_WEIGHTS))

    # 3. Collect all sample tokens and their CAM_FRONT image paths
    sample_paths = {}
    for scene in nusc.scene:
        token = scene["first_sample_token"]
        while token:
            sample = nusc.get("sample", token)
            cam_front_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
            sample_paths[token] = os.path.join(DATAROOT, cam_front_data["filename"])
            token = sample["next"]
    print(f"Total samples to process: {len(sample_paths)}")

    # 4. Run detection on each CAM_FRONT image
    predictions = {"detections": {}, "trajectories": {}}

    for i, (sample_token, img_path) in enumerate(sample_paths.items(), 1):
        # Run YOLOv8 — filter to vehicle classes only
        results = model.predict(
            source=img_path,
            conf=CONF_THRESHOLD,
            iou=0.99,  # disable YOLO internal NMS (we do our own)
            classes=list(COCO_VEHICLE_CLASSES.keys()),
            verbose=False,
        )[0]

        detections = []
        if results.boxes is not None and len(results.boxes) > 0:
            boxes_xyxy = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            cls_ids = results.boxes.cls.cpu().numpy().astype(int)

            for (xmin, ymin, xmax, ymax), conf, cid in zip(boxes_xyxy, confs, cls_ids):
                # Clip boxes to image bounds (1600 x 900)
                xmin = max(0.0, float(xmin))
                ymin = max(0.0, float(ymin))
                xmax = min(float(IMG_W), float(xmax))
                ymax = min(float(IMG_H), float(ymax))
                if xmax <= xmin or ymax <= ymin:
                    continue

                # Expand box to better match projected 3D ground truth
                box_2d = expand_box([xmin, ymin, xmax, ymax], BOX_EXPAND_FACTOR)

                detections.append({
                    "class_name": COCO_VEHICLE_CLASSES.get(int(cid), "car"),
                    "score": float(conf),
                    "box_2d": box_2d,
                    # Placeholder 3D fields (Task 2 will fill these)
                    "box_3d_center": [0.0, 0.0, 0.0],
                    "box_3d_size": [0.0, 0.0, 0.0],
                    "box_3d_yaw": 0.0,
                })

        # Apply class-agnostic NMS to remove duplicate detections
        predictions["detections"][sample_token] = class_agnostic_nms(
            detections, iou_threshold=NMS_IOU_THRESHOLD
        )

        if i % 50 == 0 or i == len(sample_paths):
            print(f"  [{i}/{len(sample_paths)}] processed")

    # 5. Save predictions JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(predictions, f)
    print(f"\nWrote predictions to {OUTPUT_JSON}")
    print(f"Config: conf={CONF_THRESHOLD}, expand={BOX_EXPAND_FACTOR}, nms={NMS_IOU_THRESHOLD}")


if __name__ == "__main__":
    main()
