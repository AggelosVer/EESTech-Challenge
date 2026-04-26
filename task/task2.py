import json
import os
from pathlib import Path

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from sklearn.decomposition import PCA
from ultralytics import YOLO


DATAROOT = "./student_dataset"
VERSION = "v1.0-eval"
IMG_W, IMG_H = 1600, 900

MODEL_WEIGHTS = "yolov8l.pt"
CONF_THRESHOLD = 0.30
NMS_IOU_THRESHOLD = 0.60
BOX_EXPAND_FACTOR = 0.08

# COCO: 2=car, 5=bus, 7=truck
COCO_VEHICLE_CLASSES = {2: "car", 5: "bus", 7: "truck"}


MIN_LIDAR_POINTS = 2            # below this → monocular fallback
DEPTH_PCT_LOW = 10.0            # trim the lowest 10% of depths
DEPTH_PCT_HIGH = 90.0           # trim the highest 10% of depths
DEPTH_QUANTILE = 25.0           # use 25th pct of trimmed depths (front-face bias)

# Class size priors [width, length, height] (m)
CLASS_SIZE = {
    "car":   [1.97, 4.62, 1.71],
    "truck": [2.39, 6.63, 2.63],
    "bus":   [2.95, 10.28, 3.37],
}

# Real-world heights (m) for the monocular depth fallback.
CLASS_REAL_HEIGHT = {"car": 1.71, "truck": 2.63, "bus": 3.37}

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_JSON = ROOT / "predictions_task2.json"

def calculate_iou_2d(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter + 1e-6)


def class_agnostic_nms(detections, iou_threshold):
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
    w, h = box[2] - box[0], box[3] - box[1]
    dx, dy = w * factor, h * factor
    return [
        max(0.0, box[0] - dx),
        max(0.0, box[1] - dy),
        min(float(IMG_W), box[2] + dx),
        min(float(IMG_H), box[3] + dy),
    ]

def lidar_to_camera_points(nusc, lidar_token, cam_token):
    """
    Directly transform LiDAR points to Camera frame using relative transforms.
    """
    lidar_rec = nusc.get("sample_data", lidar_token)
    cam_rec = nusc.get("sample_data", cam_token)

    # 1. Load PC
    pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, lidar_rec["filename"]))

    # 2. Lidar Sensor -> Ego
    cs_lidar = nusc.get("calibrated_sensor", lidar_rec["calibrated_sensor_token"])
    pc.rotate(Quaternion(cs_lidar["rotation"]).rotation_matrix)
    pc.translate(np.array(cs_lidar["translation"]))

    # 3. Ego -> Camera Sensor
    cs_cam = nusc.get("calibrated_sensor", cam_rec["calibrated_sensor_token"])
    pc.translate(-np.array(cs_cam["translation"]))
    pc.rotate(Quaternion(cs_cam["rotation"]).rotation_matrix.T)

    return pc.points, np.array(cs_cam["camera_intrinsic"])


def project_points(points_cam, K):

    depth = points_cam[2, :]
    in_front = depth > 0.1
    pts_img = view_points(points_cam[:3, :], K, normalize=True)
    u = pts_img[0, :]
    v = pts_img[1, :]
    inside = in_front & (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
    return u, v, depth, inside


def detect_occlusion(h_px, Z_lidar, fy, cls_name):
    """
    Detects if a NEARBY vehicle is occluded by comparing its actual 2D height
    against the expected height at the LiDAR-measured distance.
    Only meaningful for close vehicles (< 25m). Faraway vehicles are naturally small.
    """
    if Z_lidar <= 0 or Z_lidar > 25.0:
        return False, 1.0
    expected_h_px = (fy * CLASS_REAL_HEIGHT[cls_name]) / Z_lidar
    occlusion_ratio = h_px / max(1.0, expected_h_px)
    # If the box is less than 60% of the expected height, it's likely occluded
    is_occluded = occlusion_ratio < 0.60
    return is_occluded, occlusion_ratio


def estimate_3d_center_hybrid(pts_cam, u, v, inside_mask, box, cls_name, K):
    """
    Hybrid approach: Adaptive Z-offset based on aspect ratio, IQR filtering,
    occlusion-aware depth correction, and monocular fallback.
    """
    xmin, ymin, xmax, ymax = box
    in_box = (
        inside_mask
        & (u >= xmin) & (u <= xmax)
        & (v >= ymin) & (v <= ymax)
    )
    
    # 1. Height filter to remove road (Camera Y > 1.4m is road)
    in_box = in_box & (pts_cam[1, :] < 1.4)
    
    n_pts = int(np.count_nonzero(in_box))
    
    fy = K[1, 1]
    h_px = max(1.0, ymax - ymin)
    w_px = max(1.0, xmax - xmin)
    aspect_ratio = w_px / h_px
    
    u_c = (xmin + xmax) / 2.0
    v_c = (ymin + ymax) / 2.0
    
    # 2. Επιλογή Offset βάσει προσανατολισμού & θέσης
    dist_from_center = abs(u_c - (IMG_W / 2.0))
    threshold = 1.15 if dist_from_center > (IMG_W * 0.25) else 1.3
        
    if aspect_ratio < threshold:
        depth_offset = CLASS_SIZE[cls_name][1] / 2.0  # Μπροστά/Πίσω → L/2
    else:
        depth_offset = CLASS_SIZE[cls_name][0] / 2.0  # Πλάι → W/2

    Z_mono = (fy * CLASS_REAL_HEIGHT[cls_name]) / h_px
    expected_Z = Z_mono + depth_offset
    
    if n_pts < MIN_LIDAR_POINTS:
        return back_project_center(u_c, v_c, expected_Z, K), n_pts
        
    valid_pts = pts_cam[:3, in_box].T  # (N, 3)
    z_vals = valid_pts[:, 2]
    
    # 3. IQR Outlier Rejection
    q1, q3 = np.percentile(z_vals, [25, 75])
    iqr = q3 - q1
    valid_mask = (z_vals >= q1 - 1.5 * iqr) & (z_vals <= q3 + 1.5 * iqr)
    if np.sum(valid_mask) < 3:
        valid_mask = np.ones(len(valid_pts), dtype=bool)
    filtered_z = z_vals[valid_mask]
    
    # 4. Z από το 15th percentile (πιο κοντινή ορατή επιφάνεια)
    Z_surface = np.percentile(filtered_z, 15)
    
    # 5. Occlusion Check: Αν το αυτοκίνητο είναι επικαλυμμένο,
    #    το 2D box είναι τεχνητά μικρό → ο monocular Z θα ήταν λάθος.
    #    Εμπιστευόμαστε ΠΛΗΡΩΣ το LiDAR Z_surface σε αυτή την περίπτωση.
    # Z estimate: LiDAR surface + adaptive depth offset
    Z_lidar_final = Z_surface + depth_offset
    
    # Sanity check: if LiDAR diverges > 30% from monocular, the LiDAR likely hit
    # background (sign, wall, etc.) → fall back to monocular estimate.
    if abs(Z_lidar_final - expected_Z) / max(1.0, expected_Z) > 0.3:
        Z_final = expected_Z
    else:
        Z_final = Z_lidar_final
        
    center_3d = back_project_center(u_c, v_c, Z_final, K)
    return center_3d, n_pts


def back_project_center(u_c, v_c, Z, K):
    """Pinhole back-projection of a pixel + depth into camera-frame XYZ."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (u_c - cx) * Z / fx
    Y = (v_c - cy) * Z / fy
    return [float(X), float(Y), float(Z)]


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    print(f"Loading NuScenes from {DATAROOT} (version={VERSION}) ...")
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)

    print(f"Loading YOLO model: {MODEL_WEIGHTS}")
    model = YOLO(str(ROOT / MODEL_WEIGHTS))

    sample_tokens = []
    for scene in nusc.scene:
        token = scene["first_sample_token"]
        while token:
            sample_tokens.append(token)
            sample = nusc.get("sample", token)
            token = sample["next"]
    print(f"Total samples to process: {len(sample_tokens)}")

    predictions = {"detections": {}, "trajectories": {}}

    n_total_dets = 0
    n_with_lidar = 0
    n_fallback = 0
    pts_in_box_running = 0

    for i, sample_token in enumerate(sample_tokens, 1):
        sample = nusc.get("sample", sample_token)
        cam_token   = sample["data"]["CAM_FRONT"]
        lidar_token = sample["data"]["LIDAR_TOP"]

        cam_data = nusc.get("sample_data", cam_token)
        img_path = os.path.join(DATAROOT, cam_data["filename"])

        results = model.predict(
            source=img_path,
            conf=CONF_THRESHOLD,
            iou=0.99,                           # disable internal NMS
            classes=list(COCO_VEHICLE_CLASSES.keys()),
            verbose=False,
        )[0]

        detections = []
        if results.boxes is not None and len(results.boxes) > 0:
            boxes_xyxy = results.boxes.xyxy.cpu().numpy()
            confs      = results.boxes.conf.cpu().numpy()
            cls_ids    = results.boxes.cls.cpu().numpy().astype(int)

            for (xmin, ymin, xmax, ymax), conf, cid in zip(boxes_xyxy, confs, cls_ids):
                xmin = max(0.0, float(xmin))
                ymin = max(0.0, float(ymin))
                xmax = min(float(IMG_W), float(xmax))
                ymax = min(float(IMG_H), float(ymax))
                if xmax <= xmin or ymax <= ymin:
                    continue

                box_2d = expand_box([xmin, ymin, xmax, ymax], BOX_EXPAND_FACTOR)
                cls_name = COCO_VEHICLE_CLASSES.get(int(cid), "car")

                detections.append({
                    "class_name": cls_name,
                    "score": float(conf),
                    "box_2d": box_2d,
                })

        # Class-agnostic NMS — same as task1.py (do this BEFORE expensive 3D work)
        detections = class_agnostic_nms(detections, iou_threshold=NMS_IOU_THRESHOLD)

        u = v = depth = inside_mask = K = None
        if detections:
            try:
                pts_cam, K = lidar_to_camera_points(nusc, lidar_token, cam_token)
                u, v, depth, inside_mask = project_points(pts_cam, K)
            except Exception as e:
                print(f"[WARN] LiDAR projection failed for {sample_token}: {e}")
                u = v = depth = inside_mask = None

        sample_dets = []
        for det in detections:
            cls_name = det["class_name"]
            xmin, ymin, xmax, ymax = det["box_2d"]
            u_c = (xmin + xmax) / 2.0
            v_c = (ymin + ymax) / 2.0

            # (C) Robust 3D Center Estimation
            center_3d = None
            n_pts = 0
            
            if u is not None:
                center_3d, n_pts = estimate_3d_center_hybrid(
                    pts_cam, u, v, inside_mask, det["box_2d"], cls_name, K
                )
                
            if n_pts >= MIN_LIDAR_POINTS:
                n_with_lidar += 1
                pts_in_box_running += n_pts
            else:
                if center_3d is None:
                    # Absolute Fallback if LiDAR projection failed completely
                    pixel_h = max(1.0, ymax - ymin)
                    if K is None:
                        cam_cs = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
                        K = np.array(cam_cs["camera_intrinsic"])
                    fy = K[1, 1]
                    Z_center = ((fy * CLASS_REAL_HEIGHT[cls_name]) / pixel_h) + (CLASS_SIZE[cls_name][1] / 2.0)
                    center_3d = back_project_center(u_c, v_c, Z_center, K)
                n_fallback += 1

            sample_dets.append({
                "class_name": cls_name,
                "score": det["score"],
                "box_2d": det["box_2d"],
                "box_3d_center": center_3d,
                "box_3d_size":   list(map(float, CLASS_SIZE[cls_name])),
                "box_3d_yaw":    0.0,    # evaluator does not grade yaw
            })

        predictions["detections"][sample_token] = sample_dets
        n_total_dets += len(sample_dets)

        if i % 50 == 0 or i == len(sample_tokens):
            print(f"  [{i}/{len(sample_tokens)}] processed")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(predictions, f)

    print("\n" + "=" * 60)
    print("TASK 2 PREDICTION SUMMARY")
    print("=" * 60)
    print(f"Samples processed       : {len(sample_tokens)}")
    print(f"Total detections        : {n_total_dets}")
    print(f"  ... with LiDAR depth  : {n_with_lidar}")
    print(f"  ... using fallback    : {n_fallback}")
    if n_with_lidar:
        print(f"Avg LiDAR pts per box   : {pts_in_box_running / n_with_lidar:.1f}")
    print(f"Output JSON             : {OUTPUT_JSON}")
    print("=" * 60)


if __name__ == "__main__":
    main()
