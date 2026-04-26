import json
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points

# ==============================================================================
# SETUP - Point this to your INTACT evaluation dataset (with ground truth)
# ==============================================================================
DATAROOT = './student_dataset'
STUDENT_SUBMISSION = './predictions_task3.json'

print("Loading Ground Truth Database...")
nusc = NuScenes(version='v1.0-eval', dataroot=DATAROOT, verbose=False)

try:
    with open(STUDENT_SUBMISSION, 'r') as f:
        preds = json.load(f)
except Exception as e:
    print(f"FAILED TO LOAD PREDICTIONS. Tell students their JSON is broken: {e}")
    exit()

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def calculate_iou_2d(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

# ==============================================================================
# EVALUATION METRICS
# ==============================================================================
iou_2d_scores = []
error_3d_centers = []
error_3d_sizes = [] #Tracking size error
ade_scores = []

# Filter for vehicles only!
VEHICLE_CLASSES = ['vehicle.car', 'vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.truck']

print("\nEvaluating Task 1 & 2: 2D and 3D Detection (CAM_FRONT Only)...")
for sample_token, predicted_boxes in preds.get('detections', {}).items():
    
    # 1. Get Ground Truth for this sample
    try:
        sample = nusc.get('sample', sample_token)
        cam_token = sample['data']['CAM_FRONT']
        cam_data = nusc.get('sample_data', cam_token)
        cam_intrinsic = np.array(nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])['camera_intrinsic'])
        _, gt_boxes_3d, _ = nusc.get_sample_data(cam_token)
    except:
        continue

    # 2. Convert GT 3D boxes to 2D (Filtering for vehicles only)
    gt_2d_boxes = []
    for gt_box in gt_boxes_3d:
        if any(v_class in gt_box.name for v_class in VEHICLE_CLASSES):
            corners_2d = view_points(gt_box.corners(), cam_intrinsic, normalize=True)[:2, :]
            xmin, ymin = np.max([0, np.min(corners_2d[0])]), np.max([0, np.min(corners_2d[1])])
            xmax, ymax = np.min([1600, np.max(corners_2d[0])]), np.min([900, np.max(corners_2d[1])])
            # Appending gt_box.wlh (width, length, height)
            gt_2d_boxes.append((xmin, ymin, xmax, ymax, gt_box.center, gt_box.wlh))

    # 3. Greedy matching with strict False Positive/Negative penalties
    matched_gt_indices = set()
    predicted_boxes = sorted(predicted_boxes, key=lambda x: x.get('score', 0.0), reverse=True)

    for p_box in predicted_boxes:
        best_iou = 0
        best_gt_idx = -1
        best_gt_center = None
        best_gt_size = None
        p_2d = p_box['box_2d']
        
        for gt_idx, gt_2d in enumerate(gt_2d_boxes):
            if gt_idx in matched_gt_indices:
                continue 
                
            iou = calculate_iou_2d(p_2d, gt_2d[:4])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
                best_gt_center = gt_2d[4]
                best_gt_size = gt_2d[5] # Grab the size
                
        if best_iou > 0:
            # Valid match found
            matched_gt_indices.add(best_gt_idx)
            iou_2d_scores.append(best_iou)
            
            # Evaluate 3D Center and Size if 2D IoU is decent
            if best_iou > 0.35 and best_gt_center is not None:
                # Center Error
                p_3d_center = np.array(p_box['box_3d_center'])
                dist = np.linalg.norm(p_3d_center - best_gt_center)
                error_3d_centers.append(dist)

                # Size Error (Mean Absolute Error across W, L, H)
                if 'box_3d_size' in p_box:
                    p_3d_size = np.array(p_box['box_3d_size'])
                    size_err = np.mean(np.abs(p_3d_size - best_gt_size))
                    error_3d_sizes.append(size_err)
        else:
            # False Positive
            iou_2d_scores.append(0.0)
            
    # False Negative
    for gt_idx in range(len(gt_2d_boxes)):
        if gt_idx not in matched_gt_indices:
            iou_2d_scores.append(0.0)

print("Evaluating Task 3: Trajectory Prediction...")
for instance_token, predicted_traj in preds.get('trajectories', {}).items():
    try:
        instance = nusc.get('instance', instance_token)
        current_ann_token = instance['first_annotation_token']
    except:
        continue

    gt_traj = []
    while current_ann_token != '' and len(gt_traj) < 12:
        ann = nusc.get('sample_annotation', current_ann_token)
        gt_traj.append([ann['translation'][0], ann['translation'][1]])
        current_ann_token = ann['next']
        
    if len(gt_traj) > 0:
        gt_traj = np.array(gt_traj)
        p_traj = np.array(predicted_traj)
        
        #Slice prediction to match available GT length
        p_traj_sliced = p_traj[:len(gt_traj)]
        
        #Penalize if they predicted fewer frames than available GT
        if len(p_traj_sliced) < len(gt_traj):
            pad_len = len(gt_traj) - len(p_traj_sliced)
            if len(p_traj_sliced) > 0:
                # Pad with their last predicted position
                pad_pts = np.tile(p_traj_sliced[-1], (pad_len, 1))
                p_traj_sliced = np.vstack((p_traj_sliced, pad_pts))
            else:
                # Total failure fallback
                p_traj_sliced = np.zeros_like(gt_traj)

        ade = np.mean(np.linalg.norm(p_traj_sliced - gt_traj, axis=1))
        ade_scores.append(ade)

# ==============================================================================
# FINAL SCOREBOARD
# ==============================================================================
print("\n" + "="*55)
print("HACKATHON FINAL SCORES")
print("="*55)

avg_iou = np.mean(iou_2d_scores) if iou_2d_scores else 0
print(f"Task 1 (2D Detection)  | Mean IoU:          {avg_iou:.3f}  (Higher = Better, Max 1.0)")

avg_3d_center = np.mean(error_3d_centers) if error_3d_centers else float('inf')
print(f"Task 2 (3D Center)     | Mean Error:        {avg_3d_center:.3f} meters (Lower = Better)")

avg_3d_size = np.mean(error_3d_sizes) if error_3d_sizes else float('inf')
print(f"Task 2 (3D Size)       | Mean WLH Error:    {avg_3d_size:.3f} meters (Lower = Better)")

avg_ade = np.mean(ade_scores) if ade_scores else float('inf')
print(f"Task 3 (Trajectory)    | Mean ADE:          {avg_ade:.3f} meters (Lower = Better)")
print("="*55)