"""
eval_task3_future_only.py — Evaluate Task 3 on FUTURE frames only (skip history).

This evaluator skips the first 4 GT frames (the history window) and only
grades predictions on frames 4..N, where N is the total GT length.

Comparison:
  - Official evaluator: grades pred[0..11] vs gt[0..11]  (includes 4 "free" history frames)
  - This evaluator:     grades pred[4..11] vs gt[4..11]  (pure future prediction only)
"""

import json
import numpy as np
from nuscenes.nuscenes import NuScenes

DATAROOT = './student_dataset'
STUDENT_SUBMISSION = './predictions_task3.json'
HISTORY_FRAMES = 4  # number of history frames to skip

print("Loading Ground Truth Database...")
nusc = NuScenes(version='v1.0-eval', dataroot=DATAROOT, verbose=False)

with open(STUDENT_SUBMISSION, 'r') as f:
    preds = json.load(f)

ade_official = []     # all 12 frames (same as evaluation_greedy.py)
ade_future_only = []  # frames 4+ only (skip history)
ade_history_only = [] # frames 0-3 only (the "free" part)

print("\nEvaluating Task 3: Official vs Future-Only...\n")

n_instances = 0
n_short = 0  # instances with <= HISTORY_FRAMES annotations (no future to grade)

for instance_token, predicted_traj in preds.get('trajectories', {}).items():
    try:
        instance = nusc.get('instance', instance_token)
        current_ann_token = instance['first_annotation_token']
    except:
        continue

    # Build full GT (up to 16 frames so we can evaluate frames 4..15)
    gt_traj = []
    while current_ann_token != '' and len(gt_traj) < 16:
        ann = nusc.get('sample_annotation', current_ann_token)
        gt_traj.append([ann['translation'][0], ann['translation'][1]])
        current_ann_token = ann['next']

    if len(gt_traj) == 0:
        continue

    n_instances += 1
    gt_traj = np.array(gt_traj)
    p_traj = np.array(predicted_traj)

    # --- Official ADE (same logic as evaluation_greedy.py) ---
    gt_official = gt_traj[:12]
    p_official = p_traj[:len(gt_official)]
    if len(p_official) < len(gt_official):
        pad = np.tile(p_official[-1], (len(gt_official) - len(p_official), 1))
        p_official = np.vstack((p_official, pad))
    ade_off = np.mean(np.linalg.norm(p_official - gt_official, axis=1))
    ade_official.append(ade_off)

    # --- History-only ADE (frames 0..3) ---
    n_hist = min(HISTORY_FRAMES, len(gt_traj))
    if n_hist > 0 and len(p_traj) >= n_hist:
        hist_err = np.mean(np.linalg.norm(p_traj[:n_hist] - gt_traj[:n_hist], axis=1))
        ade_history_only.append(hist_err)

    # --- Future-only ADE (frames 4+) ---
    if len(gt_traj) <= HISTORY_FRAMES:
        n_short += 1
        continue  # no future frames to evaluate

    gt_future = gt_traj[HISTORY_FRAMES:HISTORY_FRAMES + 12]
    p_future = p_traj[HISTORY_FRAMES:HISTORY_FRAMES + len(gt_future)]
    if len(p_future) < len(gt_future):
        if len(p_future) > 0:
            pad = np.tile(p_future[-1], (len(gt_future) - len(p_future), 1))
            p_future = np.vstack((p_future, pad))
        else:
            p_future = np.zeros_like(gt_future)
    ade_fut = np.mean(np.linalg.norm(p_future - gt_future, axis=1))
    ade_future_only.append(ade_fut)

# Results
print("=" * 60)
print("TASK 3 — DETAILED ADE BREAKDOWN")
print("=" * 60)
print(f"Total instances evaluated:  {n_instances}")
print(f"Instances with no future:   {n_short} (≤{HISTORY_FRAMES} annotations)")
print()

avg_official = np.mean(ade_official) if ade_official else float('inf')
avg_history = np.mean(ade_history_only) if ade_history_only else float('inf')
avg_future = np.mean(ade_future_only) if ade_future_only else float('inf')

print(f"Official ADE (all frames):  {avg_official:.3f} m   ← same as evaluation_greedy.py")
print(f"History ADE (frames 0-3):   {avg_history:.3f} m   ← 'free' reconstruction error")
print(f"Future ADE  (frames 4+):    {avg_future:.3f} m   ← genuine prediction error")
print()
print(f"The history frames pull the official ADE down by ~{avg_future - avg_official:.3f} m")
print("=" * 60)
