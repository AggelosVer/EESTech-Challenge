# NuScenes Hackathon: The Ultimate Q&A and Pitfall Guide

Welcome to the hackathon! Working with real-world autonomous driving data like the nuScenes dataset is incredibly rewarding, but it comes with a few traps that trip up almost every beginner. 

Before you spend hours debugging, read through this guide. It covers the exact coordinate systems, evaluation constraints, and physics tricks you need to score highly on all three tasks.

---

## 📸 Task 1: 2D Object Detection

**Q: My model is finding cars perfectly, but the evaluation script is giving me a 0.0 IoU or a massive False Positive penalty. Why?**
**A:** There are two major pitfalls here:
1. **The Camera Trap:** The evaluation script **only grades `CAM_FRONT`**. NuScenes provides 6 cameras per sample. If your pipeline runs detection on `CAM_BACK` or `CAM_SIDE` and dumps those into the JSON, the evaluator will think they are False Positives in the front camera and nuke your score. Only submit predictions for `CAM_FRONT`.
2. **The Class Trap:** The evaluator only looks for vehicles. If your model correctly identifies a traffic cone, a pedestrian, or a bicycle, the script will look for a ground-truth vehicle to match it against, fail, and penalize you. **Filter your predictions!** Only output: `vehicle.car`, `vehicle.bus.bendy`, `vehicle.bus.rigid`, and `vehicle.truck`.

**Q: What if my model outputs multiple overlapping bounding boxes for the same car?**
**A:** You will be penalized! The evaluator uses a greedy matching loop. If two predicted boxes overlap the same ground-truth vehicle, the best one gets matched, but the second one falls through and is flagged as a False Positive (0.0 IoU). **Run Non-Maximum Suppression (NMS)** to remove duplicates before saving your JSON.

**Q: How do I clip my 2D bounding boxes properly?**
**A:** When you project 3D boxes to 2D using the camera intrinsics, the corners might fall outside the image frame. You must clip your `[xmin, ymin, xmax, ymax]` values to the nuScenes camera resolution: **1600 x 900**.

---

## 🧊 Task 2: 3D Object Detection

**Q: I calculated the 3D center perfectly, but the evaluation script says my error is 15+ meters. What's wrong?**
**A:** You fell into the **Coordinate System Trap**. The ground truth 3D boxes for this task are evaluated in the **Camera Sensor Coordinate Frame**, NOT the global map frame or a standard math frame.
* **X** points **Right**
* **Y** points **Down** (Yes, down!)
* **Z** points **Forward** (Into the image / depth)
If you assume Z is "Up" (which is common in other datasets), your math will be entirely inverted.

**Q: I'm spending hours trying to calculate the vehicle's yaw (rotation) using quaternions. Is there an easier way?**
**A:** Yes—stop calculating it! To save you time, the evaluation script **does not grade yaw**. It only evaluates your Mean Error for the **3D Center (X, Y, Z)** and your Mean Error for the **3D Size (Width, Length, Height)**. Save your hackathon hours for the math that gives you points.

**Q: How does the script match my 3D prediction to the right car?**
**A:** Your 3D prediction is evaluated *only if* the 2D bounding box attached to it achieves an IoU of > 0.35 with a ground truth box. Make sure your JSON format pairs the 2D box and 3D data correctly for each instance.

---

## 🏎️ Task 3: Trajectory Prediction

**Q: Are we supposed to use PyTorch or build a deep learning model to predict the trajectories?**
**A:** **No.** Due to resource and time constraints, deep learning is restricted for this task. You should use **Physics and Kinematics** (e.g., Constant Velocity models, Constant Acceleration, Kalman Filters, Bicycle models).

**Q: My trajectory predictions look right on my screen, but the script is giving me an ADE error in the thousands of meters. Why?**
**A:** The Coordinate System Trap strikes again, but this time in reverse! Unlike Task 2 (which uses the Camera frame), the ground truth trajectories in Task 3 are measured natively in the **Global Coordinate Frame** (the actual absolute map coordinates). You must explicitly output your trajectory predictions in global map coordinates, NOT relative to the camera or the ego-vehicle.

**Q: What is the time step between frames in NuScenes?**
**A:** NuScenes keyframes are exactly **0.5 seconds apart (2Hz)**. If you need to predict 6 seconds into the future, you must generate **12 coordinate pairs**. 

**Q: What if the vehicle drives out of the camera view or behind a building after 2 seconds? Do I still predict all 6 seconds?**
**A:** **Yes, always predict the full 12 frames.** The evaluation script dynamically checks the ground truth. If the car disappears after 4 frames, it will simply slice your prediction and only grade the first 4 frames. 
*Warning:* Do NOT predict fewer frames than 12! If you predict 4 frames but the ground truth actually has 8 frames available, the script will assume your vehicle "froze in place" for the remaining 4 frames and heavily penalize your ADE (Average Displacement Error) score.

---

## 🛠️ General Strategy

**Q: If I completely fail Task 1 (2D Detection), am I locked out of Task 2 and 3?**
**A:** Not at all! The tasks are designed to be decoupled. If your 2D object detector is completely broken, you can bypass it by extracting the 2D bounding boxes directly from the provided **Ground Truth dataset**. Feed those ground truth boxes into your Task 2 frustum generator, or use them to find targets for Task 3. A working downstream pipeline is better than a completely broken project!s