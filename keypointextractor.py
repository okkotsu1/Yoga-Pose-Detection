#!/usr/bin/env python3
"""
landmark_dataset.py
Preprocess all yoga images using Mediapipe Pose and save keypoints for faster training.
"""

import os
import cv2
import numpy as np
import mediapipe as mp
import concurrent.futures
import threading
import time
import joblib

# ---------------- CONFIG ----------------
DATA_DIR = r"C:\Users\shiva\Downloads\backup\backup\Work_backup_20250926_194233\dataset-yoga-final\images3"
OUTPUT_DIR = "processed_keypoints"
NUM_WORKERS = min(8, (os.cpu_count() or 4))

# ---------------- Mediapipe Thread Setup ----------------
mp_pose = mp.solutions.pose
thread_local = threading.local()

def get_thread_pose():
    if not hasattr(thread_local, "pose"):
        print("[INFO] Creating Mediapipe Pose instance for this thread...")
        thread_local.pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    return thread_local.pose

# ---------------- Helper Functions ----------------
def load_images_from_folder(folder):
    """Return all image file paths recursively under the folder."""
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(image_extensions):
                paths.append(os.path.join(root, f))
    return paths

def extract_keypoints_from_item(item):
    """Extract Mediapipe Pose keypoints for a single image."""
    img_path, label_str = item
    img = cv2.imread(img_path)
    if img is None:
        return None, label_str, img_path
    try:
        pose = get_thread_pose()
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
            return np.array(keypoints, dtype=np.float32), label_str, img_path
        else:
            return None, label_str, img_path
    except Exception as e:
        print(f"[ERROR] {img_path}: {e}")
        return None, label_str, img_path

# ---------------- Collect All Image Paths ----------------
print(f"[INFO] Scanning dataset: {DATA_DIR}")
items = []
for pose_name in os.listdir(DATA_DIR):
    pose_path = os.path.join(DATA_DIR, pose_name)
    if not os.path.isdir(pose_path):
        continue
    for status in os.listdir(pose_path):  # right/wrong
        status_path = os.path.join(pose_path, status)
        if not os.path.isdir(status_path):
            continue
        for step in os.listdir(status_path):
            step_path = os.path.join(status_path, step)
            if not os.path.isdir(step_path):
                continue
            image_paths = load_images_from_folder(step_path)
            label_str = f"{pose_name}{status}{step.replace(' ', '')}"
            for img_path in image_paths:
                items.append((img_path, label_str))

print(f"[INFO] Found {len(items)} images for processing.\n")

if len(items) == 0:
    raise ValueError("❌ No images found — check dataset path!")

# ---------------- Parallel Keypoint Extraction ----------------
X, labels_info = [], []
start_time = time.time()

print(f"[INFO] Starting keypoint extraction with {NUM_WORKERS} threads...")
with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    for i, (kp, label, img_path) in enumerate(executor.map(extract_keypoints_from_item, items), start=1):
        if kp is not None:
            X.append(kp)
            labels_info.append(label)
        if i % 100 == 0:
            print(f"[PROGRESS] {i}/{len(items)} processed | valid: {len(X)}")

elapsed = time.time() - start_time
print(f"\n✅ Extraction finished in {elapsed/60:.1f} min")
print(f"[SUMMARY] Valid: {len(X)}, Skipped: {len(items) - len(X)}")

# ---------------- Encode and Save ----------------
if len(X) == 0:
    raise RuntimeError("❌ No valid keypoints extracted — nothing to save.")

os.makedirs(OUTPUT_DIR, exist_ok=True)
X = np.array(X, dtype=np.float32)
labels_info = np.array(labels_info)

class_names = sorted(list(set(labels_info)))
y = np.array([class_names.index(lbl) for lbl in labels_info])

np.save(os.path.join(OUTPUT_DIR, "X_keypoints.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "y_labels.npy"), y)
joblib.dump(class_names, os.path.join(OUTPUT_DIR, "pose_classes.pkl"))

print(f"\n💾 Saved keypoints to: {OUTPUT_DIR}")
print(f" - X shape: {X.shape}")
print(f" - y shape: {y.shape}")
print(f" - Classes: {len(class_names)}")
print("✅ Dataset landmarking complete!")
