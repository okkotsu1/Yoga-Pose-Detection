#!/usr/bin/env python3
"""
knn_pose_tuning.py
Test multiple KNN configurations (n_neighbors, weights, metric) on extracted yoga pose features.
"""

import os
import cv2
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import concurrent.futures
import threading
import mediapipe as mp

# ---------------- Config ----------------
DATA_DIR = r"C:\Users\shiva\Downloads\backup\backup\Work_backup_20250926_194233\dataset-yoga-final\images7"
NUM_WORKERS = 12
CACHE_FILE = "pose_features_knn.pkl"

# ---------------- Mediapipe Setup ----------------
mp_pose = mp.solutions.pose
thread_local = threading.local()
def get_thread_pose():
    if not hasattr(thread_local, "pose"):
        thread_local.pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    return thread_local.pose

# ---------------- Feature extraction (same as before) ----------------
def calculate_angle(a, b, c):
    a,b,c = np.array(a), np.array(b), np.array(c)
    ab = a - b
    cb = c - b
    denom = (np.linalg.norm(ab) * np.linalg.norm(cb))
    if denom == 0: return 0.0
    cosang = np.clip(np.dot(ab, cb) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def extract_angles(landmarks):
    def p(i): return [landmarks[i].x, landmarks[i].y]
    angles = []
    try:
        mpL = mp_pose.PoseLandmark
        angles = [
            calculate_angle(p(mpL.LEFT_SHOULDER.value), p(mpL.LEFT_ELBOW.value), p(mpL.LEFT_WRIST.value)),
            calculate_angle(p(mpL.RIGHT_SHOULDER.value), p(mpL.RIGHT_ELBOW.value), p(mpL.RIGHT_WRIST.value)),
            calculate_angle(p(mpL.LEFT_HIP.value), p(mpL.LEFT_KNEE.value), p(mpL.LEFT_ANKLE.value)),
            calculate_angle(p(mpL.RIGHT_HIP.value), p(mpL.RIGHT_KNEE.value), p(mpL.RIGHT_ANKLE.value)),
            calculate_angle(p(mpL.LEFT_ELBOW.value), p(mpL.LEFT_SHOULDER.value), p(mpL.LEFT_HIP.value)),
            calculate_angle(p(mpL.RIGHT_ELBOW.value), p(mpL.RIGHT_SHOULDER.value), p(mpL.RIGHT_HIP.value)),
            calculate_angle(p(mpL.LEFT_SHOULDER.value), p(mpL.LEFT_HIP.value), p(mpL.LEFT_KNEE.value)),
            calculate_angle(p(mpL.RIGHT_SHOULDER.value), p(mpL.RIGHT_HIP.value), p(mpL.RIGHT_KNEE.value)),
            calculate_angle(p(mpL.LEFT_HIP.value), p(mpL.RIGHT_HIP.value), p(mpL.RIGHT_KNEE.value)),
            calculate_angle(p(mpL.LEFT_SHOULDER.value), p(mpL.RIGHT_SHOULDER.value), p(mpL.RIGHT_ELBOW.value)),
            calculate_angle(p(mpL.LEFT_KNEE.value), p(mpL.LEFT_ANKLE.value), p(mpL.RIGHT_ANKLE.value)),
            calculate_angle(p(mpL.RIGHT_KNEE.value), p(mpL.RIGHT_ANKLE.value), p(mpL.LEFT_ANKLE.value))
        ]
    except Exception:
        angles = [0.0]*12
    return np.array(angles, dtype=np.float32)

def normalize_landmarks_np(landmarks):
    arr = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks], dtype=np.float32)
    l_hip = arr[mp_pose.PoseLandmark.LEFT_HIP.value][:3]
    r_hip = arr[mp_pose.PoseLandmark.RIGHT_HIP.value][:3]
    mid_hip = (l_hip + r_hip) / 2.0
    arr[:, :3] -= mid_hip
    y_coords = arr[:,1]
    y_span = y_coords.max() - y_coords.min()
    l_sh = arr[mp_pose.PoseLandmark.LEFT_SHOULDER.value][:3]
    r_sh = arr[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][:3]
    sh_hip_dist = np.linalg.norm(((l_sh + r_sh)/2.0) - mid_hip)
    scale = max(y_span, sh_hip_dist, 1e-6)
    arr[:, :3] /= scale
    return arr.flatten()

def extract_features(item):
    img_path, label_str = item
    img = cv2.imread(img_path)
    if img is None:
        return None, label_str
    try:
        pose = get_thread_pose()
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return None, label_str
        nodes = normalize_landmarks_np(results.pose_landmarks.landmark)
        angles = extract_angles(results.pose_landmarks.landmark)
        feat = np.concatenate([nodes, angles])
        return feat, label_str
    except Exception:
        return None, label_str

def load_image_items(data_dir):
    items = []
    for pose_name in os.listdir(data_dir):
        pose_path = os.path.join(data_dir, pose_name)
        if not os.path.isdir(pose_path):
            continue
        for status in os.listdir(pose_path):
            status_path = os.path.join(pose_path, status)
            if not os.path.isdir(status_path):
                continue
            for step in os.listdir(status_path):
                step_path = os.path.join(status_path, step)
                if not os.path.isdir(step_path):
                    continue
                for root, _, files in os.walk(step_path):
                    for f in files:
                        if f.lower().endswith((".jpg",".jpeg",".png",".bmp")):
                            label_str = f"{pose_name}_{status}_{step.replace(' ','')}"
                            items.append((os.path.join(root,f), label_str))
    return items

def extract_all_features(items, cache_file=None):
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached features from {cache_file}")
        data = joblib.load(cache_file)
        return data["X"], data["y"], data["class_names"]

    feats, labels = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as exc:
        for feat, label in exc.map(extract_features, items):
            if feat is not None:
                feats.append(feat)
                labels.append(label)

    class_names = sorted(list(set(labels)))
    name_to_idx = {n:i for i,n in enumerate(class_names)}
    y = np.array([name_to_idx[l] for l in labels])
    X = np.vstack(feats)

    if cache_file:
        joblib.dump({"X": X, "y": y, "class_names": class_names}, cache_file)
    return X, y, class_names

# ---------------- Hyperparameter Tuning ----------------
def main():
    print("Loading features...")
    items = load_image_items(DATA_DIR)
    X, y, class_names = extract_all_features(items, cache_file=CACHE_FILE)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # define parameter grid
    neighbor_values = [3, 5, 7, 9]
    weight_types = ['uniform', 'distance']
    metrics = ['euclidean', 'manhattan', 'minkowski']

    results = []

    for n in neighbor_values:
        for w in weight_types:
            for m in metrics:
                print(f"\n=== K={n}, weights={w}, metric={m} ===")
                knn = KNeighborsClassifier(n_neighbors=n, weights=w, metric=m)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f}")
                results.append({
                    "n_neighbors": n,
                    "weights": w,
                    "metric": m,
                    "accuracy": acc,
                    "precision": prec
                })

    df = pd.DataFrame(results)
    df.to_csv("knn_precision_comparison.csv", index=False)
    print("\nSaved comparison -> knn_precision_comparison.csv")
    print(df)

    # Plot precision comparison
    plt.figure(figsize=(10,6))
    sns.barplot(x="n_neighbors", y="precision", hue="weights", data=df)
    plt.title("KNN Precision vs Neighbors (grouped by weights)")
    plt.tight_layout()
    plt.savefig("knn_precision_barplot.png")
    plt.close()
    print("Saved precision comparison plot -> knn_precision_barplot.png")

if __name__ == "__main__":
    main()
