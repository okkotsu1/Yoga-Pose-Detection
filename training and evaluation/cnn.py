#!/usr/bin/env python3
"""
cnn_pose_tuning_torch.py
CNN yoga pose classifier using PyTorch (Conv1D on pose features + angles)
"""

import os
import cv2
import joblib
import numpy as np
import pandas as pd
import concurrent.futures
import threading
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score
import mediapipe as mp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ---------------- Config ----------------
DATA_DIR = r"C:\Users\shiva\Downloads\backup\backup\Work_backup_20250926_194233\dataset-yoga-final\images6"
NUM_WORKERS = 12
CACHE_FILE = "pose_features_cnn_torch.pkl"
EPOCHS = 20
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Mediapipe Setup ----------------
mp_pose = mp.solutions.pose
thread_local = threading.local()
def get_thread_pose():
    if not hasattr(thread_local, "pose"):
        thread_local.pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    return thread_local.pose

# ---------------- Feature Extraction ----------------
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
    mpL = mp_pose.PoseLandmark
    try:
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

# ---------------- CNN Model ----------------
class CNN1D(nn.Module):
    def __init__(self, input_len, num_classes, config):
        super().__init__()
        self.conv1 = nn.Conv1d(1, config['filters'], config['kernel_size'])
        self.bn1 = nn.BatchNorm1d(config['filters'])
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = None
        if config.get('conv2_filters'):
            self.conv2 = nn.Conv1d(config['filters'], config['conv2_filters'], config['conv2_kernel'])
            self.bn2 = nn.BatchNorm1d(config['conv2_filters'])
            self.pool2 = nn.MaxPool1d(2)
            final_len = ((input_len - (config['kernel_size']-1))//2 - (config['conv2_kernel']-1))//2
            fc_input = final_len * config['conv2_filters']
        else:
            final_len = (input_len - (config['kernel_size']-1))//2
            fc_input = final_len * config['filters']
        self.fc1 = nn.Linear(fc_input, config['dense_units'])
        self.dropout = nn.Dropout(config.get('dropout',0.3))
        self.fc2 = nn.Linear(config['dense_units'], num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        if self.conv2:
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.pool2(x)
        x = x.flatten(start_dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# ---------------- Training Helper ----------------
def train_model(model, train_loader, val_loader, epochs=EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    model.to(DEVICE)
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
    # Evaluation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            preds = out.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(yb.cpu().numpy())
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    return acc, prec

# ---------------- Main ----------------
def main():
    print("Loading features...")
    items = load_image_items(DATA_DIR)
    X, y, class_names = extract_all_features(items, cache_file=CACHE_FILE)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = X[:, :, np.newaxis]  # shape = (samples, features, 1) for Conv1D

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32).permute(0,2,1),
                                  torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32).permute(0,2,1),
                                 torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    cnn_configs = [
        {'filters': 32, 'kernel_size': 3, 'dense_units': 64, 'dropout':0.3},
        {'filters': 64, 'kernel_size': 5, 'dense_units': 128, 'dropout':0.4},
        {'filters': 128, 'kernel_size': 3, 'dense_units': 256, 'dropout':0.5},
    ]

    results = []

    for cfg in cnn_configs:
        print(f"\nTraining CNN with config: {cfg}")
        model = CNN1D(X_train.shape[1], len(class_names), cfg)
        acc, prec = train_model(model, train_loader, test_loader)
        print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f}")
        results.append({"config": str(cfg), "accuracy": acc, "precision": prec})

    df = pd.DataFrame(results)
    df.to_csv("cnn_precision_comparison_torch.csv", index=False)
    print("\nSaved comparison -> cnn_precision_comparison_torch.csv")
    print(df)

    plt.figure(figsize=(10,6))
    sns.barplot(x="config", y="precision", data=df)
    plt.title("CNN Precision Comparison (PyTorch)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("cnn_precision_barplot_torch.png")
    plt.close()
    print("Saved precision comparison plot -> cnn_precision_barplot_torch.png")

if __name__ == "__main__":
    main()
