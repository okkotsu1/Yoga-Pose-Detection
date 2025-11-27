#!/usr/bin/env python3
"""
gnn_pose_pipeline.py
End-to-end: extract normalized Mediapipe keypoints -> create PyG graphs -> train GNN.
Copy & paste and run. Adjust DATA_DIR and paths as needed.
"""

import os
import cv2
import math
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

import mediapipe as mp
import concurrent.futures
import threading

# ---------------- Config ----------------
DATA_DIR = r"C:\Users\shiva\Downloads\backup\backup\Work_backup_20250926_194233\dataset-yoga-final\images6"
NUM_WORKERS = 12
BATCH_SIZE = 32
EPOCHS = 80
LR = 1e-3
PATIENCE = 10   # early stopping
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_MODEL = "pose_gcn_best.pt"
CLASS_NAMES_FILE = "pose_classes.pkl"
GRAPH_DATA_FILE = "pose_graphs_changed_image6_joblib.pkl"  # optional cache file to save extracted graphs

# ---------------- Mediapipe Setup ----------------
mp_pose = mp.solutions.pose
thread_local = threading.local()
def get_thread_pose():
    if not hasattr(thread_local, "pose"):
        thread_local.pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    return thread_local.pose

# ---------------- Pose graph edges ----------------
POSE_EDGES = [
    (11,13),(13,15),
    (12,14),(14,16),
    (11,12),
    (23,24),
    (11,23),(12,24),
    (23,25),(25,27),
    (24,26),(26,28),
    (27,31),(28,32),
    (15,19),(16,20)
]
edge_index = torch.tensor(POSE_EDGES + [(b,a) for (a,b) in POSE_EDGES], dtype=torch.long).t().contiguous()

# ---------------- Utilities ----------------
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
        angles.append(calculate_angle(p(mp_pose.PoseLandmark.LEFT_SHOULDER.value),
                                      p(mp_pose.PoseLandmark.LEFT_ELBOW.value),
                                      p(mp_pose.PoseLandmark.LEFT_WRIST.value)))
        angles.append(calculate_angle(p(mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
                                      p(mp_pose.PoseLandmark.RIGHT_ELBOW.value),
                                      p(mp_pose.PoseLandmark.RIGHT_WRIST.value)))
        angles.append(calculate_angle(p(mp_pose.PoseLandmark.LEFT_HIP.value),
                                      p(mp_pose.PoseLandmark.LEFT_KNEE.value),
                                      p(mp_pose.PoseLandmark.LEFT_ANKLE.value)))
        angles.append(calculate_angle(p(mp_pose.PoseLandmark.RIGHT_HIP.value),
                                      p(mp_pose.PoseLandmark.RIGHT_KNEE.value),
                                      p(mp_pose.PoseLandmark.RIGHT_ANKLE.value)))
        angles.append(calculate_angle(p(mp_pose.PoseLandmark.LEFT_ELBOW.value),
                                      p(mp_pose.PoseLandmark.LEFT_SHOULDER.value),
                                      p(mp_pose.PoseLandmark.LEFT_HIP.value)))
        angles.append(calculate_angle(p(mp_pose.PoseLandmark.RIGHT_ELBOW.value),
                                      p(mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
                                      p(mp_pose.PoseLandmark.RIGHT_HIP.value)))
        angles.append(calculate_angle(p(mp_pose.PoseLandmark.LEFT_SHOULDER.value),
                                      p(mp_pose.PoseLandmark.LEFT_HIP.value),
                                      p(mp_pose.PoseLandmark.LEFT_KNEE.value)))
        angles.append(calculate_angle(p(mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
                                      p(mp_pose.PoseLandmark.RIGHT_HIP.value),
                                      p(mp_pose.PoseLandmark.RIGHT_KNEE.value)))
        angles.append(calculate_angle(p(mp_pose.PoseLandmark.LEFT_HIP.value),
                                      p(mp_pose.PoseLandmark.RIGHT_HIP.value),
                                      p(mp_pose.PoseLandmark.RIGHT_KNEE.value)))
        angles.append(calculate_angle(p(mp_pose.PoseLandmark.LEFT_SHOULDER.value),
                                      p(mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
                                      p(mp_pose.PoseLandmark.RIGHT_ELBOW.value)))
        angles.append(calculate_angle(p(mp_pose.PoseLandmark.LEFT_KNEE.value),
                                      p(mp_pose.PoseLandmark.LEFT_ANKLE.value),
                                      p(mp_pose.PoseLandmark.RIGHT_ANKLE.value)))
        angles.append(calculate_angle(p(mp_pose.PoseLandmark.RIGHT_KNEE.value),
                                      p(mp_pose.PoseLandmark.RIGHT_ANKLE.value),
                                      p(mp_pose.PoseLandmark.LEFT_ANKLE.value)))
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
    return arr  # (33,4)

# ---------------- Item -> Graph ----------------
def image_item_to_graph(item):
    img_path, label_str = item
    img = cv2.imread(img_path)
    if img is None:
        return None, label_str, img_path
    try:
        pose = get_thread_pose()
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return None, label_str, img_path
        nodes_np = normalize_landmarks_np(results.pose_landmarks.landmark)
        node_feats = torch.tensor(nodes_np[:, :4], dtype=torch.float)
        angles = extract_angles(results.pose_landmarks.landmark)
        angles_tensor = torch.tensor(angles, dtype=torch.float).unsqueeze(0)
        graph = Data(x=node_feats, edge_index=edge_index, y=torch.tensor([-1], dtype=torch.long))
        graph.angles = angles_tensor
        graph.meta_label = label_str
        graph.img_path = img_path
        return graph, label_str, img_path
    except Exception as e:
        print(f"[WARN] failed to process {img_path}: {e}")
        return None, label_str, img_path

# ---------------- Load dataset items ----------------
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

# ---------------- Parallel extraction ----------------
def extract_graphs(items, cache_file=None):
    graphs = []
    labels = []
    processed = 0
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached graphs from {cache_file}")
        cache = joblib.load(cache_file)
        return cache["graphs"], cache["labels"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as exc:
        for graph, label, img_path in exc.map(image_item_to_graph, items):
            processed += 1
            if graph is not None:
                graphs.append(graph)
                labels.append(label)
            if processed % 200 == 0 or processed == len(items):
                print(f"Processed {processed}/{len(items)} | extracted {len(graphs)} graphs")
    if cache_file:
        joblib.dump({"graphs": graphs, "labels": labels}, cache_file)
        print(f"Saved graphs cache to {cache_file}")
    return graphs, labels

# ---------------- Prepare numeric labels ----------------
def prepare_graphs_and_labels(graphs, labels):
    class_names = sorted(list(set(labels)))
    name_to_idx = {n:i for i,n in enumerate(class_names)}
    for g, lbl in zip(graphs, labels):
        g.y = torch.tensor([name_to_idx[lbl]], dtype=torch.long)
    return graphs, class_names

# ---------------- GNN Model ----------------
class PoseGNN(torch.nn.Module):
    def __init__(self, node_in=4, hidden=64, angle_feat_dim=12, num_classes=10):
        super().__init__()
        self.conv1 = GCNConv(node_in, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.fc_pool = torch.nn.Linear(hidden, hidden)
        self.fc_angles = torch.nn.Sequential(
            torch.nn.Linear(angle_feat_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU()
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden + 32, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x, edge_index, batch, angles):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.fc_pool(x)
        a = self.fc_angles(angles)
        out = torch.cat([x, a], dim=1)
        return self.classifier(out)

# ---------------- Training / Eval ----------------
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        if hasattr(data, "angles"):
            angles = data.angles.view(-1, 12).to(device)
        else:
            angles = torch.zeros((data.num_graphs, 12), device=device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch, angles)
        loss = F.cross_entropy(out, data.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, preds = [], []
    for data in loader:
        data = data.to(device)
        if hasattr(data, "angles"):
            angles = data.angles.view(-1, 12).to(device)
        else:
            angles = torch.zeros((data.num_graphs, 12), device=device)
        out = model(data.x, data.edge_index, data.batch, angles)
        pred = out.argmax(dim=1).cpu().numpy()
        ys.append(data.y.view(-1).cpu().numpy())
        preds.append(pred)
    ys = np.concatenate(ys)
    preds = np.concatenate(preds)
    acc = (ys == preds).mean() if len(ys) else 0.0
    return acc, ys, preds

# ---------------- Main ----------------
def main():
    print("Loading image items...")
    items = load_image_items(DATA_DIR)
    print(f"Discovered {len(items)} images.")

    print("Extracting graphs (this may take a while)...")
    graphs, labels = extract_graphs(items, cache_file=GRAPH_DATA_FILE)

    if len(graphs) == 0:
        raise RuntimeError("No graphs extracted. Check data dir and Mediapipe results.")

    graphs, class_names = prepare_graphs_and_labels(graphs, labels)
    print(f"Prepared {len(graphs)} graphs across {len(class_names)} classes.")

    idx = list(range(len(graphs)))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=[g.y.item() for g in graphs], random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.15, stratify=[graphs[i].y.item() for i in train_idx], random_state=42)

    train_graphs = [graphs[i] for i in train_idx]
    val_graphs   = [graphs[i] for i in val_idx]
    test_graphs  = [graphs[i] for i in test_idx]

    print(f"Splits -> Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)

    model = PoseGNN(node_in=4, hidden=128, angle_feat_dim=12, num_classes=len(class_names)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)

    best_val = 0.0
    best_state = None
    patience_cnt = 0

    for epoch in range(1, EPOCHS+1):
        loss = train_epoch(model, train_loader, optimizer, DEVICE)
        val_acc, _, _ = evaluate(model, val_loader, DEVICE)
        scheduler.step(1.0 - val_acc)
        print(f"Epoch {epoch:02d} | TrainLoss: {loss:.4f} | ValAcc: {val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            best_state = model.state_dict()
            patience_cnt = 0
            torch.save({"model_state": best_state, "class_names": class_names}, OUT_MODEL)
            print(f"  -> New best val {best_val:.4f} saved.")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---------------- Test Eval + Confusion Matrix ----------------
    test_acc, ys, preds = evaluate(model, test_loader, DEVICE)
    print("\n=== Final Test Results ===")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(ys, preds, target_names=class_names, zero_division=0))

    # confusion matrix
    cm = confusion_matrix(ys, preds)
    print("\nConfusion Matrix shape:", cm.shape)

    # save confusion matrix as npy & csv
    np.save("confusion_matrix.npy", cm)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv("confusion_matrix.csv")
    print("Saved confusion matrix -> confusion_matrix.npy / confusion_matrix.csv")

    # ---------------- Confusion Matrix Heatmap PNG ----------------
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix Heatmap')
    plt.tight_layout()
    cm_png_file = "confusion_matrix.png"
    plt.savefig(cm_png_file)
    plt.close()
    print(f"Saved confusion matrix heatmap -> {cm_png_file}")

    # save artifacts
    joblib.dump(class_names, CLASS_NAMES_FILE)
    torch.save(model.state_dict(), "pose_gcn_final_state.pth")
    print(f"Saved class names -> {CLASS_NAMES_FILE}")
    print("Saved model state -> pose_gcn_final_state.pth")

if __name__ == "__main__":
    main()
