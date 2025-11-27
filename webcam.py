#!/usr/bin/env python3
"""
test_pose_gnn_webcam.py
Real-time yoga pose recognition using trained GNN model.
Shows Pose, Step, Correctness, and FPS on webcam feed.
"""

import cv2
import torch
import joblib
import numpy as np
import mediapipe as mp
import time
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

# ---------- Paths ----------
MODEL_FILE = "pose_gcn_best.pt"   # saved during training
CLASS_NAMES_FILE = "pose_classes.pkl"

# ---------- Mediapipe setup ----------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# ---------- Pose Graph Edges ----------
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

# ---------- Angle calculation ----------
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

# ---------- Normalization ----------
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
    return arr

# ---------- GNN Model ----------
class PoseGNN(torch.nn.Module):
    def __init__(self, node_in=4, hidden=128, angle_feat_dim=12, num_classes=10):
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

    def forward(self, x, edge_index, angles):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        x = self.fc_pool(x)
        a = self.fc_angles(angles)
        out = torch.cat([x, a], dim=1)
        return self.classifier(out)

# ---------- Load model ----------
checkpoint = torch.load(MODEL_FILE, map_location="cpu")
class_names = joblib.load(CLASS_NAMES_FILE)
num_classes = len(class_names)

model = PoseGNN(node_in=4, hidden=128, angle_feat_dim=12, num_classes=num_classes)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# ---------- Webcam ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Could not open webcam")

prev_time = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        nodes_np = normalize_landmarks_np(results.pose_landmarks.landmark)
        node_feats = torch.tensor(nodes_np[:, :4], dtype=torch.float).unsqueeze(0)
        angles = extract_angles(results.pose_landmarks.landmark)
        angles_tensor = torch.tensor(angles, dtype=torch.float).unsqueeze(0)

        with torch.no_grad():
            out = model(node_feats[0], edge_index, angles_tensor)
            pred_idx = out.argmax(dim=1).item()
            pred_label = class_names[pred_idx]

        # parse name (Pose, Step, Correctness)
        parts = pred_label.split('_')
        pose_name = parts[0]
        correctness = "Right" if "right" in pred_label.lower() else "Wrong"
        step = " ".join(p for p in parts if "step" in p)

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(frame, f"Pose: {pose_name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, f"Step: {step}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.putText(frame, f"Correctness: {correctness}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0,0,255) if correctness == "Wrong" else (0,255,0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Pose GNN Inference", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
