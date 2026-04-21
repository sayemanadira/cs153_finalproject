from collections import deque
import mediapipe as mp
import numpy as np
from pathlib import Path

import cv2
from sympy import re


def parse_label_from_filename(filename: str) -> int:
    # Match things like _3L, _5R
    m = re.search(r'_([0-5])[LR](?=\D|$)', filename)
    if m:
        return int(m.group(1))

    # Fallback: final _digit before extension, e.g. img6_0L.jpeg may already match above,
    # but this catches simpler patterns if needed.
    m = re.search(r'_([0-5])(?=\.[^.]+$)', filename)
    if m:
        return int(m.group(1))

    raise ValueError(f"Could not parse label from filename: {filename}")

def list_images(folder: Path):
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])


mp_hands = mp.solutions.hands

hands_static = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence =0.5,
    min_tracking_confidence=0.5,
)

TIP_IDS = [4, 8, 12, 16, 20]
BASE_IDS = [2, 5, 9, 13, 17]


def normalize_landmarks(landmarks_xyz: np.ndarray) -> np.ndarray:
    '''
    landmarks_xyz: (21, 3)
    Returns normalized landmarks centered at wrist and scaled by hand size.
    '''
    pts = landmarks_xyz.copy()

    # center on wrist
    wrist = pts[0].copy()
    pts = pts - wrist

    # scale by max distance from wrist to any landmark
    scale = np.max(np.linalg.norm(pts[:, :2], axis=1))
    if scale < 1e-6:
        scale = 1.0
    pts = pts / scale
    return pts

def extract_landmarks_from_bgr(image_bgr: np.ndarray):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = hands_static.process(image_rgb)
    if not result.multi_hand_landmarks:
        return None

    hand_landmarks = result.multi_hand_landmarks[0]
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
    return pts


def draw_landmarks_on_image(path: Path):
    image_bgr = cv2.imread(str(path))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = hands_static.process(image_rgb)

    display = image_rgb.copy()
    if result.multi_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            display,
            result.multi_hand_landmarks[0],
            mp_hands.HAND_CONNECTIONS
        )
    return display


def landmarks_to_features(pts: np.ndarray) -> np.ndarray:
    pts_norm = normalize_landmarks(pts)

    features = []

    # raw normalized landmark coords
    features.extend(pts_norm.flatten())

    # fingertip-to-base distances: useful for finger count
    for tip, base in zip(TIP_IDS, BASE_IDS):
        d = np.linalg.norm(pts_norm[tip, :2] - pts_norm[base, :2])
        features.append(d)

    # fingertip y relative to lower joint (good for non-thumb fingers)
    for tip, pip in [(8,6), (12,10), (16,14), (20,18)]:
        features.append(pts_norm[tip,1] - pts_norm[pip,1])

    return np.array(features, dtype=np.float32)


def extract_feature_from_path(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        return None
    pts = extract_landmarks_from_bgr(image)
    if pts is None:
        return None
    return landmarks_to_features(pts)




def build_feature_table(files):
    X, y, kept_files, failed_files = [], [], [], []

    for path in files:
        try:
            label = parse_label_from_filename(path.name)
            feat = extract_feature_from_path(path)
            if feat is None:
                failed_files.append(path.name)
                continue
            X.append(feat)
            y.append(label)
            kept_files.append(path.name)
        except Exception as e:
            failed_files.append(f"{path.name} | {e}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y, kept_files, failed_files

def preprocess_frame(frame):
    return frame

def process_real_time_video(model, smoothing_window=7, display_every_n=1):
    cap = cv2.VideoCapture(0)
    pred_history = deque(maxlen=smoothing_window)
    raw_preds = []
    smooth_preds = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess the frame and make a prediction
        input_data = preprocess_frame(frame)
        pred = model.predict(input_data)
        
        # Store the raw prediction and update the history
        raw_preds.append(pred)
        pred_history.append(pred)
        
        # Get the most common prediction in the history for smoothing
        smooth_pred = most_common(pred_history)
        smooth_preds.append(smooth_pred)
        
        # Display the prediction every n frames
        if len(raw_preds) % display_every_n == 0:
            print(f"Raw Prediction: {pred}, Smoothed Prediction: {smooth_pred}")
        
        # Optionally, display the frame with predictions (not implemented here)
    
    cap.release()
    return raw_preds, smooth_preds


