"""
Dataset pipeline for the Holistic LSTM model.

Since clip_capture.py now saves pre-extracted .npy tensors, this script only
needs to:
  1. Load all (50, 48) sequences from data/data_holistic/<CLASS>/seq_XXXX.npy
  2. Fit a StandardScaler on the flattened representation
  3. Save scaled TFRecord + scaler.joblib + metadata.json

Output directory:  holistic_dataset/
"""

import os
import json
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib

from settings.holistic import N_FRAMES, N_FEATURES, DATA_HOLISTIC_DIR

OUTPUT_DIR    = "holistic_dataset"
TFRECORD_PATH = os.path.join(OUTPUT_DIR, "holistic_landmarks.tfrecord")
SCALER_PATH   = os.path.join(OUTPUT_DIR, "holistic_scaler.joblib")
META_PATH     = os.path.join(OUTPUT_DIR, "holistic_metadata.json")
MIN_SEQUENCES = 5   # Minimum sequences per class; warn if fewer


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _save_tfrecord(X: np.ndarray, y: np.ndarray, path: str) -> None:
    """Serialize (N, T, F) float32 tensor and int32 labels to TFRecord."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tf.io.TFRecordWriter(path) as writer:
        for i in tqdm(range(len(X)), desc=f"Writing {path}"):
            flat = X[i].flatten().tolist()
            feature = {
                "landmarks": tf.train.Feature(
                    float_list=tf.train.FloatList(value=flat)
                ),
                "label": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[int(y[i])])
                ),
            }
            proto = tf.train.Example(
                features=tf.train.Features(feature=feature)
            )
            writer.write(proto.SerializeToString())
    print(f"[OK] TFRecord saved: {path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_holistic_dataset() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isdir(DATA_HOLISTIC_DIR):
        raise FileNotFoundError(
            f"Data directory '{DATA_HOLISTIC_DIR}' not found. "
            "Run VideoRecorder/clip_capture.py first to collect sequences."
        )

    class_names = sorted(
        d for d in os.listdir(DATA_HOLISTIC_DIR)
        if os.path.isdir(os.path.join(DATA_HOLISTIC_DIR, d))
    )
    if not class_names:
        raise ValueError(f"No class subdirectories found in '{DATA_HOLISTIC_DIR}'.")

    class_dict = {name: idx for idx, name in enumerate(class_names)}
    print(f"[INFO] Classes found ({len(class_names)}): {class_names}")

    sequences_list: list[np.ndarray] = []
    labels_list: list[int] = []

    for class_name in tqdm(class_names, desc="Loading classes"):
        class_path = os.path.join(DATA_HOLISTIC_DIR, class_name)
        npy_files = sorted(
            f for f in os.listdir(class_path) if f.endswith(".npy")
        )

        if len(npy_files) < MIN_SEQUENCES:
            print(
                f"[WARN] Class '{class_name}' has only {len(npy_files)} sequences "
                f"(minimum recommended: {MIN_SEQUENCES})."
            )

        label_idx = class_dict[class_name]
        for fname in npy_files:
            fpath = os.path.join(class_path, fname)
            tensor = np.load(fpath)

            if tensor.shape != (N_FRAMES, N_FEATURES):
                print(
                    f"[SKIP] {fpath} has unexpected shape {tensor.shape}, "
                    f"expected ({N_FRAMES}, {N_FEATURES}). Skipping."
                )
                continue

            sequences_list.append(tensor)
            labels_list.append(label_idx)

    if len(sequences_list) == 0:
        raise ValueError(
            "No valid sequences loaded. Check that .npy files exist and have "
            f"shape ({N_FRAMES}, {N_FEATURES})."
        )

    # Stack into (N, T, F) tensor
    X_raw = np.stack(sequences_list, axis=0).astype(np.float32)  # (N, 50, 48)
    y     = np.array(labels_list, dtype=np.int32)

    print(f"\n[INFO] Raw tensor shape:  {X_raw.shape}")
    print(f"[INFO] Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Scale features: flatten time → fit scaler → reshape back
    # This treats every (sample, timestep) pair as an independent observation,
    # which gives a per-feature mean/std across the whole dataset.
    N, T, F = X_raw.shape
    X_flat = X_raw.reshape(-1, F)       # (N*T, F)

    scaler = StandardScaler()
    X_scaled_flat = scaler.fit_transform(X_flat)

    X_scaled = X_scaled_flat.reshape(N, T, F)  # (N, 50, 48)

    joblib.dump(scaler, SCALER_PATH)
    print(f"[OK] Scaler saved: {SCALER_PATH}")

    _save_tfrecord(X_scaled, y, TFRECORD_PATH)

    metadata = {
        "num_classes": len(class_dict),
        "class_dict": class_dict,        # {name: index}
        "n_frames": N_FRAMES,
        "n_features": N_FEATURES,
        "n_sequences": int(N),
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"[OK] Metadata saved: {META_PATH}")
    print(f"\n[DONE] Dataset ready: {N} sequences, {len(class_dict)} classes.")


if __name__ == "__main__":
    build_holistic_dataset()
