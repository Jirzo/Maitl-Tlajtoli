"""
Holistic sequence recorder for Maitl Tlajtoli.

Replaces the old JPEG-per-frame approach with direct .npy tensor capture:
  - Uses MediaPipe Holistic (hands + face + pose) instead of Hands-only.
  - Records 50-frame windows (up from 30).
  - Saves each complete sequence as a single .npy file of shape (50, 48),
    which is ready to feed directly into the LSTM without re-processing.

Output directory:  data/data_holistic/<SIGN>/seq_XXXX.npy
"""

import cv2
import os
import numpy as np
import mediapipe as mp

from settings.holistic import (
    mp_holistic, mp_drawing, mp_drawing_styles,
    N_FRAMES, N_FEATURES, DATA_HOLISTIC_DIR,
)
from dataset.feature_engineering import holistic_frame_features

# Holistic model configuration
HOLISTIC_CONFIG = dict(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
    model_complexity=1,     # 0=lite, 1=full — use 1 for accurate face mesh
)


def _next_seq_index(path: str) -> int:
    """Return the next available sequence index in `path`."""
    existing = [
        f for f in os.listdir(path)
        if f.startswith("seq_") and f.endswith(".npy")
    ]
    if not existing:
        return 0
    indices = [int(f.split("_")[1].split(".")[0]) for f in existing]
    return max(indices) + 1


def prepare_class_dir(sign_class: str) -> tuple[str, int]:
    path = os.path.join(DATA_HOLISTIC_DIR, sign_class)
    os.makedirs(path, exist_ok=True)
    return path, _next_seq_index(path)


def _draw_overlay(frame, sign_class: str, seq_count: int, recording: bool, n: int):
    h, w = frame.shape[:2]
    color = (0, 0, 255) if recording else (0, 255, 0)
    status = f"REC {n}/{N_FRAMES}" if recording else "R=record  Q=quit  C=change sign"
    cv2.putText(frame, f"Sign: {sign_class}  Saved: {seq_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, status,
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


def run_capture():
    sign_class = input(
        "Sign/word to record (e.g. HOLA, GRACIAS, verbo_comer): "
    ).strip().upper()

    class_path, seq_counter = prepare_class_dir(sign_class)
    print(f"\n[INFO] Saving to: {class_path}")
    print(f"[INFO] Sequences already saved: {seq_counter}")
    print("[INFO] Controls:  R = record 50-frame sequence | C = change sign | Q = quit\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    with mp_holistic.Holistic(**HOLISTIC_CONFIG) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Frame capture failed.")
                break

            _draw_overlay(frame, sign_class, seq_counter, recording=False, n=0)
            cv2.imshow("Holistic Recorder — Maitl Tlajtoli", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("c"):
                cap.release()
                cv2.destroyAllWindows()
                sign_class = input("\nNew sign: ").strip().upper()
                class_path, seq_counter = prepare_class_dir(sign_class)
                print(f"[INFO] Switched to: {sign_class}  (seq idx starts at {seq_counter})")
                cap = cv2.VideoCapture(0)
                continue

            elif key == ord("r"):
                print(f"[REC] Recording 50 frames for '{sign_class}'...")

                sequence   = []
                prev_r_wrist = None
                prev_l_wrist = None
                frames_captured = 0

                while frames_captured < N_FRAMES:
                    ret, frame_rec = cap.read()
                    if not ret:
                        print("[WARN] Dropped frame during recording.")
                        continue

                    frame_rgb = cv2.cvtColor(frame_rec, cv2.COLOR_BGR2RGB)
                    frame_rgb.flags.writeable = False
                    result = holistic.process(frame_rgb)
                    frame_rgb.flags.writeable = True

                    # Extract the 48-dim feature vector for this frame
                    feat, prev_r_wrist, prev_l_wrist = holistic_frame_features(
                        result, prev_r_wrist, prev_l_wrist
                    )
                    sequence.append(feat)
                    frames_captured += 1

                    # Draw skeleton feedback during recording
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    if result.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            frame_bgr,
                            result.pose_landmarks,
                            mp_holistic.POSE_CONNECTIONS,
                        )
                    if result.right_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame_bgr,
                            result.right_hand_landmarks,
                            mp_holistic.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style(),
                        )
                    if result.left_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame_bgr,
                            result.left_hand_landmarks,
                            mp_holistic.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style(),
                        )

                    _draw_overlay(frame_bgr, sign_class, seq_counter,
                                  recording=True, n=frames_captured)
                    cv2.imshow("Holistic Recorder — Maitl Tlajtoli", frame_bgr)
                    cv2.waitKey(1)

                # Save complete sequence as a single .npy tensor (50, 48)
                tensor = np.array(sequence, dtype=np.float32)   # (50, 48)
                assert tensor.shape == (N_FRAMES, N_FEATURES), (
                    f"Shape error: {tensor.shape}, expected ({N_FRAMES}, {N_FEATURES})"
                )

                out_path = os.path.join(class_path, f"seq_{seq_counter:04d}.npy")
                np.save(out_path, tensor)
                seq_counter += 1
                print(f"[OK] Saved: {out_path}  (shape {tensor.shape})")

                # Optionally switch sign after each recording
                change = input(
                    "\nPress Enter to record another for the same sign, "
                    "or type a new sign name: "
                ).strip().upper()
                if change:
                    sign_class = change
                    class_path, seq_counter = prepare_class_dir(sign_class)
                    print(f"[INFO] Switched to: {sign_class}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_capture()
