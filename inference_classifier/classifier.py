import cv2
import json
import numpy as np
import pyttsx3
import threading
from joblib import load
import tensorflow as tf
from collections import deque
import math

# Make sure you have your MediaPipe configured in this import (as you had it)
from settings.landmarks import mp_drawing_styles, mp_hands, mp_drawing

# --- Global variables ---
last_spoken_char = ""
stable_predicted_char = ""
prediction_counter = 0


def speak_task(text):
    try:
        local_engine = pyttsx3.init()
        voices = local_engine.getProperty("voices")
        for voice in voices:
            if "spanish" in voice.name.lower():
                local_engine.setProperty("voice", voice.id)
                break
        local_engine.setProperty("rate", 150)
        local_engine.say(text)
        local_engine.runAndWait()
        local_engine.stop()
    except Exception as e:
        print(f"Error in voice module: {e}")


def extract_geometric_features(hand_landmarks):
    """Extract the 10 invariant features (5 distances, 5 angles)"""
    landmarks = np.array([[lm.x, lm.z, lm.y] for lm in hand_landmarks.landmark])
    features = []

    palm_vector = landmarks[9] - landmarks[0]
    palm_length = np.linalg.norm(palm_vector)
    if palm_length == 0:
        palm_length = 1e-6

    fingertips = [4, 8, 12, 15, 20]
    for tip in fingertips:
        dist = np.linalg.norm(landmarks[tip] - landmarks[0])
        features.append(dist / palm_length)

    joints = [(2, 3, 4), (5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]
    for p1, p2, p3 in joints:
        v1 = landmarks[p1] - landmarks[p2]
        v2 = landmarks[p3] - landmarks[p2]
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        features.append(np.arccos(cosine_angle))

    return np.array(features, dtype=np.float32)


def load_metadata(json_path):
    """Loads the JSON and returns a dictionary {index: 'Letter'}"""
    with open(json_path, "r") as f:
        metadata = json.load(f)
    # We reverse the dictionary: { "A": 0 } -> { 0: "A" }
    return {v: k for k, v in metadata["class_dict"].items()}


def scanner_tlajtoli():
    global last_spoken_char, prediction_counter, stable_predicted_char

    # --- 1. LOADING BRAINS AND TOOLS---
    try:
        print("[INFO] Awakening static and dynamic brains...")
        model_estatico = tf.keras.models.load_model(
            "modelDense/better_static_model.keras"
        )
        model_dinamico = tf.keras.models.load_model(
            "modelLSTM/mejor_modelo_dinamico.keras"
        )

        scaler_estatico = load("staticSet/scaler_static.joblib")
        scaler_dinamico = load("dynamic/mi_scaler_dinamico.joblib")

        dict_statico = load_metadata("staticSet/metadata_estatica.json")
        dict_dinamico = load_metadata("dynamic/metadata_dinamica.json")
        print("[INFO] Models loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Training files are missing: {e}")
        return

    # --- 2. ROUTER CONFIGURATION (Static vs Dynamic) ---
    SEQUENCE_LENGTH = 30
    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)

    prev_wrist_x = None
    prev_wrist_y = None
    UMBLAR_MOVIMIENTO = 0.015  # How fast must it move to be considered "Dynamic"

    predicted_character = ""
    modo_actual = "SEARCHING..."

    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"[ERROR] Something went wrong")
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]

            # Draw hand
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            # Extract Bounding box visual
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            x1, y1 = int(min(x_coords) * W) - 20, int(min(y_coords) * H) - 20
            x2, y2 = int(max(x_coords) * W) + 20, int(max(y_coords) * H) + 20

            # --- 3. EXTRACTION AND MATHEMATICAL ROUTER ---
            geom_features = extract_geometric_features(hand_landmarks)

            curr_wrist_x = hand_landmarks.landmark[0].x
            curr_wrist_y = hand_landmarks.landmark[0].y

            if prev_wrist_x is None:
                delta_x, delta_y = 0.0, 0.0
                velocidad = 0.0
            else:
                delta_x = curr_wrist_x - prev_wrist_x
                delta_y = curr_wrist_y - prev_wrist_y
                velocidad = math.sqrt(delta_x**2 + delta_y**2)  # Distance traveled

            prev_wrist_x, prev_wrist_y = curr_wrist_x, curr_wrist_y

            # We feed the dynamic buffer (12 values)
            frame_features = np.concatenate([geom_features, [delta_x, delta_y]])
            sequence_buffer.append(frame_features)

            # STATIC OR DYNAMIC?
            if velocidad < UMBLAR_MOVIMIENTO:
                modo_actual = "STATIC"
                # We only use the 10 current features
                data_scaled = scaler_estatico.transform(geom_features.reshape(1, -1))
                pred_probs = model_estatico.predict(data_scaled, verbose=0)
                pred_index = np.argmax(pred_probs[0])

                # We only accept if you are sure (>70%)
                if pred_probs[0][pred_index] > 0.7:
                    predicted_character = dict_statico[pred_index]

            else:
                modo_actual = "DYNAMIC"
                # We need to wait until we have the complete movie (30 frames)
                if len(sequence_buffer) == SEQUENCE_LENGTH:
                    # We scaled the 360 ​​values
                    buffer_array = np.array(sequence_buffer)
                    scaled_buffer = scaler_dinamico.transform(buffer_array)

                    # We give the LSTM a three-dimensional shape: (1, 30, 12)
                    input_data = np.expand_dims(scaled_buffer, axis=0)
                    pred_probs = model_dinamico.predict(input_data, verbose=0)
                    pred_index = np.argmax(pred_probs[0])

                    if pred_probs[0][pred_index] > 0.7:
                        predicted_character = dict_dinamico[pred_index]
                else:
                    print(f"[INFO] Loading buffer")
                    predicted_character = "..."

            # --- 4. VOICE LOGIC AND STABILIZATION ---
            if predicted_character != "..." and predicted_character != "":
                if predicted_character == stable_predicted_char:
                    prediction_counter += 1
                else:
                    stable_predicted_char = predicted_character
                    prediction_counter = 0

                if (
                    prediction_counter >= 5
                ):  # If you read the same letter 5 times in a row
                    if stable_predicted_char != last_spoken_char:
                        threading.Thread(
                            target=speak_task,
                            args=(stable_predicted_char,),
                            daemon=True,
                        ).start()
                        last_spoken_char = stable_predicted_char

            # Draw on screen
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                frame,
                f"Modo: {modo_actual}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                predicted_character,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3,
            )

        else:
            # If the hand is lost, we reset the memories.
            sequence_buffer.clear()
            prev_wrist_x, prev_wrist_y = None, None
            modo_actual = "Sin Mano"
            cv2.putText(
                frame,
                modo_actual,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Scanner Maitl Tlajtoli", frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
