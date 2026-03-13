import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib

# Remember to add this to your VIDEOS/SEQUENCES folder (e.g., J, Z)
from settings.collect_image import DATA_VIDEO_DIR

N_FRAMES = 30


def extract_geometric_features(hand_landmarks):
    """
    Extract the 10 invariant geometric values ​​(5 distances, 5 angles).
    """
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    features = []

    palm_vector = landmarks[9] - landmarks[0]
    palm_length = np.linalg.norm(palm_vector)
    if palm_length == 0:
        palm_length = 1e-6

    fingertips = [4, 8, 12, 16, 20]
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


def datasetTFCreation_Dynamic(hands_model):
    class_names = sorted(os.listdir(DATA_VIDEO_DIR))
    print(f"Dynamic classes found: {class_names}")

    class_dict = {name: idx for idx, name in enumerate(class_names)}
    skipped_images_count = {class_name: 0 for class_name in class_names}

    sequences_list = []
    labels_list = []

    for class_name in tqdm(class_names, desc="Processing classes"):
        class_path = os.path.join(DATA_VIDEO_DIR, class_name)
        if not os.path.isdir(class_path):
            continue

        label_idx = class_dict[class_name]
        sequence = []

        # Variables for tracking movement (Deltas)
        prev_wrist_x = None
        prev_wrist_y = None

        for img_path in tqdm(
            sorted(os.listdir(class_path)), desc=f"Frames en {class_name}", leave=False
        ):
            img_full_path = os.path.join(class_path, img_path)
            img = cv2.imread(img_full_path)

            if img is None:
                skipped_images_count[class_name] += 1
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands_model.process(img_rgb)

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]

                # 1. We obtain static geometry (10 values)
                geom_features = extract_geometric_features(hand_landmarks)

                # 2. We obtain wrist coordinates for the Delta
                current_wrist_x = hand_landmarks.landmark[0].x
                current_wrist_y = hand_landmarks.landmark[0].y

                # 3. We calculate the Delta (if it is the first frame, the delta is 0)
                if prev_wrist_x is None:
                    delta_x = 0.0
                    delta_y = 0.0
                else:
                    delta_x = current_wrist_x - prev_wrist_x
                    delta_y = current_wrist_y - prev_wrist_y

                # We updated the wrist memory
                prev_wrist_x = current_wrist_x
                prev_wrist_y = current_wrist_y

                # 4. We combined TODO: Geometry + Movement (12 values ​​per frame)
                frame_features = np.concatenate([geom_features, [delta_x, delta_y]])
                sequence.append(frame_features)

                if len(sequence) == N_FRAMES:
                    sequences_list.append(sequence.copy())
                    labels_list.append(label_idx)
                    sequence = []
                    # We reset the wrist for the next 30-frame sequence
                    prev_wrist_x = None
                    prev_wrist_y = None
            else:
                skipped_images_count[class_name] += 1
        # --- PADDING BLOCK (At the end of the class folder) ---
        if len(sequence) > 0:
            if len(sequence) >= (N_FRAMES // 2):  # If it has at least 15 frames
                print(
                    f"\n[INFO] Filling in sequence {class_name}: of {len(sequence)} to {N_FRAMES} frames."
                )
                ultimo_frame = sequence[-1]

                # We duplicate the last frame until we reach 30
                while len(sequence) < N_FRAMES:
                    sequence.append(ultimo_frame)

                sequences_list.append(sequence.copy())
                labels_list.append(label_idx)
            else:
                print(
                    f"\n[INFO] Discarded sequence in {class_name}: very short ({len(sequence)} frames)."
                )
        # --------------------------------------------------------------
    # We validate that we have actually captured sequences before scaling
    if len(sequences_list) == 0:
        raise ValueError(
            "Critical Error: No valid sequence could be collected. Check your images or MediaPipe detection."
        )

    # Convertimos a tensores 3D: (muestras, 30_frames, 12_caracteristicas)
    X_crudo = np.array(sequences_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int32)

    print(f"\nFinal form of the dynamic tensor: {X_crudo.shape}")
    print("--- Initiating sequential scaling process ---")

    nsamples, ntimesteps, nfeatures = X_crudo.shape
    X_reshaped = X_crudo.reshape(-1, nfeatures)  # We flattened time for the climber

    scaler = StandardScaler()
    X_escalado_reshaped = scaler.fit_transform(X_reshaped)

    # We reconstruct the sequence of time
    X_escalado = X_escalado_reshaped.reshape(nsamples, ntimesteps, nfeatures)

    scaler_filename = "dynamic/my_dynamic_scaler.joblib"
    joblib.dump(scaler, scaler_filename)
    print(f"Climber successfully saved as '{scaler_filename}'")

    save_to_tfrecord(X_escalado, y, "dynamic/dynamic_scaled_landmarks.tfrecord")

    metadata = {"num_classes": len(class_dict), "class_dict": class_dict}
    with open("dynamic/dynamic_metadata.json", "w") as f:
        json.dump(metadata, f)


def save_to_tfrecord(sequences, labels, output_file):
    with tf.io.TFRecordWriter(output_file) as writer:
        for i in tqdm(range(len(sequences)), desc=f"Saving in {output_file}"):
            current_sequence = sequences[i].flatten()
            feature = {
                "landmarks": tf.train.Feature(
                    float_list=tf.train.FloatList(value=current_sequence)
                ),
                "label": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[labels[i]])
                ),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
    print(f"\nDynamic data successfully saved in {output_file}!")
