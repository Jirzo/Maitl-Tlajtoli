import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib

from settings.collect_image import DATA_DIR_ESTATIC as DATA_DIR


def extract_goemetric_feature(hand_landmarks):
    """
    Transform absolute coordinates into scale- and translation-invariant features.
    """
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    features = []

    # --- SCALE INVARIANTS: Standardized distances between key points ---
    # We calculate the length of the palm (Wrist 0 to Middle Knuckle 9) as a reference to normalize the distances.
    palm_vector = landmarks[9] - landmarks[0]
    palm_length = np.linalg.norm(palm_vector)

    if palm_length == 0:
        palm_length = 1e-6  # Avoid division by zero

    # Standardized distances (Tips to the wrist divided by the palm)
    fingertips = [4, 8, 12, 16, 20]  # Index fingers
    for tip in fingertips:
        dist = np.linalg.norm(landmarks[tip] - landmarks[0])
        features.append(dist / palm_length)  # This is where the magic of scale happens

    # --- ROTATIONAL INVARIANCE: Angles between vectors ---
    # Joint Angles
    joints = [
        (2, 3, 4),
        (5, 6, 8),
        (9, 10, 12),
        (13, 14, 16),
        (17, 18, 20),
    ]  # Finger joints
    for p1, p2, p3 in joints:
        v1 = landmarks[p1] - landmarks[p2]
        v2 = landmarks[p3] - landmarks[p2]

        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        features.append(np.arccos(cosine_angle))  # Angle in radians

    return np.array(features, dtype=np.float32)


def datasetTFCreation_Static(hands_model):
    class_names = sorted(os.listdir(DATA_DIR))
    print(f"Classes found: {class_names}")

    class_dict = {name: idx for idx, name in enumerate(class_names)}
    skipped_images_count = {class_name: 0 for class_name in class_names}

    features_list = []
    labels_list = []

    for class_name in tqdm(class_names, desc="Processing classes"):
        class_path = os.path.join(DATA_DIR, class_name)
        if not os.path.isdir(class_path):
            continue

        label_idx = class_dict[class_name]

        for img_path in tqdm(
            sorted(os.listdir(class_path)),
            desc=f"images in {class_name}",
            leave=False,
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

                # --- NEW: Geometry extraction instead of (x, y, x) ---
                geometric_feratures = extract_goemetric_feature(hand_landmarks)
                features_list.append(geometric_feratures)
                labels_list.append(label_idx)
            else:
                skipped_images_count[class_name] += 1

    print("\nSummary of skipped images:")
    for class_name, count in skipped_images_count.items():
        print(f"Class '{class_name}': {count} skipped images")

    # 2D Matrix: (number of images, number of features)
    X_crudo = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int32)

    print(f"\nFinal form of the extracted data: {X_crudo.shape}")
    print("--- Starting data scaling process ---")

    # Scaling is now much more straightforward; we don't need to do any re-clipping.
    scaler = StandardScaler()
    X_escalado = scaler.fit_transform(X_crudo)

    scaler_filename = "staticSet/scaler_static.joblib"
    joblib.dump(scaler, scaler_filename)
    print(f"Climber saved as '{scaler_filename}'")

    save_to_tfrecord(X_escalado, y, "staticSet/static_scaler_landmarks.tfrecord")

    metadata = {"num_classes": len(class_dict), "class_dict": class_dict}
    with open("staticSet/metadata_estatica.json", "w") as f:
        json.dump(metadata, f)


def save_to_tfrecord(features, labels, output_file):
    with tf.io.TFRecordWriter(output_file) as writer:
        for i in tqdm(range(len(features)), desc=f"GSaving in {output_file}"):
            # Here we take the direct vector
            current_feature = features[i].flatten()
            features_dict = {
                "landmarks": tf.train.Feature(
                    float_list=tf.train.FloatList(value=current_feature)
                ),
                "label": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[labels[i]])
                ),
            }
            example = tf.train.Example(
                features=tf.train.Features(feature=features_dict)
            )
            writer.write(example.SerializeToString())
    print(f"\nData successfully saved in '{output_file}'")
