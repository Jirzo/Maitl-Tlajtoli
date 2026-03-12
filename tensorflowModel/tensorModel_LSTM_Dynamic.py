import tensorflow as tf
import json

# --- Dynamic Configuration ---
timesteps = 30
input_dim = 12  # 10 geometry + 2 motion deltas

with open("dynamic/metadata_dinamica.json", "r") as f:
    metadata = json.load(f)
num_classes = metadata["num_classes"]


def _parse_function_dynamic(example_proto):
    # We expect 30 * 12 = 360 flat values
    feature_description = {
        "landmarks": tf.io.FixedLenFeature([timesteps * input_dim], tf.float32),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)

    # We reconstruct the time matrix: (30, 12)
    landmarks = tf.reshape(parsed_example["landmarks"], (timesteps, input_dim))
    return landmarks, parsed_example["label"]


# --- CHANGE 1: We don't do batching here ---
def load_raw_dataset(tfrecord_path):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(_parse_function_dynamic)
    return dataset


def tf_trainer():
    raw_dataset = load_raw_dataset("dynamic/landmarks_dinamicos_escalados.tfrecord")

    # --- CHANGE 2: We count individual videos (not boxes) ---
    dataset_size = sum(1 for _ in raw_dataset)
    print(f"\n[INFO] Total number of sequences (videos) read: {dataset_size}")

    shuffled_dataset = raw_dataset.shuffle(
        buffer_size=1000, seed=42, reshuffle_each_iteration=False
    )

    if dataset_size < 10:
        print("[WARNING] You have very little data. The model may overfit or fail.")

    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size

    # NOW YES: We separate from the already scrambled dataset
    train_ds = shuffled_dataset.take(train_size)
    test_ds = shuffled_dataset.skip(train_size)

    print(
        f"[INFO] Training with {train_size} sequences, validating with {test_size}.\n"
    )

    # --- CHANGE 3: We group into smaller boxes (e.g., 8 at a time) AFTER separating ---
    batch_size = 8
    train_ds = train_ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # --- LSTM Model (Dynamic) ---
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(timesteps, input_dim)),
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "modelLSTM/best_dynamic_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    model.fit(
        train_ds, validation_data=test_ds, epochs=50, callbacks=[checkpoint, early_stop]
    )
    print("Dynamic training completed.")
