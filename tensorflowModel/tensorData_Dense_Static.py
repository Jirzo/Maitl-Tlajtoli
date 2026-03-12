import tensorflow as tf
import json

# --- Static Configuration ---
input_dim = 10  # 5 distances + 5 angles

# 1. Read the number of classes automatically
with open("staticSet/metadata_estatica.json", "r") as f:
    metadata = json.load(f)
num_classes = metadata["num_classes"]


def _parse_function_static(example_proto):
    feature_description = {
        "landmarks": tf.io.FixedLenFeature([input_dim], tf.float32),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    return parsed_example["landmarks"], parsed_example["label"]


def load_dataset_static(tfrecord_path, batch_size=32):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(_parse_function_static)
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    return dataset


def tf_static_trainer():
    # Load the entire dataset
    full_dataset = load_dataset_static("staticSet/static_scaler_landmarks.tfrecord")

    # --- THE CORRECT WAY TO SPLIT ---
    # Assuming you have enough batches, we use 80% for training and 20% for validation.
    dataset_size = sum(1 for _ in full_dataset)

    # THE GIANT BLENDER: buffer_size=dataset_size
    shuffled_dataset = full_dataset.shuffle(
        buffer_size=dataset_size, seed=42, reshuffle_each_iteration=False
    )

    train_size = int(0.8 * dataset_size)

    train_ds = shuffled_dataset.take(train_size)
    test_ds = shuffled_dataset.skip(train_size)

    # --- Dense Model (Static) ---
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "modelDense/better_static_model.keras",
        save_best_only=True,
        monitor="val_accuracy",
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True
    )

    model.fit(
        train_ds, validation_data=test_ds, epochs=50, callbacks=[checkpoint, early_stop]
    )
    print("\nTraining completed. Best model saved as 'better_static_model.keras'")
