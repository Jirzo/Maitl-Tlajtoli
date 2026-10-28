"""
LSTM trainer for the Holistic word-recognition pipeline.

Input tensor shape: (batch, 50, 48)
  - 50 frames (temporal window)
  - 48 features per frame (see settings/holistic.py for breakdown)

Architecture design decisions
-------------------------------
1. Masking layer
   When a hand or face is not detected, holistic_frame_features() fills
   the corresponding slots with zeros. Masking(mask_value=0.0) propagates
   a boolean mask through the recurrent layers so those timesteps are
   skipped rather than distorting the hidden state.

2. Bidirectional LSTM
   Signs unfold in time but their identity is often determined by how they
   END (return to rest position), not just how they start. BiLSTM reads
   the sequence both forward and backward, capturing both directions.

3. LayerNormalization (not BatchNorm)
   BatchNorm statistics are computed over the batch dimension, which is
   problematic for sequential data with variable-length padding and small
   batch sizes. LayerNorm normalizes per-sample, per-timestep → more stable.

4. GlobalAveragePooling1D
   Taking only the final hidden state discards information from earlier
   frames. Averaging over all timesteps (weighted by the mask) is more
   robust when the sign's peak occurs at different points in different
   recordings.

5. Gradient clipping (clipnorm=1.0)
   With a 48-dim input and two BiLSTM layers, the gradient magnitudes can
   be large. Clipping the global norm prevents exploding gradients without
   the need for very small learning rates.

6. L2 regularization on Dense layers
   L2 penalizes large weights, acting as a soft constraint that pushes
   the model toward solutions that generalize better. More important here
   than before because the feature space is 4× wider than the old pipeline.

7. Recurrent dropout (not regular dropout on LSTM outputs)
   Regular dropout applied to recurrent connections is known to cause
   vanishing gradients. Keras LSTM's `recurrent_dropout` parameter uses
   the variational dropout approach (Gal & Ghahramani, 2016) which applies
   the SAME mask at each timestep — stable and effective.
"""

import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers

from settings.holistic import N_FRAMES, N_FEATURES

TFRECORD_PATH = "holistic_dataset/holistic_landmarks.tfrecord"
META_PATH     = "holistic_dataset/holistic_metadata.json"
OUTPUT_DIR    = "modelLSTM_Holistic"
MODEL_PATH    = os.path.join(OUTPUT_DIR, "best_holistic_word_model.keras")

# Training hyperparameters
BATCH_SIZE   = 16
EPOCHS       = 100
LR           = 1e-3
L2_STRENGTH  = 1e-4


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _parse_fn(example_proto, n_frames: int, n_features: int):
    """Parse a single TFRecord example into (tensor, label)."""
    feature_description = {
        "landmarks": tf.io.FixedLenFeature([n_frames * n_features], tf.float32),
        "label":     tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    sequence = tf.reshape(parsed["landmarks"], (n_frames, n_features))
    return sequence, parsed["label"]


def load_dataset(tfrecord_path: str, n_frames: int, n_features: int) -> tf.data.Dataset:
    raw_ds = tf.data.TFRecordDataset(tfrecord_path)
    return raw_ds.map(
        lambda x: _parse_fn(x, n_frames, n_features),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

def build_holistic_lstm(n_classes: int, n_frames: int = N_FRAMES, n_features: int = N_FEATURES) -> tf.keras.Model:
    """
    Returns a compiled Bidirectional LSTM for word-level LSM recognition.

    Tensor flow:
        (batch, 50, 48)
         → Masking           — zeros-out padded timesteps
         → LayerNorm         — normalize per feature across the 50 timesteps
         → BiLSTM(128)       — 256-dim output per timestep (forward + backward)
         → LayerNorm + Drop  — stabilize + regularize
         → BiLSTM(64)        — 128-dim output per timestep
         → GlobalAvgPool1D   — collapse time dimension robustly
         → Dense(128, L2)    — classification head
         → LayerNorm + Drop
         → Dense(64, L2)
         → Drop
         → Dense(n_classes, softmax)
    """
    inp = layers.Input(shape=(n_frames, n_features), name="sequence_input")

    x = layers.Masking(mask_value=0.0, name="masking")(inp)
    x = layers.LayerNormalization(name="input_norm")(x)

    # First Bidirectional LSTM block
    x = layers.Bidirectional(
        layers.LSTM(
            128,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.1,
            name="lstm_1",
        ),
        name="bilstm_1",
    )(x)
    x = layers.LayerNormalization(name="norm_1")(x)
    x = layers.Dropout(0.3, name="drop_1")(x)

    # Second Bidirectional LSTM block
    x = layers.Bidirectional(
        layers.LSTM(
            64,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.1,
            name="lstm_2",
        ),
        name="bilstm_2",
    )(x)
    x = layers.LayerNormalization(name="norm_2")(x)
    x = layers.Dropout(0.3, name="drop_2")(x)

    # Pool over the time axis — more robust than taking only the last state
    x = layers.GlobalAveragePooling1D(name="temporal_pool")(x)

    # Classification head
    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.L2(L2_STRENGTH),
        name="dense_1",
    )(x)
    x = layers.LayerNormalization(name="norm_3")(x)
    x = layers.Dropout(0.4, name="drop_3")(x)

    x = layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=regularizers.L2(L2_STRENGTH),
        name="dense_2",
    )(x)
    x = layers.Dropout(0.3, name="drop_4")(x)

    out = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = tf.keras.Model(inputs=inp, outputs=out, name="HolisticWordLSTM")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train_holistic_lstm() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    n_classes  = meta["num_classes"]
    n_frames   = meta["n_frames"]
    n_features = meta["n_features"]
    n_total    = meta["n_sequences"]

    print(f"[INFO] Classes: {n_classes}  |  Sequences: {n_total}  |  "
          f"Input shape: ({n_frames}, {n_features})")

    if n_total < 20:
        print("[WARN] Very few sequences. Consider collecting at least 50–100 per class.")

    full_ds = load_dataset(TFRECORD_PATH, n_frames, n_features)

    # Train / validation split (80 / 20), reproducible
    full_ds = full_ds.shuffle(buffer_size=max(n_total, 500), seed=42, reshuffle_each_iteration=False)
    train_size = int(0.8 * n_total)
    val_size   = n_total - train_size

    train_ds = (
        full_ds.take(train_size)
        .shuffle(buffer_size=500)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        full_ds.skip(train_size)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    print(f"[INFO] Train: {train_size}  |  Val: {val_size}  |  Batch: {BATCH_SIZE}")

    model = build_holistic_lstm(n_classes, n_frames, n_features)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    print(f"\n[DONE] Best model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_holistic_lstm()
