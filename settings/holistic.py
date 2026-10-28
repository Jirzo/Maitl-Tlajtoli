import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class FaceLM:
    """Key indices in MediaPipe FaceMesh 468-point model."""
    NOSE_TIP = 1
    CHIN = 152
    LEFT_EAR = 234
    RIGHT_EAR = 454
    FOREHEAD = 10
    UPPER_LIP = 13
    LOWER_LIP = 14
    LEFT_MOUTH = 61
    RIGHT_MOUTH = 291
    LEFT_BROW_INNER = 107
    RIGHT_BROW_INNER = 336
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374


class PoseLM:
    """Key indices in MediaPipe Pose 33-point model."""
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24


# Per-hand geometry constants (shared with legacy pipeline)
HAND_FINGERTIPS = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky tips
HAND_JOINTS = [
    (2, 3, 4),    # Thumb IP
    (5, 6, 8),    # Index PIP
    (9, 10, 12),  # Middle PIP
    (13, 14, 16), # Ring PIP
    (17, 18, 20), # Pinky PIP
]

# Feature vector layout for the holistic pipeline
N_FEATURES = 48  # features per frame
N_FRAMES = 50    # temporal window length
# Tensor shape fed to LSTM: (N_FRAMES, N_FEATURES) = (50, 48)
#
# Breakdown:
#  [0:10]   Right hand handshape   (5 normalized distances + 5 joint angles)
#  [10:20]  Left hand handshape    (5 normalized distances + 5 joint angles)
#  [20:26]  Right hand location    (2 signed offsets + 4 Euclidean distances, /face_diag)
#  [26:32]  Left hand location     (2 signed offsets + 4 Euclidean distances, /face_diag)
#  [32:36]  Inter-hand relations   (2 distances + 2 signed offsets, /shoulder_width)
#  [36:44]  Face expression        (mouth, eyes, brows, head yaw, head tilt)
#  [44:48]  Motion deltas          (Δright_wrist_x, Δright_wrist_y, Δleft_wrist_x, Δleft_wrist_y)

DATA_HOLISTIC_DIR = "./data/data_holistic"
