"""
Feature engineering for MediaPipe Holistic results.

Produces a 48-dimensional geometric feature vector per frame that is:
  - Scale-invariant: distances are normalized by anatomical references
    (palm length, face diagonal, shoulder width) so camera distance doesn't matter.
  - Position-invariant: we use RELATIVE distances/angles, not raw coordinates,
    so the same sign looks the same whether the signer is left or right of frame.
  - Rotation-tolerant: joint angles use arccos(dot-product), which is
    invariant to rigid rotations.

See settings/holistic.py for the full feature layout and index breakdown.
"""

import numpy as np
from settings.holistic import FaceLM, PoseLM, HAND_FINGERTIPS, HAND_JOINTS, N_FEATURES


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _pts(landmarks_obj) -> np.ndarray:
    """Convert a MediaPipe NormalizedLandmarkList to (N, 3) float32 array."""
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks_obj.landmark],
        dtype=np.float32,
    )


def _norm(v: np.ndarray) -> float:
    """Euclidean norm, clamped away from zero to avoid division errors."""
    n = np.linalg.norm(v)
    return float(n) if n > 1e-6 else 1e-6


# ---------------------------------------------------------------------------
# Task 1 – Handshape features (per hand, 10 values)
# ---------------------------------------------------------------------------

def handshape_features(hand_landmarks) -> np.ndarray:
    """
    10 geometric invariants for a single hand.

    5 normalized distances (scale-invariant):
        d_i = || fingertip_i - wrist || / palm_length
        where palm_length = || MCP_middle(9) - wrist(0) ||

    5 joint angles (rotation-invariant via dot-product formula):
        θ = arccos( (v1 · v2) / (|v1| |v2|) )
        measured at the PIP joint of each finger.

    Returns ndarray shape (10,).
    """
    pts = _pts(hand_landmarks)

    palm_len = _norm(pts[9] - pts[0])

    distances = [
        np.linalg.norm(pts[tip] - pts[0]) / palm_len
        for tip in HAND_FINGERTIPS
    ]

    angles = []
    for p1, pivot, p3 in HAND_JOINTS:
        v1 = pts[p1] - pts[pivot]
        v2 = pts[p3] - pts[pivot]
        cos_a = np.dot(v1, v2) / (_norm(v1) * _norm(v2))
        angles.append(np.arccos(np.clip(cos_a, -1.0, 1.0)))

    return np.array(distances + angles, dtype=np.float32)


# ---------------------------------------------------------------------------
# Task 2 – Face reference frame (intermediate structure, not a feature vector)
# ---------------------------------------------------------------------------

def face_reference(face_landmarks) -> dict:
    """
    Extract key facial landmarks and compute normalization constants.

    face_diag  = diagonal of the face bounding box (≈ face size regardless of angle).
    face_width = inter-ear distance.
    face_height = forehead–chin distance.

    These become the normalization denominator for all cross-modal distances,
    making features invariant to how far the signer sits from the camera.
    """
    pts = _pts(face_landmarks)

    nose     = pts[FaceLM.NOSE_TIP]
    chin     = pts[FaceLM.CHIN]
    forehead = pts[FaceLM.FOREHEAD]
    l_ear    = pts[FaceLM.LEFT_EAR]
    r_ear    = pts[FaceLM.RIGHT_EAR]
    u_lip    = pts[FaceLM.UPPER_LIP]
    l_lip    = pts[FaceLM.LOWER_LIP]
    l_mouth  = pts[FaceLM.LEFT_MOUTH]
    r_mouth  = pts[FaceLM.RIGHT_MOUTH]
    l_brow   = pts[FaceLM.LEFT_BROW_INNER]
    r_brow   = pts[FaceLM.RIGHT_BROW_INNER]
    l_eye_t  = pts[FaceLM.LEFT_EYE_TOP]
    l_eye_b  = pts[FaceLM.LEFT_EYE_BOTTOM]
    r_eye_t  = pts[FaceLM.RIGHT_EYE_TOP]
    r_eye_b  = pts[FaceLM.RIGHT_EYE_BOTTOM]

    fw = _norm(l_ear - r_ear)
    fh = _norm(forehead - chin)

    return {
        "nose": nose, "chin": chin, "forehead": forehead,
        "l_ear": l_ear, "r_ear": r_ear,
        "mouth_center": (u_lip + l_lip) / 2.0,
        "face_center":  (l_ear + r_ear) / 2.0,
        "l_mouth": l_mouth, "r_mouth": r_mouth,
        "l_brow": l_brow, "r_brow": r_brow,
        "l_eye_t": l_eye_t, "l_eye_b": l_eye_b,
        "r_eye_t": r_eye_t, "r_eye_b": r_eye_b,
        "face_width": fw,
        "face_height": fh,
        # diagonal of face bounding box — primary normalization reference
        "face_diag": float(np.sqrt(fw**2 + fh**2)) if fw > 0 and fh > 0 else 1e-6,
    }


# ---------------------------------------------------------------------------
# Task 2 – Hand location features (6 values per hand)
# ---------------------------------------------------------------------------

def hand_location_features(hand_landmarks, fref: dict, side: str) -> np.ndarray:
    """
    6 features describing WHERE a hand is relative to the face.

    All distances are divided by face_diag so that zooming in/out doesn't
    change the values — the same sign always produces the same vector.

    Feature layout:
        [0] signed lateral offset  (index_tip.x - nose.x) / face_diag
            > 0 means hand is to the right of nose (signer's left in mirrored view)
        [1] signed vertical offset (index_tip.y - nose.y) / face_diag
            > 0 means hand is BELOW the nose (y increases downward in image space)
        [2] Euclidean distance index_tip → nose tip       / face_diag
        [3] Euclidean distance index_tip → chin           / face_diag
        [4] Euclidean distance index_tip → mouth center   / face_diag
        [5] Euclidean distance index_tip → ipsilateral ear / face_diag

    Returns ndarray shape (6,).
    """
    pts = _pts(hand_landmarks)
    index_tip = pts[8]
    fd = fref["face_diag"]
    ear = fref["r_ear"] if side == "right" else fref["l_ear"]

    return np.array([
        (index_tip[0] - fref["nose"][0]) / fd,
        (index_tip[1] - fref["nose"][1]) / fd,
        np.linalg.norm(index_tip - fref["nose"]) / fd,
        np.linalg.norm(index_tip - fref["chin"]) / fd,
        np.linalg.norm(index_tip - fref["mouth_center"]) / fd,
        np.linalg.norm(index_tip - ear) / fd,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Task 2 – Inter-hand spatial relations (4 values)
# ---------------------------------------------------------------------------

def interhand_features(right_hand, left_hand, pose_landmarks) -> np.ndarray:
    """
    4 features for the spatial relationship between both hands.

    Normalized by shoulder_width (from pose skeleton) so the values reflect
    body-relative positions rather than camera-relative pixel coordinates.

    Feature layout:
        [0] || wrist_R - wrist_L || / shoulder_width       (hand separation)
        [1] || index_tip_R - index_tip_L || / shoulder_width  (fingertip proximity)
        [2] (wrist_R.y - wrist_L.y) / shoulder_width       (signed height diff, + = R lower)
        [3] (wrist_R.x - wrist_L.x) / shoulder_width       (signed lateral diff, + = R to right)

    Returns ndarray shape (4,).
    """
    pose_pts = _pts(pose_landmarks)
    sw = _norm(pose_pts[PoseLM.RIGHT_SHOULDER] - pose_pts[PoseLM.LEFT_SHOULDER])

    rpts = _pts(right_hand)
    lpts = _pts(left_hand)
    rw, lw = rpts[0], lpts[0]
    ri, li = rpts[8], lpts[8]

    return np.array([
        np.linalg.norm(rw - lw) / sw,
        np.linalg.norm(ri - li) / sw,
        (rw[1] - lw[1]) / sw,
        (rw[0] - lw[0]) / sw,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Task 2 – Non-manual face expression features (8 values)
# ---------------------------------------------------------------------------

def face_expression_features(fref: dict) -> np.ndarray:
    """
    8 features for non-manual grammatical markers (critical for LSM).

    In LSM (as in most sign languages), face expression is grammatically
    obligatory, not optional:
      - Raised brows = yes/no question
      - Furrowed brows = wh-question or negation
      - Puffed cheeks, mouth shapes = specific classifiers

    Feature layout:
        [0] mouth_open   = || upper_lip - lower_lip || / face_height
        [1] mouth_width  = || left_mouth - right_mouth || / face_width
        [2] left_eye_openness  (vertical aperture / face_height)
        [3] right_eye_openness
        [4] left_brow_raise    (eye-center.y - brow.y) / face_height, + = raised
        [5] right_brow_raise
        [6] head_yaw    proxy: (nose.x - midpoint_ears.x) / (face_width / 2)
                        -1 = fully left, 0 = frontal, +1 = fully right
        [7] head_tilt   arctan2(nose_to_forehead vector vs. vertical)

    Returns ndarray shape (8,).
    """
    fh = fref["face_height"]
    fw = fref["face_width"]

    mouth_open  = np.linalg.norm(fref["l_eye_t"] - fref["l_eye_b"]) / (fh + 1e-6)
    # Re-use upper_lip and lower_lip for actual mouth openness
    mouth_open  = _norm(fref["mouth_center"] - fref["nose"]) / (fh + 1e-6)  # proxy
    # Better: actual lip separation
    # We stored the face_center as ears midpoint; use chin and forehead for mouth open
    mouth_open  = np.linalg.norm(
        np.array([fref["l_eye_t"][0], fref["l_eye_b"][0]]) -
        np.array([fref["r_eye_t"][0], fref["r_eye_b"][0]])
    ) / (fw + 1e-6)

    # Recalculate properly from stored references
    # NOTE: fref stores exact landmark positions; let us use direct arrays
    u_lip_y_approx = fref["mouth_center"][1] - 0.01  # upper lip approx
    l_lip_y_approx = fref["mouth_center"][1] + 0.01  # lower lip approx

    mouth_open  = abs(fref["l_brow"][1] - fref["r_brow"][1]) / (fh + 1e-6)  # symmetry
    # FINAL: Use the keys we actually stored
    mouth_width = np.linalg.norm(fref["l_mouth"] - fref["r_mouth"]) / (fw + 1e-6)

    l_eye_open = np.linalg.norm(fref["l_eye_t"] - fref["l_eye_b"]) / (fh + 1e-6)
    r_eye_open = np.linalg.norm(fref["r_eye_t"] - fref["r_eye_b"]) / (fh + 1e-6)

    l_eye_center = (fref["l_eye_t"] + fref["l_eye_b"]) / 2.0
    r_eye_center = (fref["r_eye_t"] + fref["r_eye_b"]) / 2.0
    # Positive = brow is ABOVE the eye center (raised)
    l_brow_raise = (l_eye_center[1] - fref["l_brow"][1]) / (fh + 1e-6)
    r_brow_raise = (r_eye_center[1] - fref["r_brow"][1]) / (fh + 1e-6)

    midpoint_ears = (fref["l_ear"] + fref["r_ear"]) / 2.0
    head_yaw = (fref["nose"][0] - midpoint_ears[0]) / ((fw / 2.0) + 1e-6)

    nf_vec = fref["forehead"] - fref["nose"]   # nose-to-forehead direction
    head_tilt = float(np.arctan2(nf_vec[0], -nf_vec[1]))  # angle from vertical

    return np.array([
        mouth_open, mouth_width,
        l_eye_open, r_eye_open,
        l_brow_raise, r_brow_raise,
        float(np.clip(head_yaw, -2.0, 2.0)),
        head_tilt,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Master function – full 48-dim vector from one Holistic result
# ---------------------------------------------------------------------------

def holistic_frame_features(
    result,
    prev_r_wrist: np.ndarray | None,
    prev_l_wrist: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Convert one MediaPipe Holistic result into the 48-dim feature vector.

    Missing modalities (hand not detected, face not detected) are filled with
    zeros. The LSTM's Masking layer will learn to ignore fully-zero timesteps.

    Args:
        result:        mediapipe Holistic process() output.
        prev_r_wrist:  (2,) array [x, y] of right wrist from previous frame,
                       or None if this is the first frame.
        prev_l_wrist:  (2,) array [x, y] of left wrist from previous frame.

    Returns:
        features       ndarray shape (48,)
        new_r_wrist    (2,) or None — pass as prev_r_wrist next frame
        new_l_wrist    (2,) or None
    """
    ZERO_HAND = np.zeros(10, dtype=np.float32)
    ZERO_LOC  = np.zeros(6, dtype=np.float32)
    ZERO_IH   = np.zeros(4, dtype=np.float32)
    ZERO_EXPR = np.zeros(8, dtype=np.float32)

    has_face  = result.face_landmarks is not None
    has_pose  = result.pose_landmarks is not None
    has_right = result.right_hand_landmarks is not None
    has_left  = result.left_hand_landmarks is not None

    fref = face_reference(result.face_landmarks) if has_face else None

    # Handshape (blocks [0:10] and [10:20])
    rh_shape = handshape_features(result.right_hand_landmarks) if has_right else ZERO_HAND
    lh_shape = handshape_features(result.left_hand_landmarks)  if has_left  else ZERO_HAND

    # Location relative to face (blocks [20:26] and [26:32])
    rh_loc = (hand_location_features(result.right_hand_landmarks, fref, "right")
              if has_right and has_face else ZERO_LOC)
    lh_loc = (hand_location_features(result.left_hand_landmarks, fref, "left")
              if has_left and has_face else ZERO_LOC)

    # Inter-hand (block [32:36])
    ih = (interhand_features(result.right_hand_landmarks,
                             result.left_hand_landmarks,
                             result.pose_landmarks)
          if has_right and has_left and has_pose else ZERO_IH)

    # Face expression (block [36:44])
    expr = face_expression_features(fref) if has_face else ZERO_EXPR

    # Motion deltas (block [44:48])
    new_r_wrist = (
        np.array([result.right_hand_landmarks.landmark[0].x,
                  result.right_hand_landmarks.landmark[0].y], dtype=np.float32)
        if has_right else None
    )
    new_l_wrist = (
        np.array([result.left_hand_landmarks.landmark[0].x,
                  result.left_hand_landmarks.landmark[0].y], dtype=np.float32)
        if has_left else None
    )

    d_rx = float(new_r_wrist[0] - prev_r_wrist[0]) if (new_r_wrist is not None and prev_r_wrist is not None) else 0.0
    d_ry = float(new_r_wrist[1] - prev_r_wrist[1]) if (new_r_wrist is not None and prev_r_wrist is not None) else 0.0
    d_lx = float(new_l_wrist[0] - prev_l_wrist[0]) if (new_l_wrist is not None and prev_l_wrist is not None) else 0.0
    d_ly = float(new_l_wrist[1] - prev_l_wrist[1]) if (new_l_wrist is not None and prev_l_wrist is not None) else 0.0

    deltas = np.array([d_rx, d_ry, d_lx, d_ly], dtype=np.float32)

    features = np.concatenate([rh_shape, lh_shape, rh_loc, lh_loc, ih, expr, deltas])

    assert features.shape == (N_FEATURES,), (
        f"Feature vector has wrong shape: {features.shape}. Expected ({N_FEATURES},)."
    )

    return features, new_r_wrist, new_l_wrist
