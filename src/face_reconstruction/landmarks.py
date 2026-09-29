from pathlib import Path

import numpy as np


def validate_landmarks(
    landmarks: np.ndarray,
    image_shape: tuple[int, ...],
    expected_n: int = 68,
) -> tuple[bool, str]:
    landmarks = np.asarray(
        landmarks,
        dtype=np.float32,
    )

    if landmarks.shape != (expected_n, 2):
        return False, "unexpected_shape"

    if not np.isfinite(landmarks).all():
        return False, "non_finite_coordinates"

    image_height, image_width = image_shape[:2]

    x = landmarks[:, 0]
    y = landmarks[:, 1]

    if (x < 0).any() or (x >= image_width).any():
        return False, "x_outside_image"

    if (y < 0).any() or (y >= image_height).any():
        return False, "y_outside_image"

    landmark_width = float(np.ptp(x))
    landmark_height = float(np.ptp(y))

    if landmark_width < 0.10 * image_width:
        return False, "face_bbox_too_narrow"

    if landmark_height < 0.10 * image_height:
        return False, "face_bbox_too_short"

    if landmark_width > 0.80 * image_width:
        return False, "face_bbox_too_wide"

    if landmark_height > 0.90 * image_height:
        return False, "face_bbox_too_tall"

    left_eye_center = landmarks[36:42].mean(axis=0)
    right_eye_center = landmarks[42:48].mean(axis=0)

    interocular_distance = float(
        np.linalg.norm(
            right_eye_center - left_eye_center
        )
    )

    if interocular_distance < 0.05 * image_width:
        return False, "interocular_distance_too_small"

    if interocular_distance > 0.50 * image_width:
        return False, "interocular_distance_too_large"

    eye_vertical_difference = abs(
        float(
            left_eye_center[1]
            - right_eye_center[1]
        )
    )

    if eye_vertical_difference > 0.20 * interocular_distance:
        return False, "eyes_excessively_tilted"

    nose_center = landmarks[27:36].mean(axis=0)
    mouth_center = landmarks[48:68].mean(axis=0)

    if nose_center[1] <= left_eye_center[1]:
        return False, "nose_not_below_eyes"

    if mouth_center[1] <= nose_center[1]:
        return False, "mouth_not_below_nose"

    consecutive_distances = np.linalg.norm(
        np.diff(landmarks, axis=0),
        axis=1,
    )

    median_distance = float(
        np.median(consecutive_distances)
    )

    if median_distance <= 0:
        return False, "degenerate_landmarks"

    extreme_jump = float(
        consecutive_distances.max()
    )

    if extreme_jump > 12 * median_distance:
        return False, "extreme_landmark_jump"

    return True, "ok"
def save_landmarks(
    landmarks: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_path,
        np.asarray(landmarks, dtype=np.float32),
    )


def load_landmarks(
    landmark_path: Path,
) -> np.ndarray:
    landmarks = np.load(landmark_path)

    if landmarks.ndim != 2 or landmarks.shape[1] != 2:
        raise ValueError(
            f"Invalid landmark array shape: {landmarks.shape}"
        )

    return landmarks.astype(np.float32)