import numpy as np

from face_reconstruction.landmarks import (
    load_landmarks,
    save_landmarks,
    validate_landmarks,
)


def test_validate_landmarks_accepts_valid_array():
    landmarks = np.column_stack(
        [
            np.linspace(10, 90, 68),
            np.linspace(20, 80, 68),
        ]
    )

    valid, reason = validate_landmarks(
        landmarks,
        image_shape=(100, 100, 3),
    )

    assert valid is True
    assert reason == "ok"


def test_validate_landmarks_rejects_wrong_shape():
    landmarks = np.zeros((67, 2))

    valid, reason = validate_landmarks(
        landmarks,
        image_shape=(100, 100, 3),
    )

    assert valid is False
    assert reason == "unexpected_shape"


def test_validate_landmarks_rejects_points_outside_image():
    landmarks = np.zeros((68, 2))
    landmarks[:, 0] = 150
    landmarks[:, 1] = 50

    valid, reason = validate_landmarks(
        landmarks,
        image_shape=(100, 100, 3),
    )

    assert valid is False
    assert reason == "x_outside_image"


def test_save_and_load_landmarks(tmp_path):
    landmarks = np.random.default_rng(42).random((68, 2))
    output_path = tmp_path / "face.npy"

    save_landmarks(landmarks, output_path)
    loaded = load_landmarks(output_path)

    assert output_path.exists()
    assert loaded.shape == (68, 2)
    assert np.allclose(loaded, landmarks)