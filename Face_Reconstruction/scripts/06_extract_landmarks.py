from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


DEFAULT_STIMULI_PATH = Path("data/processed/stimuli.csv")
DEFAULT_OUTPUT_DIR = Path("data/interim/landmarks_mediapipe")
DEFAULT_MODEL_PATH = Path("models/face_landmarker.task")
DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
LANDMARK_COUNT = 478


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract MediaPipe Face Landmarker landmarks from selected stimuli "
            "and create a quality-control manifest."
        )
    )
    parser.add_argument(
        "--stimuli-path",
        type=Path,
        default=DEFAULT_STIMULI_PATH,
        help=f"Stimuli CSV. Default: {DEFAULT_STIMULI_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"MediaPipe .task model bundle. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--download-model",
        action="store_true",
        help=(
            "Download the official Face Landmarker model bundle when "
            "--model-path does not exist."
        ),
    )
    parser.add_argument(
        "--all-stimuli",
        action="store_true",
        help="Process every row instead of selected_for_study == true.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Process only the first N selected stimuli.",
    )
    parser.add_argument(
        "--max-num-faces",
        type=int,
        default=1,
        help=(
            "Maximum faces returned by MediaPipe. CFD images should contain "
            "one face, so the default is 1."
        ),
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.7,
        help="MediaPipe face detection threshold. Default: 0.7.",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.7,
        help="MediaPipe tracking threshold. Default: 0.7.",
    )
    parser.add_argument(
        "--min-face-presence-confidence",
        type=float,
        default=0.7,
        help="MediaPipe face-presence threshold. Default: 0.7.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npy arrays.",
    )
    parser.add_argument(
        "--fail-on-qc-error",
        action="store_true",
        help="Return exit status 1 if at least one image fails QC.",
    )
    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.max_images is not None and arguments.max_images <= 0:
        raise ValueError("--max-images must be strictly positive.")
    if arguments.max_num_faces <= 0:
        raise ValueError("--max-num-faces must be strictly positive.")
    for name in (
        "min_detection_confidence",
        "min_face_presence_confidence",
        "min_tracking_confidence",
    ):
        value = float(getattr(arguments, name))
        if not 0.0 <= value <= 1.0:
            flag = "--" + name.replace("_", "-")
            raise ValueError(f"{flag} must be between 0 and 1.")


def parse_boolean_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int).eq(1)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "oui"})
    )


def read_stimuli(
    stimuli_path: Path,
    *,
    selected_only: bool,
    max_images: int | None,
) -> pd.DataFrame:
    if not stimuli_path.exists():
        raise FileNotFoundError(f"Stimuli file not found: {stimuli_path}")

    stimuli = pd.read_csv(stimuli_path)
    required = {"face_id", "image_path"}
    missing = required.difference(stimuli.columns)
    if missing:
        raise ValueError(
            f"The stimuli file is missing columns: {sorted(missing)}"
        )

    if selected_only:
        if "selected_for_study" not in stimuli.columns:
            raise ValueError(
                "Column 'selected_for_study' is required unless "
                "--all-stimuli is used."
            )
        stimuli = stimuli.loc[
            parse_boolean_series(stimuli["selected_for_study"])
        ].copy()

    stimuli["face_id"] = stimuli["face_id"].astype(str)
    stimuli["image_path"] = stimuli["image_path"].astype(str)

    if stimuli.empty:
        raise ValueError("No stimuli were selected for landmark extraction.")
    if stimuli["face_id"].duplicated().any():
        duplicates = sorted(
            stimuli.loc[
                stimuli["face_id"].duplicated(keep=False), "face_id"
            ].unique()
        )
        raise ValueError(f"Duplicate face identifiers: {duplicates}")

    stimuli = stimuli.reset_index(drop=True)
    if max_images is not None:
        stimuli = stimuli.head(max_images).copy()
    return stimuli


def ensure_model(
    model_path: Path,
    *,
    download_model: bool,
) -> Path:
    """Resolve the .task bundle and optionally download the official model."""
    model_path = model_path.expanduser().resolve()
    if model_path.exists():
        if not model_path.is_file():
            raise ValueError(f"Model path is not a file: {model_path}")
        return model_path

    if not download_model:
        raise FileNotFoundError(
            f"MediaPipe model not found: {model_path}\n"
            "Download it by adding --download-model, or pass an existing "
            "bundle with --model-path."
        )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = model_path.with_suffix(model_path.suffix + ".part")
    print(f"Downloading Face Landmarker model to: {model_path}")
    try:
        urllib.request.urlretrieve(DEFAULT_MODEL_URL, temporary_path)
        temporary_path.replace(model_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return model_path


def build_detector(
    arguments: argparse.Namespace,
    *,
    model_path: Path,
) -> Any:
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path),
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=arguments.max_num_faces,
        min_face_detection_confidence=(
            arguments.min_detection_confidence
        ),
        min_face_presence_confidence=(
            arguments.min_face_presence_confidence
        ),
        min_tracking_confidence=arguments.min_tracking_confidence,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def normalized_to_pixel_coordinates(
    face_landmarks: Any,
    *,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Convert MediaPipe normalized x/y values to original-image pixels."""
    coordinates = np.asarray(
        [
            (
                landmark.x * (image_width - 1),
                landmark.y * (image_height - 1),
            )
            for landmark in face_landmarks
        ],
        dtype=np.float32,
    )
    return coordinates


def select_central_face(
    candidates: list[np.ndarray],
    *,
    image_width: int,
    image_height: int,
) -> tuple[np.ndarray | None, int, int | None]:
    """Select the face closest to image centre; break ties by larger area."""
    if not candidates:
        return None, 0, None

    image_center = np.array(
        [image_width / 2.0, image_height / 2.0], dtype=np.float32
    )
    image_diagonal = float(np.hypot(image_width, image_height))
    scores: list[float] = []

    for landmarks in candidates:
        face_center = landmarks.mean(axis=0)
        centre_distance = float(
            np.linalg.norm(face_center - image_center) / image_diagonal
        )
        width = float(np.ptp(landmarks[:, 0]))
        height = float(np.ptp(landmarks[:, 1]))
        area_ratio = (width * height) / float(image_width * image_height)
        scores.append(centre_distance - 0.10 * area_ratio)

    selected_index = int(np.argmin(scores))
    return candidates[selected_index], len(candidates), selected_index


def extract_landmarks_from_image(
    detector: Any,
    image_bgr: np.ndarray,
) -> tuple[np.ndarray | None, int, int | None]:
    image_height, image_width = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = np.ascontiguousarray(image_rgb)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb,
    )
    results = detector.detect(mp_image)

    detected_faces = results.face_landmarks or []
    candidates = [
        normalized_to_pixel_coordinates(
            face,
            image_width=image_width,
            image_height=image_height,
        )
        for face in detected_faces
    ]
    return select_central_face(
        candidates,
        image_width=image_width,
        image_height=image_height,
    )


def validate_mediapipe_landmarks(
    landmarks: np.ndarray,
    image_shape: tuple[int, ...],
    *,
    expected_n: int,
) -> tuple[bool, str]:
    """Basic numerical and anatomical checks for MediaPipe topology."""
    landmarks = np.asarray(landmarks, dtype=np.float32)
    if landmarks.shape != (expected_n, 2):
        return False, "unexpected_shape"
    if not np.isfinite(landmarks).all():
        return False, "non_finite_coordinates"

    image_height, image_width = image_shape[:2]
    x = landmarks[:, 0]
    y = landmarks[:, 1]
    margin_x = 0.02 * image_width
    margin_y = 0.02 * image_height
    if (x < -margin_x).any() or (x > image_width - 1 + margin_x).any():
        return False, "x_far_outside_image"
    if (y < -margin_y).any() or (y > image_height - 1 + margin_y).any():
        return False, "y_far_outside_image"

    face_width = float(np.ptp(x))
    face_height = float(np.ptp(y))
    if face_width < 0.12 * image_width:
        return False, "face_bbox_too_narrow"
    if face_height < 0.12 * image_height:
        return False, "face_bbox_too_short"
    if face_width > 0.95 * image_width:
        return False, "face_bbox_too_wide"
    if face_height > 0.98 * image_height:
        return False, "face_bbox_too_tall"

    # Stable MediaPipe Face Mesh indices (valid for 468 and 478 points).
    left_eye = landmarks[[33, 133, 159, 145]].mean(axis=0)
    right_eye = landmarks[[362, 263, 386, 374]].mean(axis=0)
    eyes_y = float((left_eye[1] + right_eye[1]) / 2.0)
    interocular = float(np.linalg.norm(right_eye - left_eye))
    if interocular < 0.08 * image_width:
        return False, "interocular_distance_too_small"
    if interocular > 0.55 * image_width:
        return False, "interocular_distance_too_large"
    if abs(float(left_eye[1] - right_eye[1])) > 0.20 * interocular:
        return False, "eyes_excessively_tilted"

    forehead_y = float(landmarks[10, 1])
    nose_y = float(landmarks[1, 1])
    mouth_y = float(landmarks[[13, 14], 1].mean())
    chin_y = float(landmarks[152, 1])
    if not forehead_y < eyes_y < nose_y < mouth_y < chin_y:
        return False, "implausible_vertical_anatomy"

    return True, "ok"


def save_landmarks(landmarks: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(landmarks, dtype=np.float32), allow_pickle=False)


def make_manifest_row(
    *,
    face_id: str,
    image_path: Path,
    landmark_path: Path | None,
    landmark_detected: bool,
    detection_count: int,
    selected_face_index: int | None,
    n_landmarks: int,
    image_width: int | None,
    image_height: int | None,
    qc_pass: bool,
    qc_reason: str,
    model_path: Path,
    reused_existing_file: bool = False,
    error_type: str = "",
    error_message: str = "",
) -> dict[str, Any]:
    return {
        "face_id": face_id,
        "detector": "mediapipe_face_landmarker_tasks",
        "mediapipe_version": getattr(mp, "__version__", "unknown"),
        "model_path": model_path.as_posix(),
        "image_path": image_path.as_posix(),
        "landmark_path": landmark_path.as_posix() if landmark_path else "",
        "landmark_file_exists": bool(landmark_path and landmark_path.exists()),
        "landmark_detected": landmark_detected,
        "detection_count": detection_count,
        "multiple_faces_detected": detection_count > 1,
        "selected_face_index": selected_face_index,
        "n_landmarks": n_landmarks,
        "image_width": image_width,
        "image_height": image_height,
        "qc_pass": qc_pass,
        "qc_reason": qc_reason,
        "reused_existing_file": reused_existing_file,
        "error_type": error_type,
        "error_message": error_message,
    }


def process_stimulus(
    row: Any,
    *,
    detector: Any,
    arrays_dir: Path,
    overwrite: bool,
    expected_n: int,
    model_path: Path,
) -> dict[str, Any]:
    face_id = str(row.face_id)
    image_path = Path(str(row.image_path))
    landmark_path = arrays_dir / f"{face_id}.npy"
    common = {
        "face_id": face_id,
        "image_path": image_path,
        "model_path": model_path,
    }

    if not image_path.exists():
        return make_manifest_row(
            **common,
            landmark_path=None,
            landmark_detected=False,
            detection_count=0,
            selected_face_index=None,
            n_landmarks=0,
            image_width=None,
            image_height=None,
            qc_pass=False,
            qc_reason="image_file_missing",
        )

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return make_manifest_row(
            **common,
            landmark_path=None,
            landmark_detected=False,
            detection_count=0,
            selected_face_index=None,
            n_landmarks=0,
            image_width=None,
            image_height=None,
            qc_pass=False,
            qc_reason="image_read_failed",
        )

    image_height, image_width = image.shape[:2]
    dimensions = {"image_width": image_width, "image_height": image_height}

    if landmark_path.exists() and not overwrite:
        try:
            existing = np.load(landmark_path, allow_pickle=False)
            qc_pass, qc_reason = validate_mediapipe_landmarks(
                existing, image.shape, expected_n=expected_n
            )
            return make_manifest_row(
                **common,
                **dimensions,
                landmark_path=landmark_path,
                landmark_detected=True,
                detection_count=1,
                selected_face_index=0,
                n_landmarks=int(existing.shape[0]),
                qc_pass=qc_pass,
                qc_reason=(
                    "existing_file_ok"
                    if qc_pass
                    else f"existing_file_{qc_reason}"
                ),
                reused_existing_file=True,
            )
        except Exception as error:
            return make_manifest_row(
                **common,
                **dimensions,
                landmark_path=landmark_path,
                landmark_detected=False,
                detection_count=0,
                selected_face_index=None,
                n_landmarks=0,
                qc_pass=False,
                qc_reason="existing_landmark_read_failed",
                reused_existing_file=True,
                error_type=type(error).__name__,
                error_message=str(error),
            )

    try:
        landmarks, detection_count, selected_index = (
            extract_landmarks_from_image(detector, image)
        )
    except Exception as error:
        return make_manifest_row(
            **common,
            **dimensions,
            landmark_path=None,
            landmark_detected=False,
            detection_count=0,
            selected_face_index=None,
            n_landmarks=0,
            qc_pass=False,
            qc_reason="detector_exception",
            error_type=type(error).__name__,
            error_message=str(error),
        )

    if landmarks is None:
        return make_manifest_row(
            **common,
            **dimensions,
            landmark_path=None,
            landmark_detected=False,
            detection_count=detection_count,
            selected_face_index=None,
            n_landmarks=0,
            qc_pass=False,
            qc_reason="no_face_detected",
        )

    qc_pass, qc_reason = validate_mediapipe_landmarks(
        landmarks, image.shape, expected_n=expected_n
    )
    if qc_pass:
        # Very small excursions can occur through normalized coordinates.
        landmarks[:, 0] = np.clip(landmarks[:, 0], 0, image_width - 1)
        landmarks[:, 1] = np.clip(landmarks[:, 1], 0, image_height - 1)
        save_landmarks(landmarks, landmark_path)
        saved_path: Path | None = landmark_path
        if detection_count > 1:
            qc_reason = "multiple_faces_central_selected"
    else:
        saved_path = None

    return make_manifest_row(
        **common,
        **dimensions,
        landmark_path=saved_path,
        landmark_detected=True,
        detection_count=detection_count,
        selected_face_index=selected_index,
        n_landmarks=int(landmarks.shape[0]),
        qc_pass=qc_pass,
        qc_reason=qc_reason,
    )


def print_summary(manifest: pd.DataFrame, manifest_path: Path) -> None:
    total = len(manifest)
    detected = int(manifest["landmark_detected"].sum())
    passed = int(manifest["qc_pass"].sum())
    multiple = int(manifest["multiple_faces_detected"].sum())
    print("\nMediaPipe landmark extraction completed")
    print(f"Stimuli processed: {total}")
    print(f"Landmarks detected: {detected}/{total}")
    print(f"QC passed: {passed}/{total}")
    print(f"Multiple-face detections: {multiple}")
    print(f"Manifest saved to: {manifest_path}")
    print("\nQC reasons:")
    print(manifest["qc_reason"].value_counts(dropna=False).to_string())

    failed = manifest.loc[
        ~manifest["qc_pass"],
        ["face_id", "image_path", "qc_reason", "error_type", "error_message"],
    ]
    if not failed.empty:
        print("\nFailed rows:")
        print(failed.to_string(index=False))


def main() -> int:
    arguments = parse_arguments()
    validate_arguments(arguments)
    model_path = ensure_model(
        arguments.model_path,
        download_model=arguments.download_model,
    )
    expected_n = LANDMARK_COUNT

    output_dir = arguments.output_dir
    arrays_dir = output_dir / "arrays"
    manifest_path = output_dir / "landmarks_manifest.csv"
    arrays_dir.mkdir(parents=True, exist_ok=True)

    stimuli = read_stimuli(
        arguments.stimuli_path,
        selected_only=not arguments.all_stimuli,
        max_images=arguments.max_images,
    )
    print(
        "Loading MediaPipe Face Landmarker Tasks "
        f"({expected_n} landmarks, max faces={arguments.max_num_faces})"
    )
    print(f"MediaPipe version: {getattr(mp, '__version__', 'unknown')}")
    print(f"Model: {model_path}")

    manifest_rows: list[dict[str, Any]] = []
    with build_detector(arguments, model_path=model_path) as detector:
        total = len(stimuli)
        for position, row in enumerate(
            stimuli.itertuples(index=False), start=1
        ):
            print(f"[{position:03d}/{total:03d}] Processing {row.face_id}")
            manifest_rows.append(
                process_stimulus(
                    row,
                    detector=detector,
                    arrays_dir=arrays_dir,
                    overwrite=arguments.overwrite,
                    expected_n=expected_n,
                    model_path=model_path,
                )
            )

    manifest = pd.DataFrame(manifest_rows).sort_values(
        "face_id"
    ).reset_index(drop=True)
    manifest.to_csv(manifest_path, index=False)
    print_summary(manifest, manifest_path)

    has_failures = bool((~manifest["qc_pass"]).any())
    return 1 if arguments.fail_on_qc_error and has_failures else 0


if __name__ == "__main__":
    sys.exit(main())
