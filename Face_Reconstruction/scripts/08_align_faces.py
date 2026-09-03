from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


DEFAULT_MANIFEST_PATH = Path(
    "data/interim/landmarks_mediapipe/landmarks_manifest.csv"
)
DEFAULT_OUTPUT_DIR = Path("data/interim/aligned_faces")
EXPECTED_LANDMARK_COUNT = 478

# Stable MediaPipe Face Mesh indices. The groups are deliberately averaged so
# that the alignment is less sensitive to a single imperfect landmark.
EYE_GROUP_A = np.array([33, 133, 159, 145], dtype=np.int32)
EYE_GROUP_B = np.array([362, 263, 386, 374], dtype=np.int32)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Align images and 478-point MediaPipe landmarks to a common "
            "canvas using an eye-based similarity transform."
        )
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=f"Landmark manifest. Default: {DEFAULT_MANIFEST_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--output-width",
        type=int,
        default=512,
        help="Aligned canvas width in pixels. Default: 512.",
    )
    parser.add_argument(
        "--output-height",
        type=int,
        default=512,
        help="Aligned canvas height in pixels. Default: 512.",
    )
    parser.add_argument(
        "--left-eye-x",
        type=float,
        default=0.35,
        help="Target x of the image-left eye, as a width fraction. Default: 0.35.",
    )
    parser.add_argument(
        "--right-eye-x",
        type=float,
        default=0.65,
        help="Target x of the image-right eye, as a width fraction. Default: 0.65.",
    )
    parser.add_argument(
        "--eye-y",
        type=float,
        default=0.38,
        help="Target y of both eyes, as a height fraction. Default: 0.38.",
    )
    parser.add_argument(
        "--expected-landmarks",
        type=int,
        default=EXPECTED_LANDMARK_COUNT,
        help=f"Expected number of landmarks. Default: {EXPECTED_LANDMARK_COUNT}.",
    )
    parser.add_argument(
        "--min-in-bounds-fraction",
        type=float,
        default=0.98,
        help=(
            "Minimum fraction of transformed landmarks inside the canvas for "
            "alignment QC. Default: 0.98."
        ),
    )
    parser.add_argument(
        "--eye-error-tolerance",
        type=float,
        default=0.5,
        help="Maximum target-eye error in pixels for alignment QC. Default: 0.5.",
    )
    parser.add_argument(
        "--border-mode",
        choices=("reflect101", "constant"),
        default="reflect101",
        help="OpenCV border handling outside the source image. Default: reflect101.",
    )
    parser.add_argument(
        "--include-qc-failures",
        action="store_true",
        help="Also align manifest rows whose landmark QC did not pass.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Process only the first N eligible rows after sorting by face_id.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing aligned images and landmark arrays.",
    )
    parser.add_argument(
        "--fail-on-qc-error",
        action="store_true",
        help="Return exit status 1 if at least one alignment fails QC.",
    )
    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.output_width < 2 or arguments.output_height < 2:
        raise ValueError("Output width and height must both be at least 2.")
    if arguments.expected_landmarks <= int(max(EYE_GROUP_A.max(), EYE_GROUP_B.max())):
        raise ValueError("--expected-landmarks is incompatible with the eye indices.")
    if arguments.max_images is not None and arguments.max_images <= 0:
        raise ValueError("--max-images must be strictly positive.")
    if not 0.0 < arguments.left_eye_x < arguments.right_eye_x < 1.0:
        raise ValueError(
            "Eye x targets must satisfy 0 < left-eye-x < right-eye-x < 1."
        )
    if not 0.0 < arguments.eye_y < 1.0:
        raise ValueError("--eye-y must be strictly between 0 and 1.")
    if not 0.0 <= arguments.min_in_bounds_fraction <= 1.0:
        raise ValueError("--min-in-bounds-fraction must be between 0 and 1.")
    if arguments.eye_error_tolerance < 0.0:
        raise ValueError("--eye-error-tolerance cannot be negative.")


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


def read_manifest(
    manifest_path: Path,
    *,
    include_qc_failures: bool,
    max_images: int | None,
) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Landmark manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    required = {"face_id", "image_path", "landmark_path"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(
            f"The landmark manifest is missing columns: {sorted(missing)}"
        )

    manifest["face_id"] = manifest["face_id"].astype(str).str.strip()
    manifest["image_path"] = manifest["image_path"].fillna("").astype(str)
    manifest["landmark_path"] = (
        manifest["landmark_path"].fillna("").astype(str)
    )
    if manifest["face_id"].eq("").any():
        raise ValueError("The landmark manifest contains an empty face_id.")
    if manifest["face_id"].duplicated().any():
        duplicates = sorted(
            manifest.loc[
                manifest["face_id"].duplicated(keep=False), "face_id"
            ].unique()
        )
        raise ValueError(f"Duplicate face identifiers: {duplicates}")

    if "qc_pass" not in manifest.columns:
        manifest["qc_pass"] = True
    manifest["qc_pass"] = parse_boolean_series(manifest["qc_pass"])

    eligible = (
        manifest["image_path"].str.strip().ne("")
        & manifest["landmark_path"].str.strip().ne("")
    )
    if not include_qc_failures:
        eligible &= manifest["qc_pass"]

    manifest = (
        manifest.loc[eligible]
        .copy()
        .sort_values("face_id", kind="stable")
        .reset_index(drop=True)
    )
    if max_images is not None:
        manifest = manifest.head(max_images).copy()
    if manifest.empty:
        raise ValueError("No eligible landmark rows were found for alignment.")
    return manifest


def resolve_file_path(raw_path: str, manifest_path: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path

    candidates = [
        Path.cwd() / path,
        manifest_path.resolve().parent / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned or "unnamed_face"


def eye_centres(landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return eye centres ordered by their x position in the image."""
    centre_a = landmarks[EYE_GROUP_A].mean(axis=0)
    centre_b = landmarks[EYE_GROUP_B].mean(axis=0)
    if float(centre_a[0]) <= float(centre_b[0]):
        return centre_a, centre_b
    return centre_b, centre_a


def target_eye_centres(arguments: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    left = np.array(
        [
            arguments.left_eye_x * (arguments.output_width - 1),
            arguments.eye_y * (arguments.output_height - 1),
        ],
        dtype=np.float64,
    )
    right = np.array(
        [
            arguments.right_eye_x * (arguments.output_width - 1),
            arguments.eye_y * (arguments.output_height - 1),
        ],
        dtype=np.float64,
    )
    return left, right


def similarity_transform_from_eyes(
    source_left: np.ndarray,
    source_right: np.ndarray,
    target_left: np.ndarray,
    target_right: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    source_vector = np.asarray(source_right - source_left, dtype=np.float64)
    target_vector = np.asarray(target_right - target_left, dtype=np.float64)
    source_distance = float(np.linalg.norm(source_vector))
    target_distance = float(np.linalg.norm(target_vector))
    if not np.isfinite(source_distance) or source_distance <= 1.0:
        raise ValueError("Source interocular distance is degenerate.")
    if not np.isfinite(target_distance) or target_distance <= 1.0:
        raise ValueError("Target interocular distance is degenerate.")

    source_angle = float(np.arctan2(source_vector[1], source_vector[0]))
    target_angle = float(np.arctan2(target_vector[1], target_vector[0]))
    rotation_radians = target_angle - source_angle
    scale = target_distance / source_distance
    cosine = float(np.cos(rotation_radians))
    sine = float(np.sin(rotation_radians))
    linear = scale * np.array(
        [[cosine, -sine], [sine, cosine]], dtype=np.float64
    )
    translation = target_left - linear @ np.asarray(source_left, dtype=np.float64)
    matrix = np.column_stack([linear, translation]).astype(np.float64)
    return matrix, scale, float(np.degrees(rotation_radians))


def transform_landmarks(landmarks: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack(
        [np.asarray(landmarks, dtype=np.float64), np.ones(len(landmarks))]
    )
    transformed = homogeneous @ matrix.T
    return np.asarray(transformed, dtype=np.float32)


def in_bounds_fraction(
    landmarks: np.ndarray,
    *,
    width: int,
    height: int,
) -> float:
    inside = (
        (landmarks[:, 0] >= 0.0)
        & (landmarks[:, 0] <= width - 1)
        & (landmarks[:, 1] >= 0.0)
        & (landmarks[:, 1] <= height - 1)
    )
    return float(inside.mean())


def save_landmarks(path: Path, landmarks: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(landmarks, dtype=np.float32), allow_pickle=False)


def process_row(
    row: Any,
    *,
    arguments: argparse.Namespace,
    manifest_path: Path,
    images_dir: Path,
    landmarks_dir: Path,
    target_left: np.ndarray,
    target_right: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray | None, np.ndarray | None]:
    face_id = str(row.face_id)
    source_image_path = resolve_file_path(str(row.image_path), manifest_path)
    source_landmark_path = resolve_file_path(
        str(row.landmark_path), manifest_path
    )
    stem = safe_filename(face_id)
    aligned_image_path = images_dir / f"{stem}.png"
    aligned_landmark_path = landmarks_dir / f"{stem}.npy"

    result: dict[str, Any] = {
        "face_id": face_id,
        "source_image_path": source_image_path.as_posix(),
        "source_landmark_path": source_landmark_path.as_posix(),
        "aligned_image_path": aligned_image_path.as_posix(),
        "aligned_landmark_path": aligned_landmark_path.as_posix(),
        "alignment_method": "eye_centres_similarity",
        "output_width": arguments.output_width,
        "output_height": arguments.output_height,
        "source_interocular_distance": np.nan,
        "target_interocular_distance": float(
            np.linalg.norm(target_right - target_left)
        ),
        "scale": np.nan,
        "rotation_degrees": np.nan,
        "matrix_00": np.nan,
        "matrix_01": np.nan,
        "matrix_02": np.nan,
        "matrix_10": np.nan,
        "matrix_11": np.nan,
        "matrix_12": np.nan,
        "left_eye_error_px": np.nan,
        "right_eye_error_px": np.nan,
        "max_eye_error_px": np.nan,
        "landmarks_in_bounds_fraction": np.nan,
        "alignment_qc_pass": False,
        "alignment_qc_reason": "not_processed",
        "outputs_available": False,
        "reused_existing_files": False,
        "error_type": "",
        "error_message": "",
    }

    try:
        if not source_image_path.exists():
            raise FileNotFoundError(f"Image not found: {source_image_path}")
        if not source_landmark_path.exists():
            raise FileNotFoundError(
                f"Landmark array not found: {source_landmark_path}"
            )

        image = cv2.imread(str(source_image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV could not read: {source_image_path}")
        landmarks = np.asarray(
            np.load(source_landmark_path, allow_pickle=False), dtype=np.float32
        )
        expected_shape = (arguments.expected_landmarks, 2)
        if landmarks.shape != expected_shape:
            raise ValueError(
                f"Expected landmark shape {expected_shape}, got {landmarks.shape}."
            )
        if not np.isfinite(landmarks).all():
            raise ValueError("Landmark array contains non-finite coordinates.")

        source_left, source_right = eye_centres(landmarks)
        source_interocular = float(np.linalg.norm(source_right - source_left))
        matrix, scale, rotation_degrees = similarity_transform_from_eyes(
            source_left, source_right, target_left, target_right
        )
        aligned_landmarks = transform_landmarks(landmarks, matrix)
        if not np.isfinite(aligned_landmarks).all():
            raise ValueError("The transformed landmarks are non-finite.")

        aligned_left, aligned_right = eye_centres(aligned_landmarks)
        left_error = float(np.linalg.norm(aligned_left - target_left))
        right_error = float(np.linalg.norm(aligned_right - target_right))
        max_eye_error = max(left_error, right_error)
        bounds_fraction = in_bounds_fraction(
            aligned_landmarks,
            width=arguments.output_width,
            height=arguments.output_height,
        )

        qc_reasons: list[str] = []
        if max_eye_error > arguments.eye_error_tolerance:
            qc_reasons.append("eye_target_error")
        if bounds_fraction < arguments.min_in_bounds_fraction:
            qc_reasons.append("too_many_landmarks_outside_canvas")
        determinant = float(np.linalg.det(matrix[:, :2]))
        if not np.isfinite(determinant) or determinant <= 0.0:
            qc_reasons.append("invalid_transform_determinant")
        qc_pass = not qc_reasons

        border_mode = (
            cv2.BORDER_REFLECT_101
            if arguments.border_mode == "reflect101"
            else cv2.BORDER_CONSTANT
        )
        aligned_image = cv2.warpAffine(
            image,
            matrix,
            (arguments.output_width, arguments.output_height),
            flags=cv2.INTER_CUBIC,
            borderMode=border_mode,
            borderValue=(0, 0, 0),
        )

        image_exists = aligned_image_path.exists()
        landmark_exists = aligned_landmark_path.exists()
        reuse = image_exists and landmark_exists and not arguments.overwrite
        if not arguments.overwrite and image_exists != landmark_exists:
            raise FileExistsError(
                "Only one of the two aligned outputs exists for "
                f"{face_id}. Use --overwrite to regenerate a consistent pair."
            )
        if reuse:
            existing_landmarks = np.asarray(
                np.load(aligned_landmark_path, allow_pickle=False),
                dtype=np.float32,
            )
            if existing_landmarks.shape != aligned_landmarks.shape or not np.allclose(
                existing_landmarks, aligned_landmarks, rtol=0.0, atol=1e-3
            ):
                raise FileExistsError(
                    f"Existing alignment for {face_id} does not match the "
                    "current configuration. Use --overwrite."
                )
        else:
            images_dir.mkdir(parents=True, exist_ok=True)
            landmarks_dir.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(aligned_image_path), aligned_image):
                raise OSError(f"OpenCV could not save: {aligned_image_path}")
            save_landmarks(aligned_landmark_path, aligned_landmarks)

        result.update(
            {
                "source_interocular_distance": source_interocular,
                "scale": scale,
                "rotation_degrees": rotation_degrees,
                "matrix_00": float(matrix[0, 0]),
                "matrix_01": float(matrix[0, 1]),
                "matrix_02": float(matrix[0, 2]),
                "matrix_10": float(matrix[1, 0]),
                "matrix_11": float(matrix[1, 1]),
                "matrix_12": float(matrix[1, 2]),
                "left_eye_error_px": left_error,
                "right_eye_error_px": right_error,
                "max_eye_error_px": max_eye_error,
                "landmarks_in_bounds_fraction": bounds_fraction,
                "alignment_qc_pass": qc_pass,
                "alignment_qc_reason": "ok" if qc_pass else ";".join(qc_reasons),
                "outputs_available": True,
                "reused_existing_files": reuse,
            }
        )
        return result, aligned_landmarks, aligned_image
    except Exception as error:
        result["alignment_qc_reason"] = "processing_exception"
        result["error_type"] = type(error).__name__
        result["error_message"] = str(error)
        return result, None, None


def save_run_artifacts(
    *,
    output_dir: Path,
    results: pd.DataFrame,
    aligned_shapes: list[np.ndarray],
    aligned_images: list[np.ndarray],
    arguments: argparse.Namespace,
    target_left: np.ndarray,
    target_right: np.ndarray,
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "alignment_manifest.csv"
    mean_landmarks_path = output_dir / "mean_aligned_landmarks.npy"
    mean_image_path = output_dir / "mean_aligned_face.png"
    config_path = output_dir / "alignment_config.json"

    results.to_csv(manifest_path, index=False)
    if not aligned_shapes or not aligned_images:
        raise RuntimeError("No successful alignments are available for averages.")

    shape_stack = np.stack(aligned_shapes).astype(np.float32)
    mean_landmarks = shape_stack.mean(axis=0, dtype=np.float64).astype(np.float32)
    save_landmarks(mean_landmarks_path, mean_landmarks)

    image_stack = np.stack(aligned_images).astype(np.float32)
    mean_image = np.clip(
        image_stack.mean(axis=0, dtype=np.float64), 0, 255
    ).astype(np.uint8)
    if not cv2.imwrite(str(mean_image_path), mean_image):
        raise OSError(f"OpenCV could not save: {mean_image_path}")

    configuration = {
        "alignment_method": "eye_centres_similarity",
        "landmark_topology": "mediapipe_face_landmarker_478",
        "eye_group_a": EYE_GROUP_A.tolist(),
        "eye_group_b": EYE_GROUP_B.tolist(),
        "target_left_eye_pixels": target_left.tolist(),
        "target_right_eye_pixels": target_right.tolist(),
        "output_width": arguments.output_width,
        "output_height": arguments.output_height,
        "interpolation": "opencv_inter_cubic",
        "border_mode": arguments.border_mode,
        "min_in_bounds_fraction": arguments.min_in_bounds_fraction,
        "eye_error_tolerance": arguments.eye_error_tolerance,
        "n_faces_in_mean": len(aligned_shapes),
    }
    config_path.write_text(
        json.dumps(configuration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path, mean_landmarks_path, mean_image_path, config_path


def print_summary(
    results: pd.DataFrame,
    *,
    manifest_path: Path,
    mean_landmarks_path: Path,
    mean_image_path: Path,
    config_path: Path,
) -> None:
    total = len(results)
    available = int(results["outputs_available"].sum())
    passed = int(results["alignment_qc_pass"].sum())
    reused = int(results["reused_existing_files"].sum())
    print("\nFace alignment completed")
    print(f"Rows processed: {total}")
    print(f"Outputs available: {available}/{total}")
    print(f"Alignment QC passed: {passed}/{total}")
    print(f"Existing outputs reused: {reused}")
    print(f"Alignment manifest: {manifest_path}")
    print(f"Mean landmarks: {mean_landmarks_path}")
    print(f"Mean aligned face: {mean_image_path}")
    print(f"Run configuration: {config_path}")

    failures = results.loc[
        ~results["alignment_qc_pass"],
        [
            "face_id",
            "alignment_qc_reason",
            "landmarks_in_bounds_fraction",
            "max_eye_error_px",
            "error_type",
            "error_message",
        ],
    ]
    if not failures.empty:
        print("\nFailed alignment QC:")
        print(failures.to_string(index=False))


def main() -> int:
    arguments = parse_arguments()
    validate_arguments(arguments)
    manifest_path = arguments.manifest_path.expanduser()
    output_dir = arguments.output_dir.expanduser()
    images_dir = output_dir / "images"
    landmarks_dir = output_dir / "landmarks"

    manifest = read_manifest(
        manifest_path,
        include_qc_failures=arguments.include_qc_failures,
        max_images=arguments.max_images,
    )
    output_stems = manifest["face_id"].map(safe_filename)
    if output_stems.duplicated().any():
        collisions = sorted(
            manifest.loc[output_stems.duplicated(keep=False), "face_id"].tolist()
        )
        raise ValueError(
            "face_id values collide after filename sanitisation: "
            f"{collisions}"
        )
    target_left, target_right = target_eye_centres(arguments)

    results: list[dict[str, Any]] = []
    aligned_shapes: list[np.ndarray] = []
    aligned_images: list[np.ndarray] = []
    total = len(manifest)
    for position, row in enumerate(manifest.itertuples(index=False), start=1):
        print(f"[{position:03d}/{total:03d}] Aligning {row.face_id}")
        result, aligned_shape, aligned_image = process_row(
            row,
            arguments=arguments,
            manifest_path=manifest_path,
            images_dir=images_dir,
            landmarks_dir=landmarks_dir,
            target_left=target_left,
            target_right=target_right,
        )
        results.append(result)
        if (
            aligned_shape is not None
            and aligned_image is not None
            and bool(result["alignment_qc_pass"])
        ):
            aligned_shapes.append(aligned_shape)
            aligned_images.append(aligned_image)

    results_df = pd.DataFrame(results)
    artifact_paths = save_run_artifacts(
        output_dir=output_dir,
        results=results_df,
        aligned_shapes=aligned_shapes,
        aligned_images=aligned_images,
        arguments=arguments,
        target_left=target_left,
        target_right=target_right,
    )
    print_summary(
        results_df,
        manifest_path=artifact_paths[0],
        mean_landmarks_path=artifact_paths[1],
        mean_image_path=artifact_paths[2],
        config_path=artifact_paths[3],
    )

    has_failures = bool((~results_df["alignment_qc_pass"]).any())
    return 1 if arguments.fail_on_qc_error and has_failures else 0


if __name__ == "__main__":
    sys.exit(main())
