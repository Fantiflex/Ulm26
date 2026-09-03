from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd


DEFAULT_MANIFEST_PATH = Path(
    "data/interim/landmarks_mediapipe/landmarks_manifest.csv"
)
DEFAULT_OUTPUT_DIR = Path("data/results/landmark_qc_mediapipe")

# OpenCV uses BGR colours.
CONTOUR_COLOUR = (40, 220, 40)
MESH_COLOUR = (160, 160, 160)
POINT_COLOUR = (30, 30, 235)
INDEX_COLOUR = (255, 255, 255)
IRIS_COLOUR = (0, 215, 255)
QC_PASS_COLOUR = (50, 210, 50)
QC_FAIL_COLOUR = (40, 40, 235)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw MediaPipe Face Landmarker landmarks produced by "
            "06_extract_landmarks.py."
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
        "--max-images",
        type=int,
        default=None,
        help="Visualise only the first N eligible rows.",
    )
    parser.add_argument(
        "--include-qc-failures",
        action="store_true",
        help=(
            "Include rows with qc_pass=false when a landmark file exists. "
            "By default, only QC-passing rows are visualised."
        ),
    )
    parser.add_argument(
        "--draw-mesh",
        action="store_true",
        help="Also draw the dense MediaPipe tessellation.",
    )
    parser.add_argument(
        "--no-contours",
        action="store_true",
        help="Do not draw the official face, eye, eyebrow and lip contours.",
    )
    parser.add_argument(
        "--no-points",
        action="store_true",
        help="Do not draw individual landmark points.",
    )
    parser.add_argument(
        "--show-indices",
        action="store_true",
        help="Write every landmark index next to its point (visually dense).",
    )
    parser.add_argument(
        "--point-radius",
        type=int,
        default=1,
        help="Radius of landmark points in pixels. Default: 1.",
    )
    parser.add_argument(
        "--line-thickness",
        type=int,
        default=1,
        help="Contour and mesh line thickness. Default: 1.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing visualisations.",
    )
    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.max_images is not None and arguments.max_images <= 0:
        raise ValueError("--max-images must be strictly positive.")
    if arguments.point_radius < 1:
        raise ValueError("--point-radius must be at least 1.")
    if arguments.line_thickness < 1:
        raise ValueError("--line-thickness must be at least 1.")


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

    if "qc_pass" not in manifest.columns:
        manifest["qc_pass"] = True
    manifest["qc_pass"] = parse_boolean_series(manifest["qc_pass"])

    manifest["face_id"] = manifest["face_id"].astype(str)
    manifest["image_path"] = manifest["image_path"].fillna("").astype(str)
    manifest["landmark_path"] = (
        manifest["landmark_path"].fillna("").astype(str)
    )

    eligible = manifest["landmark_path"].str.strip().ne("")
    if not include_qc_failures:
        eligible &= manifest["qc_pass"]
    manifest = manifest.loc[eligible].copy().reset_index(drop=True)

    if max_images is not None:
        manifest = manifest.head(max_images).copy()
    if manifest.empty:
        raise ValueError(
            "No eligible rows were found. If failed-QC rows have saved .npy "
            "files, retry with --include-qc-failures."
        )
    return manifest


def resolve_file_path(raw_path: str, manifest_path: Path) -> Path:
    """Resolve absolute paths and common project-relative manifest paths."""
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


def get_connection_group(attribute: str) -> Iterable[Any]:
    """Load a connection group from the current MediaPipe Tasks API."""
    try:
        import mediapipe as mp

        connection_class = mp.tasks.vision.FaceLandmarksConnections
    except (ImportError, AttributeError) as first_error:
        try:
            from mediapipe.tasks.python.vision import (
                FaceLandmarksConnections as connection_class,
            )
        except (ImportError, AttributeError) as second_error:
            raise RuntimeError(
                "The installed MediaPipe package does not expose "
                "FaceLandmarksConnections. This script targets the current "
                "MediaPipe Tasks Face Landmarker API."
            ) from second_error

    connections = getattr(connection_class, attribute, None)
    if connections is None:
        raise RuntimeError(
            f"MediaPipe does not expose connection group {attribute!r}."
        )
    return connections


def connection_indices(connection: Any) -> tuple[int, int]:
    """Support both Tasks connection objects and two-item tuples."""
    if hasattr(connection, "start") and hasattr(connection, "end"):
        return int(connection.start), int(connection.end)
    if isinstance(connection, (tuple, list)) and len(connection) == 2:
        return int(connection[0]), int(connection[1])
    raise TypeError(f"Unsupported MediaPipe connection: {connection!r}")


def draw_connections(
    image: np.ndarray,
    landmarks: np.ndarray,
    connections: Iterable[Any],
    *,
    colour: tuple[int, int, int],
    thickness: int,
) -> None:
    height, width = image.shape[:2]
    for connection in connections:
        start_index, end_index = connection_indices(connection)
        if start_index >= len(landmarks) or end_index >= len(landmarks):
            continue

        start = landmarks[start_index]
        end = landmarks[end_index]
        if not np.isfinite(start).all() or not np.isfinite(end).all():
            continue

        start_point = (
            int(np.clip(round(float(start[0])), 0, width - 1)),
            int(np.clip(round(float(start[1])), 0, height - 1)),
        )
        end_point = (
            int(np.clip(round(float(end[0])), 0, width - 1)),
            int(np.clip(round(float(end[1])), 0, height - 1)),
        )
        cv2.line(
            image,
            start_point,
            end_point,
            colour,
            thickness,
            lineType=cv2.LINE_AA,
        )


def draw_points(
    image: np.ndarray,
    landmarks: np.ndarray,
    *,
    radius: int,
    show_indices: bool,
) -> None:
    height, width = image.shape[:2]
    iris_indices = set(range(468, min(478, len(landmarks))))

    for index, coordinate in enumerate(landmarks):
        if not np.isfinite(coordinate).all():
            continue
        point = (
            int(np.clip(round(float(coordinate[0])), 0, width - 1)),
            int(np.clip(round(float(coordinate[1])), 0, height - 1)),
        )
        colour = IRIS_COLOUR if index in iris_indices else POINT_COLOUR
        cv2.circle(
            image,
            point,
            radius,
            colour,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        if show_indices:
            label_point = (
                min(point[0] + radius + 1, width - 1),
                max(point[1] - radius - 1, 0),
            )
            cv2.putText(
                image,
                str(index),
                label_point,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.28,
                INDEX_COLOUR,
                1,
                lineType=cv2.LINE_AA,
            )


def add_information_banner(
    image: np.ndarray,
    *,
    face_id: str,
    qc_pass: bool,
    qc_reason: str,
    n_landmarks: int,
) -> np.ndarray:
    banner_height = 34
    output = cv2.copyMakeBorder(
        image,
        banner_height,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(25, 25, 25),
    )
    qc_label = "PASS" if qc_pass else "FAIL"
    qc_colour = QC_PASS_COLOUR if qc_pass else QC_FAIL_COLOUR
    text = (
        f"{face_id} | MediaPipe: {n_landmarks} points | "
        f"QC={qc_label} ({qc_reason})"
    )
    cv2.putText(
        output,
        text,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        qc_colour,
        1,
        lineType=cv2.LINE_AA,
    )
    return output


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned or "unnamed_face"


def visualise_row(
    row: Any,
    *,
    manifest_path: Path,
    output_dir: Path,
    contour_connections: Iterable[Any] | None,
    iris_connections: Iterable[Any] | None,
    mesh_connections: Iterable[Any] | None,
    draw_landmark_points: bool,
    show_indices: bool,
    point_radius: int,
    line_thickness: int,
    overwrite: bool,
) -> dict[str, Any]:
    face_id = str(row.face_id)
    image_path = resolve_file_path(str(row.image_path), manifest_path)
    landmark_path = resolve_file_path(str(row.landmark_path), manifest_path)
    output_path = output_dir / f"{safe_filename(face_id)}_landmarks.png"
    qc_pass = bool(row.qc_pass)
    qc_reason = str(getattr(row, "qc_reason", "not_recorded"))

    result: dict[str, Any] = {
        "face_id": face_id,
        "image_path": image_path.as_posix(),
        "landmark_path": landmark_path.as_posix(),
        "visualisation_path": output_path.as_posix(),
        "visualisation_created": False,
        "reused_existing_file": False,
        "qc_pass": qc_pass,
        "qc_reason": qc_reason,
        "n_landmarks": 0,
        "error_type": "",
        "error_message": "",
    }

    try:
        if output_path.exists() and not overwrite:
            result["visualisation_created"] = True
            result["reused_existing_file"] = True
            return result
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not landmark_path.exists():
            raise FileNotFoundError(f"Landmark array not found: {landmark_path}")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV could not read image: {image_path}")

        landmarks = np.load(landmark_path, allow_pickle=False)
        landmarks = np.asarray(landmarks, dtype=np.float32)
        if landmarks.ndim != 2 or landmarks.shape[1] != 2:
            raise ValueError(
                "Expected a landmark array shaped (N, 2), got "
                f"{landmarks.shape}."
            )
        if len(landmarks) < 468:
            raise ValueError(
                f"Expected at least 468 MediaPipe landmarks, got {len(landmarks)}."
            )
        if not np.isfinite(landmarks).all():
            raise ValueError("Landmark array contains non-finite coordinates.")

        annotated = image.copy()
        if mesh_connections is not None:
            draw_connections(
                annotated,
                landmarks,
                mesh_connections,
                colour=MESH_COLOUR,
                thickness=line_thickness,
            )
        if contour_connections is not None:
            draw_connections(
                annotated,
                landmarks,
                contour_connections,
                colour=CONTOUR_COLOUR,
                thickness=line_thickness,
            )
        if iris_connections is not None:
            draw_connections(
                annotated,
                landmarks,
                iris_connections,
                colour=IRIS_COLOUR,
                thickness=line_thickness,
            )
        if draw_landmark_points:
            draw_points(
                annotated,
                landmarks,
                radius=point_radius,
                show_indices=show_indices,
            )

        annotated = add_information_banner(
            annotated,
            face_id=face_id,
            qc_pass=qc_pass,
            qc_reason=qc_reason,
            n_landmarks=len(landmarks),
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), annotated):
            raise OSError(f"OpenCV could not save: {output_path}")

        result["visualisation_created"] = True
        result["n_landmarks"] = int(len(landmarks))
        return result
    except Exception as error:
        result["error_type"] = type(error).__name__
        result["error_message"] = str(error)
        return result


def print_summary(results: pd.DataFrame, *, output_dir: Path) -> None:
    total = len(results)
    created = int(results["visualisation_created"].sum())
    reused = int(results["reused_existing_file"].sum())
    failed = total - created
    print("\nMediaPipe landmark visualisation completed")
    print(f"Rows processed: {total}")
    print(f"Visualisations available: {created}/{total}")
    print(f"Existing files reused: {reused}")
    print(f"Failures: {failed}")
    print(f"Output directory: {output_dir}")

    failures = results.loc[
        ~results["visualisation_created"],
        ["face_id", "error_type", "error_message"],
    ]
    if not failures.empty:
        print("\nFailed rows:")
        print(failures.to_string(index=False))


def main() -> int:
    arguments = parse_arguments()
    validate_arguments(arguments)
    manifest_path = arguments.manifest_path.expanduser()
    output_dir = arguments.output_dir.expanduser()

    manifest = read_manifest(
        manifest_path,
        include_qc_failures=arguments.include_qc_failures,
        max_images=arguments.max_images,
    )

    contour_connections = None
    iris_connections = None
    if not arguments.no_contours:
        contour_connections = get_connection_group(
            "FACE_LANDMARKS_CONTOURS"
        )
        iris_connections = list(
            get_connection_group("FACE_LANDMARKS_LEFT_IRIS")
        ) + list(get_connection_group("FACE_LANDMARKS_RIGHT_IRIS"))

    mesh_connections = None
    if arguments.draw_mesh:
        # MediaPipe's public constant intentionally spells TESSELATION with
        # one L in this part of the API.
        mesh_connections = get_connection_group(
            "FACE_LANDMARKS_TESSELATION"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    total = len(manifest)
    for position, row in enumerate(
        manifest.itertuples(index=False), start=1
    ):
        print(f"[{position:03d}/{total:03d}] Visualising {row.face_id}")
        result = visualise_row(
            row,
            manifest_path=manifest_path,
            output_dir=output_dir,
            contour_connections=contour_connections,
            iris_connections=iris_connections,
            mesh_connections=mesh_connections,
            draw_landmark_points=not arguments.no_points,
            show_indices=arguments.show_indices,
            point_radius=arguments.point_radius,
            line_thickness=arguments.line_thickness,
            overwrite=arguments.overwrite,
        )
        results.append(result)

    results_df = pd.DataFrame(results)
    results_path = output_dir / "landmark_visualisation_manifest.csv"
    results_df.to_csv(results_path, index=False)
    print_summary(results_df, output_dir=output_dir)
    print(f"Visualisation manifest saved to: {results_path}")

    has_failures = bool((~results_df["visualisation_created"]).any())
    return 1 if has_failures else 0


if __name__ == "__main__":
    sys.exit(main())
