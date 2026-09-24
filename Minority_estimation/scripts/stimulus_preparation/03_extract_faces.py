"""
Extract face-only images and compute face luminance.

This script processes the CFD images selected for the experiment,
detects the face using MediaPipe, removes the background, computes
luminance statistics using face pixels only, and saves a summary CSV.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "cfd"
    / "images"
)

STIMULI_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stimuli.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "faces_only"
)

DEFAULT_CSV_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "faces_only"
    / "face_luminance.csv"
)

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "face_landmarker.task"
)


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


# ============================================================
# IMAGE DISCOVERY
# ============================================================

def list_image_paths(
    input_dir: Path,
    recursive: bool = False,
) -> list[Path]:
    """Return supported image files from the requested directory."""

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}"
        )

    iterator = (
        input_dir.rglob("*")
        if recursive
        else input_dir.iterdir()
    )

    files = [
        path
        for path in iterator
        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    ]

    return sorted(files)


# ============================================================
# MEDIAPIPE
# ============================================================

def build_face_landmarker(
    model_path: Path,
) -> Any:
    """Create the MediaPipe Face Landmarker using CPU inference."""

    if not model_path.exists():
        raise FileNotFoundError(
            f"MediaPipe model not found: {model_path}"
        )

    BaseOptions = mp.tasks.BaseOptions

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(model_path),
            delegate=BaseOptions.Delegate.CPU,
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    return (
        mp.tasks.vision.FaceLandmarker
        .create_from_options(options)
    )
# ============================================================
# FACE MASK
# ============================================================

def compute_face_mask(
    image_bgr: np.ndarray,
    face_landmarker: Any,
) -> np.ndarray | None:
    """Return a convex-hull mask covering the detected face."""

    rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb,
    )

    results = face_landmarker.detect(mp_image)

    face_landmarks = getattr(
        results,
        "face_landmarks",
        None,
    )

    if not face_landmarks:
        return None

    landmarks = face_landmarks[0]

    height, width = image_bgr.shape[:2]

    face_points = []

    for landmark in landmarks:

        x = int(
            round(
                landmark.x
                * width
            )
        )

        y = int(
            round(
                landmark.y
                * height
            )
        )

        if (
            0 <= x < width
            and
            0 <= y < height
        ):
            face_points.append(
                (x, y)
            )

    if len(face_points) < 3:
        return None

    hull = cv2.convexHull(
        np.asarray(
            face_points,
            dtype=np.int32,
        )
    )

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    cv2.fillPoly(
        mask,
        [hull],
        255,
    )

    return mask


# ============================================================
# LUMINANCE
# ============================================================

def compute_luminance_statistics(
    face_pixels: np.ndarray,
) -> tuple[float, float]:
    """
    Compute mean and median luminance from RGB face pixels.
    """

    if face_pixels.size == 0:
        return float("nan"), float("nan")

    pixels = face_pixels.astype(
        np.float32
    )

    luminance = (
        0.299 * pixels[:, 0]
        + 0.587 * pixels[:, 1]
        + 0.114 * pixels[:, 2]
    )

    return (
        float(np.mean(luminance)),
        float(np.median(luminance)),
    )


# ============================================================
# ONE IMAGE
# ============================================================

def extract_face(
    image_path: Path,
    output_dir: Path,
    face_landmarker: Any,
) -> dict[str, Any]:
    """
    Extract the face-only image and compute luminance statistics.
    """

    image_bgr = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image_bgr is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    mask = compute_face_mask(
        image_bgr,
        face_landmarker,
    )

    if mask is None:

        return {
            "face_id": image_path.stem,
            "image_path": str(image_path),
            "face_detected": False,
            "luminance_mean": None,
            "luminance_median": None,
            "output_path": None,
        }

    rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    face_pixels = rgb[
        mask > 0
    ]

    mean_luminance, median_luminance = (
        compute_luminance_statistics(
            face_pixels
        )
    )

    # Transparent background
    rgba = np.zeros(
        (
            rgb.shape[0],
            rgb.shape[1],
            4,
        ),
        dtype=np.uint8,
    )

    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = mask

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"{image_path.stem}_face_only.png"
    )

    cv2.imwrite(
        str(output_path),
        cv2.cvtColor(
            rgba,
            cv2.COLOR_RGBA2BGRA,
        ),
    )

    return {
        "face_id": image_path.stem,
        "image_path": str(image_path),
        "face_detected": True,
        "luminance_mean": mean_luminance,
        "luminance_median": median_luminance,
        "output_path": str(output_path),
    }


# ============================================================
# PROCESS DATASET
# ============================================================

def load_selected_image_paths(
    stimuli_path: Path,
    image_directory: Path,
) -> list[Path]:
    """
    Load only the CFD images selected by 02_build_stimuli.py.
    """

    if not stimuli_path.exists():
        raise FileNotFoundError(
            f"Stimulus table not found: {stimuli_path}"
        )

    stimuli = pd.read_csv(
        stimuli_path
    )

    if stimuli.empty:
        raise ValueError(
            f"Stimulus table is empty: {stimuli_path}"
        )

    # --------------------------------------------------------
    # Preferred case:
    # stimuli.csv already contains an image_path column.
    # --------------------------------------------------------

    if "image_path" in stimuli.columns:

        image_paths = []

        for value in stimuli["image_path"].dropna():

            path = Path(
                str(value)
            )

            # If the CSV contains a relative path, try resolving
            # it from the project root.
            if not path.is_absolute():

                project_relative = (
                    PROJECT_ROOT
                    / path
                )

                if project_relative.exists():
                    path = project_relative

                else:
                    path = (
                        image_directory
                        / path.name
                    )

            image_paths.append(
                path
            )

    # --------------------------------------------------------
    # Fallback:
    # construct paths from face_id.
    # --------------------------------------------------------

    elif "face_id" in stimuli.columns:

        available_images = {
            path.stem: path
            for path in list_image_paths(
                image_directory,
                recursive=True,
            )
        }

        image_paths = []

        for face_id in stimuli["face_id"]:

            face_id = str(
                face_id
            )

            if face_id not in available_images:

                raise FileNotFoundError(
                    f"No image found for face_id: {face_id}"
                )

            image_paths.append(
                available_images[
                    face_id
                ]
            )

    else:

        raise ValueError(
            "stimuli.csv must contain either "
            "'image_path' or 'face_id'."
        )


    # --------------------------------------------------------
    # Validate all paths
    # --------------------------------------------------------

    missing = [
        path
        for path in image_paths
        if not path.exists()
    ]

    if missing:

        preview = "\n".join(
            str(path)
            for path in missing[:10]
        )

        raise FileNotFoundError(
            "Some selected stimulus images could not be found:\n"
            f"{preview}"
        )


    # Remove accidental duplicates while preserving order.

    return list(
        dict.fromkeys(
            image_paths
        )
    )


def process_selected_stimuli(
    stimuli_path: Path,
    image_directory: Path,
    output_dir: Path,
    model_path: Path,
    max_images: int | None = None,
) -> list[dict[str, Any]]:
    """
    Extract faces only for stimuli selected for the experiment.
    """

    image_paths = load_selected_image_paths(
        stimuli_path=stimuli_path,
        image_directory=image_directory,
    )

    if max_images is not None:

        image_paths = image_paths[
            :max_images
        ]


    print(
        f"Selected stimuli to process: "
        f"{len(image_paths)}"
    )


    face_landmarker = (
        build_face_landmarker(
            model_path
        )
    )


    try:

        results = []

        for index, image_path in enumerate(
            image_paths,
            start=1,
        ):

            print(
                f"[{index}/{len(image_paths)}] "
                f"{image_path.name}"
            )

            result = extract_face(
                image_path=image_path,
                output_dir=output_dir,
                face_landmarker=face_landmarker,
            )

            results.append(
                result
            )

        return results

    finally:

        if hasattr(
            face_landmarker,
            "close",
        ):
            face_landmarker.close()
# ============================================================
# SAVE SUMMARY
# ============================================================

def write_csv(
    rows: list[dict[str, Any]],
    csv_path: Path,
) -> None:

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "face_id",
        "image_path",
        "face_detected",
        "luminance_mean",
        "luminance_median",
        "output_path",
    ]

    with csv_path.open(
        "w",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# COMMAND LINE
# ============================================================

def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Extract CFD face regions and "
            "compute face luminance."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )

    parser.add_argument(
        "--csv-output",
        type=Path,
        default=DEFAULT_CSV_OUTPUT,
    )


    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    args = parse_arguments()

    rows = process_selected_stimuli(
        stimuli_path=STIMULI_PATH,
        image_directory=args.input_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        max_images=args.max_images,
    )

    write_csv(
        rows,
        args.csv_output,
    )

    detected = sum(
        row["face_detected"]
        for row in rows
    )

    print(
        "\nFace extraction"
    )

    print(
        "---------------"
    )

    print(
        f"Images processed: {len(rows)}"
    )

    print(
        f"Faces detected: {detected}"
    )

    print(
        f"Detection failures: "
        f"{len(rows) - detected}"
    )

    print(
        f"\nFace images saved to: "
        f"{args.output_dir}"
    )

    print(
        f"Luminance table saved to: "
        f"{args.csv_output}"
    )


if __name__ == "__main__":
    main()