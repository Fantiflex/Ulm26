"""Create face-only crops from images using a lightweight contour mask.

This script keeps the face contour rather than extracting the full MediaPipe
mesh, removes the background, and computes face luminance using only the face
pixels. It is intended as a lighter-weight companion to the full landmark
extraction workflow used in 06_extract_landmarks.py.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_INPUT_DIR = Path("data/raw/cfd/images")
DEFAULT_OUTPUT_DIR = Path("data/interim/faces_only")
DEFAULT_MODEL_PATH = Path("models/face_landmarker.task")
CIRCLE_RADIUS_PCT = 0.50
GRID_COLS = 1
GRID_ROWS = 1
CELL_PX = 400

GROUP_LUMINANCE_COLORS = {
    "white": {
        "base_rgb": (173, 216, 230),
        "target_luminance": 180.0,
    },
    "black": {
        "base_rgb": (30, 110, 70),
        "target_luminance": 120.0,
    },
}

GROUP_COLOR_SCHEMES = {
    "blue_green": {
        "white": (173, 216, 230),
        "black": (30, 110, 70),
    },
    "green_blue": {
        "white": (30, 110, 70),
        "black": (173, 216, 230),
    },
}


def list_image_paths(input_dir: Path, recursive: bool = False) -> list[Path]:
    """Return supported image files in the directory."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if recursive:
        iterator = input_dir.rglob("*")
    else:
        iterator = input_dir.iterdir()

    files = [
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files)


def compute_face_mask(image_bgr: np.ndarray, face_landmarker: Any) -> tuple[np.ndarray, Any] | tuple[None, None]:
    """Return a binary convex-hull mask covering the detected face region."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    if hasattr(face_landmarker, "detect"):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = face_landmarker.detect(mp_image)
        face_landmarks = getattr(results, "face_landmarks", None)
        if not face_landmarks:
            return None, None
        landmarks = face_landmarks[0]
    elif hasattr(face_landmarker, "process"):
        results = face_landmarker.process(rgb)
        if results.multi_face_landmarks is None or len(results.multi_face_landmarks) == 0:
            return None, None
        landmarks = results.multi_face_landmarks[0].landmark
    else:
        raise TypeError("Unsupported MediaPipe face detector API.")

    height, width = image_bgr.shape[:2]
    face_points: list[tuple[int, int]] = []

    for landmark in landmarks:
        x = int(round(landmark.x * width))
        y = int(round(landmark.y * height))
        if 0 <= x < width and 0 <= y < height:
            face_points.append((x, y))

    if len(face_points) < 3:
        return None, results

    hull = cv2.convexHull(np.asarray(face_points, dtype=np.int32))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [hull], 255)
    return mask, results


def compute_luminance(face_pixels: np.ndarray) -> float:
    """Compute average luminance for RGB face pixels."""
    if face_pixels.size == 0:
        return float("nan")

    rgb = face_pixels.astype(np.float32)
    luminance = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    return float(np.mean(luminance))


def extract_face_only(
    image_path: Path,
    output_dir: Path,
    face_landmarker: Any,
) -> dict[str, Any]:
    """Extract the face-only crop and luminance statistics for one image."""
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    mask, results = compute_face_mask(image_bgr, face_landmarker)
    if mask is None or results is None:
        return {
            "image_path": str(image_path),
            "face_detected": False,
            "luminance_mean": None,
            "luminance_median": None,
            "output_path": None,
        }

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    face_pixels = rgb[mask > 0]
    face_pixels = face_pixels.reshape(-1, 3)

    luminance_values = 0.299 * face_pixels[:, 0] + 0.587 * face_pixels[:, 1] + 0.114 * face_pixels[:, 2]
    mean_l = float(np.mean(luminance_values))
    median_l = float(np.median(luminance_values))

    face_only = np.zeros_like(rgb)
    face_only[mask > 0] = rgb[mask > 0]

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    out_path = output_dir / f"{stem}_face_only.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(face_only, cv2.COLOR_RGB2BGR))

    return {
        "image_path": str(image_path),
        "face_detected": True,
        "luminance_mean": mean_l,
        "luminance_median": median_l,
        "output_path": str(out_path),
    }


def build_face_landmarker(model_path: Path) -> Any:
    """Create a MediaPipe Face Landmarker object compatible with the installed API."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"MediaPipe model not found: {model_path}. "
            "The project includes models/face_landmarker.task; "
            "or pass --model-path to an existing .task bundle."
        )

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def process_folder(
    input_dir: Path,
    output_dir: Path,
    recursive: bool = False,
    max_images: int | None = None,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> list[dict[str, Any]]:
    """Process a folder of images and save face-only crops."""
    image_paths = list_image_paths(input_dir, recursive=recursive)
    if max_images is not None:
        image_paths = image_paths[:max_images]

    face_landmarker = build_face_landmarker(model_path)
    try:
        results = []
        for path in image_paths:
            results.append(extract_face_only(path, output_dir, face_landmarker))
        return results
    finally:
        if hasattr(face_landmarker, "close"):
            face_landmarker.close()


def write_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    """Write the luminance summary to a CSV file."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_path",
        "face_detected",
        "luminance_mean",
        "luminance_median",
        "output_path",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Keep only the face region using the outer contour and compute "
            "face luminance with no background."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Folder containing the input images. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for the processed face-only outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the MediaPipe Face Landmarker .task model. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively under the input directory.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit processing to the first N images.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional CSV file to store luminance summaries.",
    )
    return parser.parse_args()


def assemble_grid(cells: list[Image.Image]) -> Image.Image:
    """Assemble a grid by pasting each circular cell into a white canvas."""
    if not cells:
        raise ValueError("No cells to assemble.")

    grid_cols = int(np.ceil(np.sqrt(len(cells))))
    grid_rows = int(np.ceil(len(cells) / grid_cols))
    matrix = Image.new("RGB", (grid_cols * CELL_PX, grid_rows * CELL_PX), color=(255, 255, 255))

    for idx, cell in enumerate(cells):
        row = idx // grid_cols
        col = idx % grid_cols
        x = col * CELL_PX
        y = row * CELL_PX
        matrix.paste(cell, (x, y))

    return matrix


def make_luminance_matched_circle_cell(cell_px: int, gray_value: float | int) -> Image.Image:
    """Create a grayscale circle whose fill matches the face luminance."""
    gray_value = int(np.clip(round(float(gray_value)), 0, 255))

    img = Image.new("L", (cell_px, cell_px), color=255)
    draw = ImageDraw.Draw(img)

    radius = max(1, round(CIRCLE_RADIUS_PCT * cell_px))
    cx = cell_px // 2 
    cy = cell_px // 2
    bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
    draw.ellipse(bbox, fill=gray_value, outline=gray_value)

    return img


def luminance_from_rgb(rgb: tuple[int, int, int]) -> float:
    """Convert an RGB triple to luminance using the standard formula."""
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b


def match_group_luminance(base_rgb: tuple[int, int, int], group: str, target_luminance: float | None = None) -> tuple[int, int, int]:
    """Adjust the provided base color so its luminance matches the target value."""
    group_key = str(group).strip().lower()
    base = np.asarray(base_rgb, dtype=np.float32)

    if target_luminance is None:
        target = float(GROUP_LUMINANCE_COLORS.get(group_key, {}).get("target_luminance", 180.0))
    else:
        target = float(target_luminance)

    current = luminance_from_rgb(tuple(np.clip(base, 0, 255).astype(int)))
    if current <= 0:
        return tuple(int(v) for v in base)

    scale = target / current
    adjusted = np.clip(base * scale, 0, 255).astype(int)
    return tuple(int(v) for v in adjusted)


def group_mean_luminance(cell_info: Any) -> dict[str, float]:
    """Return the mean luminance for each ethnicity group in the selected set."""
    target: dict[str, list[float]] = {"white": [], "black": []}
    for _, row in cell_info.iterrows():
        group = str(row.get("group", "white")).strip().lower()
        if group in {"minority", "black", "b"}:
            target["black"].append(float(row["face_luminance"]))
        elif group in {"majority", "white", "w"}:
            target["white"].append(float(row["face_luminance"]))

    return {
        group: float(np.mean(values)) if values else default
        for group, values, default in [
            ("white", target["white"], 180.0),
            ("black", target["black"], 120.0),
        ]
    }


def make_ethnicity_circle_cell(
    cell_px: int,
    gray_value: float | int,
    group: str,
    target_luminance: float | None = None,
    color_scheme: str = "blue_green",
) -> Image.Image:
    """Create a circle whose dominant color reflects the selected ethnicity palette."""
    group_key = str(group).strip().lower()
    base_color = GROUP_COLOR_SCHEMES.get(color_scheme, GROUP_COLOR_SCHEMES["blue_green"]).get(
        group_key,
        GROUP_COLOR_SCHEMES["blue_green"]["white"],
    )
    if group_key in {"w", "white"}:
        base_color = GROUP_COLOR_SCHEMES.get(color_scheme, GROUP_COLOR_SCHEMES["blue_green"]).get(
            "white",
            (173, 216, 230),
        )
    elif group_key in {"b", "black"}:
        base_color = GROUP_COLOR_SCHEMES.get(color_scheme, GROUP_COLOR_SCHEMES["blue_green"]).get(
            "black",
            (30, 110, 70),
        )
    else:
        base_color = GROUP_COLOR_SCHEMES["blue_green"]["white"]

    if target_luminance is None:
        if group_key in {"white", "w"}:
            target_luminance = 180.0
        elif group_key in {"black", "b"}:
            target_luminance = 120.0
        else:
            target_luminance = float(gray_value)

    rgb = match_group_luminance(base_color, group_key, target_luminance)

    img = Image.new("RGB", (cell_px, cell_px), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    radius = max(1, round(CIRCLE_RADIUS_PCT * cell_px))
    cx = cell_px // 2
    cy = cell_px // 2
    bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
    draw.ellipse(bbox, fill=tuple(rgb), outline=tuple(rgb))

    return img


def make_nonsocial_from_cell_info(
    cell_info: Any,
    cell_px: int,
    color_scheme: str = "blue_green",
) -> Image.Image:
    """Build a matrix in which each cell is a luminance-matched circle colored by group."""
    if cell_info.empty:
        raise ValueError("cell_info is empty; no cells to build.")

    targets = group_mean_luminance(cell_info)
    cells: list[Image.Image] = []
    ordered = cell_info.sort_values("cell_idx").copy()

    for _, row in ordered.iterrows():
        gray = float(row["face_luminance"])
        group = str(row.get("group", "white")).lower()
        if group == "minority":
            group = "black"
        elif group == "majority":
            group = "white"

        target_luminance = targets.get(group, gray)
        cells.append(
            make_ethnicity_circle_cell(
                cell_px,
                gray,
                group,
                target_luminance,
                color_scheme=color_scheme,
            )
        )

    return assemble_grid(cells)


def main() -> None:
    args = parse_arguments()
    rows = process_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        recursive=args.recursive,
        max_images=args.max_images,
        model_path=args.model_path,
    )

    if args.csv_output is not None:
        write_csv(rows, args.csv_output)

    print(f"Processed {len(rows)} images.")
    detected = sum(1 for row in rows if row["face_detected"])
    print(f"Faces detected: {detected}")

    for row in rows:
        if row["face_detected"]:
            print(
                f"{row['image_path']} | luminance_mean={row['luminance_mean']:.2f} | "
                f"luminance_median={row['luminance_median']:.2f}"
            )


if __name__ == "__main__":
    main()
