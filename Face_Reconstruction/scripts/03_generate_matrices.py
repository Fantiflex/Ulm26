"""
Generate square image matrices from the filtered CFD stimulus set.

This script reads the final stimulus table produced by 02_build_stimuli.py,
classifies the selected faces by gender x ethnicity, creates category folders,
and asks the user for the desired percentage composition of each matrix.
It then samples the requested number of images from each category and calls the
matrix builder to save a JPG collage.
"""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont
import argparse
import random
import shutil
import sys
from pathlib import Path
import json
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from matrix_construction.build_matrix import build_matrix
from matrix_construction.choose_circle_face import make_nonsocial_from_cell_info


STIMULI_PATH = Path("data/processed/stimuli.csv")
MATRIX_ROOT = Path("data/matrix")


def load_stimuli(stimuli_path: Path) -> pd.DataFrame:
    """Load the final stimulus CSV."""
    if not stimuli_path.exists():
        raise FileNotFoundError(f"Stimulus table not found: {stimuli_path}")

    stimuli = pd.read_csv(stimuli_path)
    required = {"gender_self", "ethnicity_self", "image_path"}
    missing = required - set(stimuli.columns)
    if missing:
        raise ValueError(
            "Stimulus table is missing required columns: "
            f"{sorted(missing)}"
        )

    return stimuli


def build_category_label(row: pd.Series) -> str:
    """Create a category label from gender and ethnicity."""
    gender = str(row["gender_self"]).strip()
    ethnicity = str(row["ethnicity_self"]).strip()
    return f"{gender}_{ethnicity}"


def prepare_category_folders(
    stimuli: pd.DataFrame,
    output_root: Path,
    category_subdir: str = "categories",
) -> dict[str, list[Path]]:
    """Copy stimuli images into category folders and return their file lists."""
    category_root = output_root / category_subdir
    if category_root.exists():
        shutil.rmtree(category_root)
    category_root.mkdir(parents=True, exist_ok=True)

    category_files: dict[str, list[Path]] = {}

    for _, row in stimuli.iterrows():
        category = build_category_label(row)
        image_path = Path(row["image_path"]).expanduser()

        if not image_path.exists():
            continue

        category_dir = category_root / category
        category_dir.mkdir(parents=True, exist_ok=True)

        destination = category_dir / image_path.name
        if not destination.exists():
            shutil.copy2(image_path, destination)

        category_files.setdefault(category, []).append(destination)

    if not category_files:
        raise ValueError("No stimulus images were available to classify.")

    return category_files


def ask_percentage(category: str) -> float:
    """Prompt the user for a percentage for a category."""
    while True:
        raw = input(f"Percentage for {category} (%): ").strip()
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a number, for example 25 or 33.3.")
            continue

        if value < 0:
            print("Percentage must be non-negative.")
            continue

        return value


def compute_max_matrix_size(
    category_files: dict[str, list[Path]],
    percentages: dict[str, float],
) -> int:
    """Compute the largest square matrix size allowed by the category shares and image counts."""
    category_caps: dict[str, int] = {
        category: len(paths)
        for category, paths in category_files.items()
    }

    max_size = 0
    for category, share in percentages.items():
        if share <= 0:
            continue
        max_category_size = category_caps.get(category, 0)
        if max_category_size == 0:
            return 0
        max_size += max_category_size

    return max_size


def ask_matrix_size(
    category_files: dict[str, list[Path]],
    percentages: dict[str, float],
) -> int:
    """Ask the user for the target matrix size and warn them if it exceeds the max feasible size."""
    max_size = compute_max_matrix_size(category_files, percentages)
    print(f"\nMaximum matrix size allowed by your selected composition and available images: {max_size}")

    while True:
        raw = input("Desired matrix size (number of images): ").strip()
        try:
            value = int(raw)
        except ValueError:
            print("Please enter a whole number.")
            continue

        if value <= 0:
            print("Matrix size must be positive.")
            continue

        if value > max_size:
            print(
                "Requested matrix size exceeds the maximum feasible size. "
                f"Maximum allowed: {max_size}"
            )
            continue

        return value


def ask_matrix_percentages(categories: list[str]) -> dict[str, float]:
    """Ask the user for a percentage for each category and validate the total."""
    percentages: dict[str, float] = {}
    for category in categories:
        percentages[category] = ask_percentage(category)

    total = sum(percentages.values())
    if abs(total - 100.0) > 1e-6:
        print(
            "The percentages must sum to 100. "
            f"Current total: {total:.2f}%"
        )
        raise ValueError("Matrix percentages must sum to 100%.")

    return percentages


def black_percentage_label(percentages: dict[str, float]) -> str:
    """Return a folder label based on total Black-category percentage."""
    black_share = sum(
        value
        for category, value in percentages.items()
        if (
            "black" in category.lower()
            or category.lower().endswith("_b")
        )
    )

    return f"{float(black_share):g}pct_black"


def build_matrix_from_composition(
    category_files: dict[str, list[Path]],
    percentages: dict[str, float],
    matrix_size: int,
    output_path: Path,
    tile_size: int = 200,
) -> dict[str, int]:
    """Build both the face matrix and the corresponding ethnicity-colored circle matrix."""
    chosen_files: list[Path] = []
    counts: dict[str, int] = {}
    chosen_meta: list[dict[str, object]] = []

    for category, share in percentages.items():
        if share <= 0:
            continue

        target_count = round(matrix_size * share / 100.0)
        if target_count == 0:
            continue

        available = category_files.get(category, [])
        if len(available) < target_count:
            raise ValueError(
                f"Category {category} has only {len(available)} images, "
                f"but {target_count} were requested."
            )

        chosen = random.sample(available, target_count)
        chosen_files.extend(chosen)
        counts[category] = target_count

    if len(chosen_files) < matrix_size:
        remaining = matrix_size - len(chosen_files)
        if remaining > 0:
            print(
                "The requested percentages underfilled the matrix; "
                f"adding {remaining} extra image(s) from the largest available category."
            )
            all_candidates = [
                path
                for files in category_files.values()
                for path in files
            ]
            all_candidates = [
                path for path in all_candidates if path not in chosen_files
            ]
            if len(all_candidates) < remaining:
                raise ValueError(
                    "Not enough remaining images to fill the requested matrix size."
                )
            extra = random.sample(all_candidates, remaining)
            chosen_files.extend(extra)

    random.shuffle(chosen_files)

    for index, image_path in enumerate(chosen_files):
        ethnicity = ""
        category = None
        for candidate_category, paths in category_files.items():
            if image_path in paths:
                category = candidate_category
                ethnicity = str(candidate_category).split("_", 1)[-1]
                break

        face_luminance = 180.0
        try:
            image = Image.open(image_path).convert("RGB")
            luminance_pixels = [
                0.299 * r + 0.587 * g + 0.114 * b
                for r, g, b in image.getdata()
            ]
            if luminance_pixels:
                face_luminance = float(sum(luminance_pixels) / len(luminance_pixels))
        except Exception as e:
            raise RuntimeError(
                f"Failed to compute luminance for {image_path}: {e}"
            )

        group = "white" if str(ethnicity).lower() in {"w", "white"} else "black"
        chosen_meta.append(
        {
            "cell_idx": index,
            "source_filename": image_path.name,
            "source_path": str(image_path),
            "category": category,
            "group": group,
            "face_luminance": float(face_luminance),
        }
    
        )

    if chosen_meta:
        for row in chosen_meta[:10]:
            print(
                row["source_filename"],
                "| group =", row["group"],
                "| face_luminance =", row["face_luminance"],
            )
        cell_info = pd.DataFrame(chosen_meta)
        cell_info_path = output_path.parent / "cells.csv"
        cell_info.to_csv(cell_info_path, index=False)

        face_matrix_dir = output_path.parent / f"{output_path.stem}_faces"
        if face_matrix_dir.exists():
            shutil.rmtree(face_matrix_dir)
        face_matrix_dir.mkdir(parents=True, exist_ok=True)

        for idx, image_path in enumerate(chosen_files):
            target = face_matrix_dir / f"cell_{idx + 1:02d}_{image_path.name}"
            shutil.copy2(image_path, target)

        face_matrix_path = output_path.parent / "face.jpg"
        build_matrix(
            input_dir=face_matrix_dir,
            output_path=face_matrix_path,
            n_images=len(chosen_files),
            tile_size=tile_size,
            quality=100,
        )

        circle_blue_green = make_nonsocial_from_cell_info(
            cell_info,
            tile_size,
            color_scheme="blue_green",
        )
        circle_green_blue = make_nonsocial_from_cell_info(
            cell_info,
            tile_size,
            color_scheme="green_blue",
        )

        circle_output_standard = output_path.parent / "circles_blue_green.jpg"
        circle_output_inverse = output_path.parent / "circles_green_blue.jpg"
        circle_blue_green.save(circle_output_standard, format="JPEG", quality=100)
        circle_green_blue.save(circle_output_inverse, format="JPEG", quality=100)

        print(f"Created face matrix: {face_matrix_path}")
        print(f"Created circle matrix (white blue / black green): {circle_output_standard}")
        print(f"Created circle matrix (white green / black blue): {circle_output_inverse}")
        return counts

    return counts

def generate_matrices(
    stimuli_path: Path,
    matrix_root: Path,
    n_matrices: int,
    matrix_size: int | None,
    tile_size: int,
) -> list[Path]:
    """Generate matrices based on user-defined category percentages."""
    stimuli = load_stimuli(stimuli_path)
    category_files = prepare_category_folders(stimuli, matrix_root)
    categories = sorted(category_files)

    if not categories:
        raise ValueError("No categories found in the stimulus table.")

    output_dir = matrix_root / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_paths: list[Path] = []

    for matrix_index in range(1, n_matrices + 1):
        print(f"\nMatrix {matrix_index}/{n_matrices}")

        percentages = ask_matrix_percentages(categories)

        if matrix_size is None:
            matrix_size = ask_matrix_size(
                category_files,
                percentages,
            )

        black_label = black_percentage_label(percentages)


        black_percentage = sum(
            value
            for category, value in percentages.items()
            if (
                "black" in category.lower()
                or category.lower().endswith("_b")
            )
        )

        black_count = round(matrix_size * black_percentage / 100)

        matrix_id = (
            f"mb{black_count:02d}"
            f"_n{matrix_size:02d}"
            f"_v{matrix_index:02d}"
        )

        condition_dir = output_dir / black_label
        matrix_dir = condition_dir / matrix_id
        matrix_dir.mkdir(parents=True, exist_ok=True)

        output_path = matrix_dir / "matrix.jpg"

        counts = build_matrix_from_composition(
            category_files=category_files,
            percentages=percentages,
            matrix_size=matrix_size,
            output_path=output_path,
            tile_size=tile_size,
        )

        black_percentage = sum(
            value
            for category, value in percentages.items()
            if (
                "black" in category.lower()
                or category.lower().endswith("_b")
            )
        )
        black_count = round(matrix_size * black_percentage / 100)

        matrix_id = (
            f"mb{black_count:02d}"
            f"_n{matrix_size:02d}"
            f"_v{matrix_index:02d}"
        )
        metadata = {
            "matrix_id": matrix_id,
            "matrix_size": matrix_size,
            "black_percentage": black_percentage,
            "percentages": percentages,
            "counts": counts,
            "tile_size": tile_size,
        }

        metadata_path = matrix_dir / "metadata.json"

        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(
                metadata,
                f,
                indent=2,
            )

        generated_paths.append(matrix_dir)

        print(f"Created matrix set: {matrix_dir}")
        print(f"Metadata saved to: {metadata_path}")
        print("Counts used:")

        for category, count in counts.items():
            print(f"  - {category}: {count}")

    return generated_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate square share-based matrices from the filtered stimuli "
            "by ethnicity x gender category."
        )
    )
    parser.add_argument(
        "--stimuli-path",
        type=Path,
        default=STIMULI_PATH,
        help="Path to the processed stimulus CSV (default: data/processed/stimuli.csv)",
    )
    parser.add_argument(
        "--matrix-root",
        type=Path,
        default=MATRIX_ROOT,
        help="Root directory where category folders and output matrices are stored.",
    )
    parser.add_argument(
        "--n-matrices",
        type=int,
        default=1,
        help="Number of matrices to generate.",
    )
    parser.add_argument(
        "--matrix-size",
        type=int,
        default=None,
        help="Total number of images in each matrix. If omitted, you will be prompted for it.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=200,
        help="Pixel size of each image tile within the matrix.",
    )

    args = parser.parse_args()

    if args.n_matrices <= 0:
        raise ValueError("n_matrices must be a positive integer.")

    if args.matrix_size is not None and args.matrix_size <= 0:
        raise ValueError("matrix_size must be a positive integer.")

    generate_matrices(
        stimuli_path=args.stimuli_path,
        matrix_root=args.matrix_root,
        n_matrices=args.n_matrices,
        matrix_size=args.matrix_size,
        tile_size=args.tile_size,
    )


if __name__ == "__main__":
    main()
