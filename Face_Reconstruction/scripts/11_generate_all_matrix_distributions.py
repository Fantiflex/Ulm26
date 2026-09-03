"""Generate all valid black/white stimulus compositions for M_L = 0.

This script iterates over the requested percentage lists:
- M_B = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
- M_L = 0
- M_W = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

For each valid pair satisfying M_B + M_W = 100 and M_L = 0, it builds a matrix and
stores it in a dedicated folder named as {B}pct_black.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STIMULI_PATH = REPO_ROOT / "data/processed/stimuli.csv"
MATRIX_ROOT = REPO_ROOT / "data/matrix"

BLACK_VALUES = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
L_VALUES = [0]
WHITE_VALUES = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]


def load_matrix_module():
    """Load the existing matrix-generation script as a module."""
    script_path = Path(__file__).with_name("03_generate_matrices.py")
    spec = importlib.util.spec_from_file_location("generate_matrices_three", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load matrix generation script from {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_combinations() -> list[tuple[float, float]]:
    """Return all valid (black, white) combinations with M_L = 0."""
    combos: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()

    for black in BLACK_VALUES:
        for white in WHITE_VALUES:
            if black + white != 100:
                continue
            for light in L_VALUES:
                if light != 0:
                    continue
                pair = (float(black), float(white))
                if pair not in seen:
                    seen.add(pair)
                    combos.append(pair)

    return sorted(combos)


def category_group_key(category: str) -> str:
    """Return the ethnicity code corresponding to the category label."""
    return str(category).split("_")[-1].upper()


def build_distribution_percentages(
    category_files: dict[str, list[Path]],
    black_pct: float,
    white_pct: float,
) -> dict[str, float]:
    """Create a percentage dictionary that matches the requested black/white composition.

    The actual category labels in the stimulus table are gender_ethnicity, e.g. M_B,
    M_W, M_L, so we need to group by the last code segment rather than searching for the
    literal words 'black' or 'white'.
    """
    percentages: dict[str, float] = {}

    black_categories = sorted(
        category for category in category_files if category_group_key(category) == "B"
    )
    white_categories = sorted(
        category for category in category_files if category_group_key(category) == "W"
    )
    light_categories = sorted(
        category for category in category_files if category_group_key(category) == "L"
    )

    if black_categories:
        share = black_pct / len(black_categories)
        for category in black_categories:
            percentages[category] = share

    if white_categories:
        share = white_pct / len(white_categories)
        for category in white_categories:
            percentages[category] = share

    for category in light_categories:
        percentages[category] = 0.0

    missing_categories = [
        category for category in sorted(category_files)
        if category not in percentages
    ]
    for category in missing_categories:
        percentages[category] = 0.0

    total = sum(percentages.values())
    if abs(total - 100.0) > 1e-6:
        raise ValueError(
            f"Distribution composition does not sum to 100: black={black_pct}, white={white_pct}, total={total}"
        )

    return percentages


def generate_distribution_matrix(
    stimuli_path: Path,
    matrix_root: Path,
    black_pct: float,
    white_pct: float,
    matrix_size: int,
    tile_size: int,
) -> Path:
    """Generate one matrix for the requested black/white distribution."""
    mod = load_matrix_module()
    stimuli = mod.load_stimuli(stimuli_path)
    category_files = mod.prepare_category_folders(stimuli, matrix_root)

    percentages = build_distribution_percentages(category_files, black_pct, white_pct)
    folder_name = f"{black_pct:g}pct_black"
    output_dir = matrix_root / "generated" / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "matrix.jpg"
    mod.build_matrix_from_composition(
        category_files=category_files,
        percentages=percentages,
        matrix_size=matrix_size,
        output_path=output_path,
        tile_size=tile_size,
    )

    print(f"Generated: {output_dir}")
    print(f"  black={black_pct}%, white={white_pct}%, M_L=0%")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate all valid black/white matrix distributions for M_L = 0, "
            "using the requested M_B and M_W lists."
        )
    )
    parser.add_argument(
        "--stimuli-path",
        type=Path,
        default=STIMULI_PATH,
        help="Path to the processed stimuli CSV. Default: data/processed/stimuli.csv",
    )
    parser.add_argument(
        "--matrix-root",
        type=Path,
        default=MATRIX_ROOT,
        help="Root directory used to store generated matrix folders.",
    )
    parser.add_argument(
        "--matrix-size",
        type=int,
        default=72,
        help="Number of images in each generated matrix. Fixed at 72 in this workflow.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=200,
        help="Pixel size of each tile in the final matrix.",
    )
    args = parser.parse_args()

    if args.matrix_size <= 0:
        raise ValueError("--matrix-size must be > 0")
    if args.matrix_size != 72:
        print(f"Note: fixed matrix size for this workflow is 72. Using {args.matrix_size} as requested.")

    valid_pairs = valid_combinations()
    if not valid_pairs:
        raise ValueError("No valid M_B + M_W combinations were found for the requested lists.")

    print("Valid distributions:")
    for black, white in valid_pairs:
        print(f"  M_B={black}%, M_L=0, M_W={white}%")

    for black, white in valid_pairs:
        generate_distribution_matrix(
            stimuli_path=args.stimuli_path,
            matrix_root=args.matrix_root,
            black_pct=black,
            white_pct=white,
            matrix_size=args.matrix_size,
            tile_size=args.tile_size,
        )


if __name__ == "__main__":
    main()
