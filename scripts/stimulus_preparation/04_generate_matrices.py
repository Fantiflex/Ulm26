"""
Generate the final social and non-social experimental matrices.

For each target Black-face count, this script:

1. samples Black and White CFD stimuli;
2. arranges 64 stimuli in a randomized 8x8 matrix;
3. generates the corresponding social face matrix;
4. generates the corresponding non-social circle matrix:
5. saves cell-level metadata and matrix-level metadata.

The generated directory structure is directly compatible with the
participant experiment.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from minority_estimation.config import (
    BACKGROUND_GRAY_VALUE,
    CELL_PX,
    FACE_CENTER_Y_RATIO,
    FACE_WITH_HAIR_ZOOM,
    GRID_COLS,
    GRID_ROWS,
    MATRIX_SIZE,
    N_VERSIONS,
    RANDOM_SEED,
    TARGET_BLACK_COUNTS,
)

from minority_estimation.image_processing import (
    compute_face_crop_box,
)

FACE_AREA_CALIBRATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "calibration"
    / "face_area_calibration.json"
)

# ============================================================
# PROJECT PATHS
# ============================================================



STIMULI_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stimuli.csv"
)

FACE_LUMINANCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "faces_only"
    / "face_luminance.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiment"
    / "images_matrices"
)



# ============================================================
# DATA LOADING
# ============================================================

def load_data() -> pd.DataFrame:
    """
    Load stimulus metadata and face-luminance measurements.
    """

    if not STIMULI_PATH.exists():

        raise FileNotFoundError(
            f"Stimulus table not found: {STIMULI_PATH}"
        )

    if not FACE_LUMINANCE_PATH.exists():

        raise FileNotFoundError(
            f"Face luminance table not found: "
            f"{FACE_LUMINANCE_PATH}"
        )


    stimuli = pd.read_csv(
        STIMULI_PATH
    )

    luminance = pd.read_csv(
        FACE_LUMINANCE_PATH
    )


    # --------------------------------------------------------
    # Only keep successfully detected faces
    # --------------------------------------------------------

    if "face_detected" in luminance.columns:

        luminance = luminance.loc[
            luminance["face_detected"] == True
        ].copy()


    # --------------------------------------------------------
    # Create a matching identifier from filenames
    # --------------------------------------------------------

    stimuli["image_name"] = (
        stimuli["image_path"]
        .astype(str)
        .map(
            lambda path:
            Path(path).stem
        )
    )

    luminance["image_name"] = (
        luminance["image_path"]
        .astype(str)
        .map(
            lambda path:
            Path(path).stem
        )
    )


    merged = stimuli.merge(
        luminance[
            [
                "image_name",
                "luminance_mean",
                "luminance_median",
                "output_path",
            ]
        ],
        on="image_name",
        how="inner",
        validate="one_to_one",
    )


    if merged.empty:

        raise RuntimeError(
            "No stimuli could be matched with "
            "face-luminance measurements."
        )


    return merged


# ============================================================
# GROUP NORMALIZATION
# ============================================================

def normalize_group(
    value: str,
) -> str:
    """
    Normalize perceived ethnicity to 'black' or 'white'.
    """

    value = str(value).strip().lower()

    if value in {
        "black",
        "b",
    }:
        return "black"

    if value in {
        "white",
        "w",
    }:
        return "white"

    raise ValueError(
        f"Unexpected group value: {value}"
    )


# ============================================================
# FACE CELLS
# ============================================================



def prepare_face_cell_with_hair(
    image_path: Path,
    cell_px: int = CELL_PX,
    zoom: float = FACE_WITH_HAIR_ZOOM,
) -> Image.Image:
    """
    Prepare an original CFD portrait while keeping the hair.

    The image is cropped around the head and enlarged so the face
    occupies more of the 100x100 experimental cell.
    """

    if not image_path.exists():
        raise FileNotFoundError(
            f"Original CFD image not found: {image_path}"
        )

    with Image.open(image_path) as image:

        image = image.convert("RGB")

        width, height = image.size

        left, top, right, bottom = compute_face_crop_box(
            width=width,
            height=height,
            zoom=zoom,
        )

    cropped = image.crop(
        (
            left,
            top,
            right,
            bottom,
        )
    )

    cropped = cropped.resize(
        (
            cell_px,
            cell_px,
        ),
        Image.Resampling.LANCZOS,
    )

    return cropped




# ============================================================
# MATRIX ASSEMBLY
# ============================================================

def assemble_matrix(
    cells: list[Image.Image],
) -> Image.Image:
    """
    Assemble 64 cells into an 8x8 matrix.
    """

    if len(cells) != MATRIX_SIZE:

        raise ValueError(
            f"Expected {MATRIX_SIZE} cells, "
            f"received {len(cells)}."
        )


    matrix = Image.new(
        "RGB",
        (
            GRID_COLS * CELL_PX,
            GRID_ROWS * CELL_PX,
        ),
        color=(
            BACKGROUND_GRAY_VALUE,
            BACKGROUND_GRAY_VALUE,
            BACKGROUND_GRAY_VALUE,
        ),
    )


    for index, cell in enumerate(cells):

        row = (
            index
            // GRID_COLS
        )

        column = (
            index
            % GRID_COLS
        )

        x = (
            column
            * CELL_PX
        )

        y = (
            row
            * CELL_PX
        )

        matrix.paste(
            cell,
            (
                x,
                y,
            ),
        )


    return matrix


# ============================================================
# SOCIAL MATRICES
# ============================================================

def build_face_matrix_with_hair(
    cells: pd.DataFrame,
) -> Image.Image:
    """
    Construct the main social matrix using original CFD portraits.
    Hair is preserved.
    """

    images = []

    for _, row in cells.iterrows():

        image_path = Path(
            row["image_path"]
        )

        images.append(
            prepare_face_cell_with_hair(
                image_path
            )
        )

    return assemble_matrix(
        images
    )



# ============================================================
# NON-SOCIAL CIRCLE
# ============================================================

def load_circle_radius() -> int:
    """
    Load the empirically calibrated circle radius.

    The radius is derived from the median detected face area
    by 03b_analyze_face_area.py.
    """

    if not FACE_AREA_CALIBRATION_PATH.exists():
        raise FileNotFoundError(
            "Face-area calibration file not found: "
            f"{FACE_AREA_CALIBRATION_PATH}. "
            "Run 03b_analyze_face_area.py first."
        )

    with FACE_AREA_CALIBRATION_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        calibration = json.load(handle)

    required_key = "rendered_circle_radius_px"

    if required_key not in calibration:
        raise KeyError(
            f"Calibration file is missing '{required_key}'."
        )

    radius = calibration[required_key]

    if not isinstance(radius, int):
        raise TypeError(
            "rendered_circle_radius_px must be an integer."
        )

    if radius <= 0:
        raise ValueError(
            "Circle radius must be strictly positive."
        )

    if radius * 2 > CELL_PX:
        raise ValueError(
            "Circle diameter exceeds cell size."
        )

    return radius

def make_circle_cell(
    *,
    face_luminance: float,
    circle_radius_px: int,
) -> Image.Image:
    """
    Generate one non-social circle whose grayscale intensity
    matches the mean luminance of the corresponding face region.
    """

    gray = int(
        np.clip(
            round(float(face_luminance)),
            0,
            255,
        )
    )

    image = Image.new(
        "RGB",
        (
            CELL_PX,
            CELL_PX,
        ),
        color=(
            BACKGROUND_GRAY_VALUE,
            BACKGROUND_GRAY_VALUE,
            BACKGROUND_GRAY_VALUE,
        ),
    )

    draw = ImageDraw.Draw(image)

    radius = circle_radius_px

    center_x = CELL_PX // 2
    center_y = CELL_PX // 2

    bounding_box = [
        center_x - radius,
        center_y - radius,
        center_x + radius,
        center_y + radius,
    ]

    draw.ellipse(
        bounding_box,
        fill=(gray, gray, gray),
    )

    return image
# ============================================================
# NON-SOCIAL MATRIX
# ============================================================

def build_circle_matrix(
    cells: pd.DataFrame,
    circle_radius_px: int,
) -> Image.Image:
    """
    Generate the non-social equivalent of one face matrix.

    Each circle's grayscale intensity directly matches the
    measured mean luminance of the corresponding face region.
    """

    images = []

    for _, row in cells.iterrows():

        images.append(
            make_circle_cell(
                face_luminance=row["luminance_mean"],
                circle_radius_px=circle_radius_px,
            )
        )

    return assemble_matrix(images)

# ============================================================
# SAMPLE ONE MATRIX
# ============================================================

def sample_matrix(
    data: pd.DataFrame,
    target_black: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Sample one randomized 64-cell composition.
    """

    target_white = (
        MATRIX_SIZE
        - target_black
    )


    black = data.loc[
        data["group"]
        == "black"
    ]


    white = data.loc[
        data["group"]
        == "white"
    ]


    if black.empty:

        raise RuntimeError(
            "No Black stimuli are available."
        )


    if white.empty:

        raise RuntimeError(
            "No White stimuli are available."
        )


    if len(black) < target_black:
        raise RuntimeError(
            f"Need {target_black} unique Black stimuli, "
            f"but only {len(black)} are available."
        )

    if len(white) < target_white:
        raise RuntimeError(
            f"Need {target_white} unique White stimuli, "
            f"but only {len(white)} are available."
        )
    

    black_indices = rng.choice(
        black.index.to_numpy(),
        size=target_black,
        replace=False,
    )


    white_indices = rng.choice(
        white.index.to_numpy(),
        size=target_white,
        replace=False,
    )

    

    sampled = pd.concat(
        [
            black.loc[
                black_indices
            ],

            white.loc[
                white_indices
            ],
        ],
        ignore_index=True,
    )


    permutation = rng.permutation(
        len(sampled)
    )


    sampled = (
        sampled
        .iloc[
            permutation
        ]
        .reset_index(
            drop=True
        )
    )


    sampled.insert(
        0,
        "cell_idx",
        np.arange(
            MATRIX_SIZE
        ),
    )


    sampled["row"] = (
        sampled[
            "cell_idx"
        ]
        // GRID_COLS
    )


    sampled["column"] = (
        sampled[
            "cell_idx"
        ]
        % GRID_COLS
    )


    return sampled


# ============================================================
# METADATA
# ============================================================

def make_metadata(
    *,
    target_black: int,
    version: int,
    cells: pd.DataFrame,
    circle_radius_px: int,
) -> dict:
    """Build metadata for one generated matrix."""

    target_white = MATRIX_SIZE - target_black

    actual_black = int(
        (cells["group"] == "black").sum()
    )

    actual_white = int(
        (cells["group"] == "white").sum()
    )

    matrix_mean_black = float(
        cells.loc[
            cells["group"] == "black",
            "luminance_mean",
        ].mean()
    )

    matrix_mean_white = float(
        cells.loc[
            cells["group"] == "white",
            "luminance_mean",
        ].mean()
    )

    return {
        "matrix_size": MATRIX_SIZE,
        "grid_rows": GRID_ROWS,
        "grid_columns": GRID_COLS,
        "cell_px": CELL_PX,

        "social_image": "face.png",
        "non_social_image": "circles.png",

        "face_with_hair_zoom": FACE_WITH_HAIR_ZOOM,

        "target_black_count": target_black,
        "target_white_count": target_white,

        "actual_black_count": actual_black,
        "actual_white_count": actual_white,

        "black_percentage":
            100 * target_black / MATRIX_SIZE,

        "white_percentage":
            100 * target_white / MATRIX_SIZE,

        "version": version,

        "random_seed": RANDOM_SEED,

        "sampling_with_replacement": False,

        "circle_radius_px": circle_radius_px,

        "circle_radius_ratio":
            circle_radius_px / CELL_PX,

        "circle_size_calibration":
            "median_mediapipe_face_area",

        "circle_luminance_mapping":
            "direct_face_luminance",

        "matrix_mean_black_face_luminance":
            matrix_mean_black,

        "matrix_mean_white_face_luminance":
            matrix_mean_white,
    }
# ============================================================
# GENERATE ONE VERSION
# ============================================================

def generate_one_matrix(
    *,
    data: pd.DataFrame,
    target_black: int,
    version: int,
    rng: np.random.Generator,
    circle_radius_px: int,
) -> None:
    """
    Generate one social matrix and one non-social matrix.
    """

    folder_percentage = round(
        100
        * target_black
        / MATRIX_SIZE
    )


    matrix_name = (

        f"mb"
        f"{target_black:02d}"
        f"_n64"
        f"_v{version:02d}"

    )


    output_directory = (

        OUTPUT_DIR
        / f"{folder_percentage}pct_black"
        / matrix_name

    )


    if output_directory.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_directory}"
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=False,
    )


    # --------------------------------------------------------
    # Sample matrix
    # --------------------------------------------------------

    cells = sample_matrix(
        data=data,
        target_black=target_black,
        rng=rng,
    )


    # --------------------------------------------------------
    # Social
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Social — original CFD portraits WITH hair
    # --------------------------------------------------------

    face_matrix = (
        build_face_matrix_with_hair(
            cells
        )
    )

    face_matrix.save(
        output_directory
        / "face.png",
        quality=95,
    )




    # --------------------------------------------------------
    # Non-social circles
    # --------------------------------------------------------

    circle_matrix = circle_matrix = build_circle_matrix(
        cells,
        circle_radius_px=circle_radius_px,
    )

    circle_matrix.save(
        output_directory / "circles.png"
    )


    # --------------------------------------------------------
    # Cell metadata
    # --------------------------------------------------------

    cells.to_csv(
        output_directory
        / "cells.csv",
        index=False,
    )


    # --------------------------------------------------------
    # Matrix metadata
    # --------------------------------------------------------

    metadata = make_metadata(
        target_black=target_black,
        version=version,
        cells=cells,
        circle_radius_px=circle_radius_px,
    )


    with (
        output_directory
        / "metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            metadata,
            handle,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# MAIN GENERATION
# ============================================================

def generate_matrices() -> None:
    """
    Generate every experimental matrix.
    """

    data = load_data()
    circle_radius_px = load_circle_radius()

    print(
        f"Using calibrated circle radius: "
        f"{circle_radius_px} px"
    )

    # --------------------------------------------------------
    # Determine experimental group
    # --------------------------------------------------------

    if (
        "ethnicity_perceived"
        not in data.columns
    ):

        raise ValueError(
            "stimuli.csv must contain "
            "'ethnicity_perceived'."
        )


    data["group"] = (
        data[
            "ethnicity_perceived"
        ]
        .map(
            normalize_group
        )
    )

    rng = np.random.default_rng(RANDOM_SEED)

    total = len(TARGET_BLACK_COUNTS) * N_VERSIONS
    generated = 0

    for target_black in TARGET_BLACK_COUNTS:
        for version in range(1, N_VERSIONS + 1):

            generate_one_matrix(
                data=data,
                target_black=target_black,
                version=version,
                rng=rng,
                circle_radius_px=circle_radius_px,
            )

            generated += 1

            print(
                f"[{generated:02d}/{total}] "
                f"Black={target_black:02d}/64 | "
                f"version={version:02d}"
            )

    print(f"\nGenerated {generated} matrix pairs.")
    print(f"Output directory: {OUTPUT_DIR}")

    
# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    generate_matrices()