"""
Generate the final social and non-social experimental matrices.

For each target Black-face count, this script:

1. samples Black and White CFD stimuli;
2. arranges 64 stimuli in a randomized 8x8 matrix;
3. generates the corresponding social face matrix;
4. generates two non-social circle matrices:
   - blue_green
   - green_blue
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


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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
# EXPERIMENT PARAMETERS
# ============================================================

MATRIX_SIZE = 64

GRID_ROWS = 8
GRID_COLS = 8

CELL_PX = 100

TARGET_COUNTS = [
    8,
    14,
    20,
    26,
    32,
    38,
    44,
    50,
    56,
]

N_VERSIONS = 5

RANDOM_SEED = 42


# ============================================================
# NON-SOCIAL STIMULI
# ============================================================

CIRCLE_RADIUS_RATIO = 0.30

WITHIN_GROUP_LUMINANCE_SCALE = 1.0


# Mean display luminance of each artificial group.
#
# blue_green:
#   White-face-equivalent group -> lighter
#   Black-face-equivalent group -> darker
#
# green_blue:
#   mapping is reversed.

GROUP_COLOR_SCHEMES = {

    "blue_green": {
        "white": 190,
        "black": 90,
    },

    "green_blue": {
        "white": 90,
        "black": 190,
    },
}


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

FACE_WITH_HAIR_ZOOM = 1.10
FACE_NO_HAIR_PADDING = 0.06


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

        # Smaller crop => larger apparent face
        crop_size = int(
            min(width, height) / zoom
        )

        center_x = width // 2

        # Shift upward slightly because the face/head is above
        # the geometric centre of a CFD portrait.
        center_y = int(
            height * 0.42
        )

        left = (
            center_x
            - crop_size // 2
        )

        top = (
            center_y
            - crop_size // 2
        )

        # Keep crop inside image bounds
        left = max(
            0,
            min(
                left,
                width - crop_size,
            ),
        )

        top = max(
            0,
            min(
                top,
                height - crop_size,
            ),
        )

        right = (
            left
            + crop_size
        )

        bottom = (
            top
            + crop_size
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


def prepare_face_cell_no_hair(
    image_path: Path,
    cell_px: int = CELL_PX,
    padding_ratio: float = FACE_NO_HAIR_PADDING,
) -> Image.Image:
    """
    Prepare a transparent face-only PNG.

    The visible face is tightly cropped, slightly padded,
    and enlarged inside a white experimental cell.
    """

    if not image_path.exists():
        raise FileNotFoundError(
            f"Face-only image not found: {image_path}"
        )

    with Image.open(image_path) as image:

        image = image.convert("RGBA")

        alpha = image.getchannel("A")

        bbox = alpha.getbbox()

        if bbox is None:
            raise ValueError(
                f"No visible face found in: {image_path}"
            )

        left, top, right, bottom = bbox

        width = (
            right
            - left
        )

        height = (
            bottom
            - top
        )

        padding = int(
            max(
                width,
                height,
            )
            * padding_ratio
        )

        left = max(
            0,
            left - padding,
        )

        top = max(
            0,
            top - padding,
        )

        right = min(
            image.width,
            right + padding,
        )

        bottom = min(
            image.height,
            bottom + padding,
        )

        face = image.crop(
            (
                left,
                top,
                right,
                bottom,
            )
        )

        # Preserve aspect ratio.
        face.thumbnail(
            (
                cell_px,
                cell_px,
            ),
            Image.Resampling.LANCZOS,
        )

        background = Image.new(
            "RGBA",
            (
                cell_px,
                cell_px,
            ),
            (
                255,
                255,
                255,
                255,
            ),
        )

        x = (
            cell_px
            - face.width
        ) // 2

        y = (
            cell_px
            - face.height
        ) // 2

        background.alpha_composite(
            face,
            (
                x,
                y,
            ),
        )

        return background.convert(
            "RGB"
        )

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
            255,
            255,
            255,
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


def build_face_matrix_no_hair(
    cells: pd.DataFrame,
) -> Image.Image:
    """
    Construct an alternative social matrix using face-only PNGs.
    """

    images = []

    for _, row in cells.iterrows():

        image_path = Path(
            row["output_path"]
        )

        images.append(
            prepare_face_cell_no_hair(
                image_path
            )
        )

    return assemble_matrix(
        images
    )
# ============================================================
# NON-SOCIAL CIRCLE
# ============================================================

def make_circle_cell(
    *,
    face_luminance: float,
    group: str,
    human_group_mean: float,
    scheme: str,
) -> Image.Image:
    """
    Generate one circle preserving within-group luminance variation.
    """

    group = normalize_group(
        group
    )


    display_group_mean = float(
        GROUP_COLOR_SCHEMES[
            scheme
        ][
            group
        ]
    )


    # Individual deviation from the human group's mean.

    deviation = (
        float(face_luminance)
        - float(human_group_mean)
    )


    display_luminance = (
        display_group_mean
        + (
            WITHIN_GROUP_LUMINANCE_SCALE
            * deviation
        )
    )


    gray = int(
        np.clip(
            round(
                display_luminance
            ),
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
            255,
            255,
            255,
        ),
    )


    draw = ImageDraw.Draw(
        image
    )


    radius = int(
        CELL_PX
        * CIRCLE_RADIUS_RATIO
    )


    center_x = (
        CELL_PX
        // 2
    )

    center_y = (
        CELL_PX
        // 2
    )


    bounding_box = [

        center_x - radius,
        center_y - radius,

        center_x + radius,
        center_y + radius,

    ]


    draw.ellipse(
        bounding_box,
        fill=(
            gray,
            gray,
            gray,
        ),
    )


    return image


# ============================================================
# NON-SOCIAL MATRIX
# ============================================================

def build_circle_matrix(
    cells: pd.DataFrame,
    scheme: str,
) -> Image.Image:
    """
    Generate the non-social equivalent of one face matrix.
    """

    if scheme not in GROUP_COLOR_SCHEMES:

        raise ValueError(
            f"Unknown color scheme: {scheme}"
        )


    group_means = (
        cells
        .groupby(
            "group"
        )[
            "luminance_mean"
        ]
        .mean()
        .to_dict()
    )


    images = []


    for _, row in cells.iterrows():

        group = row["group"]

        images.append(

            make_circle_cell(

                face_luminance=(
                    row["luminance_mean"]
                ),

                group=group,

                human_group_mean=(
                    group_means[group]
                ),

                scheme=scheme,

            )

        )


    return assemble_matrix(
        images
    )


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


    # Sampling WITH replacement.
    #
    # This allows every requested composition even if
    # one group contains fewer than 56 usable stimuli.

    black_indices = rng.choice(
        black.index.to_numpy(),
        size=target_black,
        replace=True,
    )


    white_indices = rng.choice(
        white.index.to_numpy(),
        size=target_white,
        replace=True,
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
) -> dict:

    target_white = (
        MATRIX_SIZE
        - target_black
    )

    return {

        "matrix_size":
            MATRIX_SIZE,

        "grid_rows":
            GRID_ROWS,

        "grid_columns":
            GRID_COLS,

        "cell_px":
            CELL_PX,

        "social_images": {
            "with_hair":
                "face.jpg",

            "no_hair":
                "face_no_hair.jpg",
        },

        "face_with_hair_zoom":
            FACE_WITH_HAIR_ZOOM,

        "face_no_hair_padding":
            FACE_NO_HAIR_PADDING,

        "target_black_count":
            target_black,

        "target_white_count":
            target_white,

        "black_percentage":
            100
            * target_black
            / MATRIX_SIZE,

        "white_percentage":
            100
            * target_white
            / MATRIX_SIZE,

        "folder_percentage":
            round(
                100
                * target_black
                / MATRIX_SIZE
            ),

        "version":
            version,

        "within_group_luminance_scale":
            WITHIN_GROUP_LUMINANCE_SCALE,

        "mean_black_face_luminance":
            float(
                cells.loc[
                    cells["group"]
                    == "black",
                    "luminance_mean",
                ].mean()
            ),

        "mean_white_face_luminance":
            float(
                cells.loc[
                    cells["group"]
                    == "white",
                    "luminance_mean",
                ].mean()
            ),

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
) -> None:
    """
    Generate one social matrix and both non-social matrices.
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


    output_directory.mkdir(
        parents=True,
        exist_ok=True,
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
        / "face.jpg",
        quality=95,
    )


    # --------------------------------------------------------
    # Social — face-only version WITHOUT hair
    # --------------------------------------------------------

    face_matrix_no_hair = (
        build_face_matrix_no_hair(
            cells
        )
    )

    face_matrix_no_hair.save(
        output_directory
        / "face_no_hair.jpg",
        quality=95,
    )


    # --------------------------------------------------------
    # Blue / green
    # --------------------------------------------------------

    blue_green = build_circle_matrix(
        cells,
        scheme="blue_green",
    )


    blue_green.save(
        output_directory
        / "circles_blue_green.jpg",
        quality=95,
    )


    # --------------------------------------------------------
    # Green / blue
    # --------------------------------------------------------

    green_blue = build_circle_matrix(
        cells,
        scheme="green_blue",
    )


    green_blue.save(
        output_directory
        / "circles_green_blue.jpg",
        quality=95,
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


    rng = np.random.default_rng(
        RANDOM_SEED
    )


    total = (
        len(TARGET_COUNTS)
        * N_VERSIONS
    )


    generated = 0


    for target_black in TARGET_COUNTS:

        for version in range(
            1,
            N_VERSIONS + 1,
        ):

            generate_one_matrix(
                data=data,
                target_black=target_black,
                version=version,
                rng=rng,
            )

            generated += 1

            print(
                f"[{generated:02d}/{total}] "
                f"Black={target_black:02d}/64 | "
                f"version={version:02d}"
            )


    print(
        "\nMatrix generation complete."
    )

    print(
        f"Generated social matrices: "
        f"{total}"
    )

    print(
        f"Generated non-social matrices: "
        f"{total * 2}"
    )

    print(
        f"Output directory: "
        f"{OUTPUT_DIR}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    generate_matrices()