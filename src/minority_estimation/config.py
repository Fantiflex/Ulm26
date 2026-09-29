"""
Central configuration for the Minority Estimation experiment.

This module contains parameters that define stimulus selection,
face processing, and final matrix generation.

Keeping these values in one place ensures that all preprocessing
steps use the same experimental configuration.
"""

from typing import Final


# ============================================================
# STIMULUS SELECTION
# ============================================================

GENDERS: Final[tuple[str, ...]] = ("M",)

ETHNICITIES_SELF: Final[tuple[str, ...] | None] = None

ETHNICITIES_PERCEIVED: Final[tuple[str, ...]] = (
    "Black",
    "White",
)

ETHNICITY_SELECTION_MODE: Final[str] = "perceived"

MINIMUM_PERCEIVED_PROBABILITY: Final[float] = 0.90
MINIMUM_PERCEIVED_GENDER: Final[float] = 0.90


# ============================================================
# FACE PROCESSING
# ============================================================
CELL_PX: Final[int] = 100
FACE_WITH_HAIR_ZOOM: Final[float] = 1.10

FACE_CENTER_Y_RATIO: Final[float] = 0.42

LUMINANCE_RED_WEIGHT: Final[float] = 0.299
LUMINANCE_GREEN_WEIGHT: Final[float] = 0.587
LUMINANCE_BLUE_WEIGHT: Final[float] = 0.114


# ============================================================
# MEDIAPIPE FACE DETECTION
# ============================================================

NUM_FACES: Final[int] = 1

MIN_FACE_DETECTION_CONFIDENCE: Final[float] = 0.50
MIN_FACE_PRESENCE_CONFIDENCE: Final[float] = 0.50
MIN_TRACKING_CONFIDENCE: Final[float] = 0.50


# ============================================================
# MATRIX GENERATION
# ============================================================

GRID_ROWS: Final[int] = 8
GRID_COLS: Final[int] = 8

MATRIX_SIZE: Final[int] = GRID_ROWS * GRID_COLS

TARGET_BLACK_COUNTS: Final[tuple[int, ...]] = (
    8,
    14,
    20,
    26,
    32,
    38,
    44,
    50,
    56,
)

N_VERSIONS: Final[int] = 5

RANDOM_SEED: Final[int] = 42


# ============================================================
# NON-SOCIAL STIMULI
# ============================================================



BACKGROUND_GRAY_VALUE: Final[int] = 255


# ============================================================
# INTERNAL CONSISTENCY CHECKS
# ============================================================

if MATRIX_SIZE != 64:
    raise ValueError(
        "Experimental matrix size must contain exactly 64 cells."
    )

if not all(
    0 <= count <= MATRIX_SIZE
    for count in TARGET_BLACK_COUNTS
):
    raise ValueError(
        "All target Black-face counts must fall within matrix bounds."
    )

if not (
    0.0 <= MINIMUM_PERCEIVED_PROBABILITY <= 1.0
):
    raise ValueError(
        "MINIMUM_PERCEIVED_PROBABILITY must be between 0 and 1."
    )

if not (
    0.0 <= MINIMUM_PERCEIVED_GENDER <= 1.0
):
    raise ValueError(
        "MINIMUM_PERCEIVED_GENDER must be between 0 and 1."
    )