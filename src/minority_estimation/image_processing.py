"""
Shared image-processing utilities for stimulus preparation.
"""

from __future__ import annotations

from minority_estimation.config import (
    FACE_CENTER_Y_RATIO,
    FACE_WITH_HAIR_ZOOM,
)


def compute_face_crop_box(
    width: int,
    height: int,
    zoom: float = FACE_WITH_HAIR_ZOOM,
) -> tuple[int, int, int, int]:
    """
    Compute the crop box used for final face stimuli.

    Returns
    -------
    tuple[int, int, int, int]
        (left, top, right, bottom)
    """

    if width <= 0 or height <= 0:
        raise ValueError(
            "Image width and height must be positive."
        )

    if zoom <= 0:
        raise ValueError(
            "Zoom must be strictly positive."
        )

    crop_size = int(
        min(width, height) / zoom
    )

    center_x = width // 2

    center_y = int(
        height * FACE_CENTER_Y_RATIO
    )

    left = (
        center_x
        - crop_size // 2
    )

    top = (
        center_y
        - crop_size // 2
    )

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

    return (
        left,
        top,
        right,
        bottom,
    )