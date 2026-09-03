"""Stimulus indexing, image matching, and study-specific selection."""

from pathlib import Path
from typing import Literal

import re
import pandas as pd

from .schemas import STIMULUS_REQUIRED_COLUMNS, require_columns


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


ManifestMode = Literal["self", "perceived", "both"]


CFD_MODEL_PATTERN = re.compile(
    r"^CFD-([A-Za-z]{2}-\d{3})",
    flags=re.IGNORECASE,
)


def normalise_face_id(
    value: str,
) -> str:
    """
    Normalise a CFD face identifier for metadata-to-image matching.
    """

    return str(value).strip().lower()


def extract_cfd_model_id(
    image_stem: str,
) -> str:
    """
    Extract the CFD model identifier from an image filename stem.

    Examples
    --------
    CFD-BF-007-001-N -> CFD-BF-007
    CFD-AM-241-287-N -> CFD-AM-241
    """

    match = CFD_MODEL_PATTERN.match(
        str(image_stem).strip()
    )

    if match is None:
        raise ValueError(
            "Could not extract a CFD model ID from "
            f"image stem: {image_stem}"
        )

    return match.group(1).upper()


def build_image_index(
    image_directory: Path,
) -> pd.DataFrame:
    """
    Recursively index supported CFD image files.

    The model identifier is extracted from filenames such as
    ``CFD-BF-007-001-N.jpg`` and converted to ``BF-007``.
    """
    print("NEW build_image_index version is running")
    if not image_directory.exists():
        raise FileNotFoundError(
            f"Image directory not found: {image_directory}"
        )

    if not image_directory.is_dir():
        raise NotADirectoryError(
            f"Expected a directory: {image_directory}"
        )

    rows = []

    for image_path in sorted(
        image_directory.rglob("*")
    ):
        if not image_path.is_file():
            continue

        if (
            image_path.suffix.lower()
            not in SUPPORTED_IMAGE_EXTENSIONS
        ):
            continue

        try:
            model_id = extract_cfd_model_id(
                image_path.stem
            )
        except ValueError:
            continue

        rows.append(
            {
                "image_filename": image_path.name,
                "image_stem": image_path.stem,
                "image_path": image_path.as_posix(),
                "image_model_id": model_id,
                "image_face_id_normalised": (
                    normalise_face_id(model_id)
                ),
            }
        )

    image_index = pd.DataFrame(rows)

    if image_index.empty:
        raise ValueError(
            "No supported CFD images with valid model "
            f"identifiers were found in {image_directory}."
        )

    return image_index


def attach_images_to_manifest(
    manifest: pd.DataFrame,
    image_index: pd.DataFrame,
) -> pd.DataFrame:
    """Attach locally available image files to a harmonised manifest.

    Notes
    -----
    This exact-match version assumes the manifest ``face_id`` equals the image
    filename stem after normalisation. Adapt the matching rule if the CFD image
    filenames contain extra pose or expression suffixes.
    """

    require_columns(
        manifest,
        [
            "face_id",
            "gender_self",
            "ethnicity_self",
            "ethnicity_perceived",
            "ethnicity_perceived_probability",
        ],
        "harmonised manifest",
    )
    require_columns(
        image_index,
        [
            "image_filename",
            "image_stem",
            "image_path",
            "image_face_id_normalised",
        ],
        "image index",
    )

    duplicate_mask = image_index["image_face_id_normalised"].duplicated(
        keep=False
    )
    if duplicate_mask.any():
        examples = (
            image_index.loc[duplicate_mask, "image_face_id_normalised"]
            .drop_duplicates()
            .head(10)
            .tolist()
        )
        raise ValueError(
            "Several image files share the same normalised face ID. "
            f"Examples: {examples}"
        )

    result = manifest.copy()
    result["face_id_normalised"] = result["face_id"].map(normalise_face_id)

    result = result.merge(
        image_index,
        how="left",
        left_on="face_id_normalised",
        right_on="image_face_id_normalised",
        validate="one_to_one",
    )
    result["image_exists"] = result["image_path"].notna()

    return result.drop(
        columns=["image_face_id_normalised"],
        errors="ignore",
    )


def filter_stimuli(
    stimuli: pd.DataFrame,
    genders: list[str] | None = None,
    ethnicities_self: list[str] | None = None,
    ethnicities_perceived: list[str] | None = None,
    minimum_perceived_probability: float | None = None,
    ethnicity_selection_mode: ManifestMode = "self",
    require_existing_image: bool = True,
) -> pd.DataFrame:
    """Select stimuli according to reproducible study criteria."""

    require_columns(stimuli, STIMULUS_REQUIRED_COLUMNS, "stimuli table")

    if ethnicity_selection_mode not in {"self", "perceived", "both"}:
        raise ValueError(
            "ethnicity_selection_mode must be 'self', 'perceived', or 'both'."
        )

    if minimum_perceived_probability is not None and not (
        0.0 <= minimum_perceived_probability <= 1.0
    ):
        raise ValueError(
            "minimum_perceived_probability must be between 0 and 1."
        )

    selected = stimuli.copy()

    if require_existing_image:
        selected = selected.loc[selected["image_exists"].fillna(False)]

    if genders:
        selected = selected.loc[selected["gender_self"].isin(genders)]

    if ethnicity_selection_mode in {"self", "both"} and ethnicities_self:
        selected = selected.loc[
            selected["ethnicity_self"].isin(ethnicities_self)
        ]

    if ethnicity_selection_mode in {"perceived", "both"}:
        if ethnicities_perceived:
            selected = selected.loc[
                selected["ethnicity_perceived"].isin(ethnicities_perceived)
            ]
        if minimum_perceived_probability is not None:
            selected = selected.loc[
                selected["ethnicity_perceived_probability"]
                >= minimum_perceived_probability
            ]

    selected = selected.copy()
    selected["selected_for_study"] = True
    return selected.reset_index(drop=True)


def build_stimuli_table(
    manifest: pd.DataFrame,
    image_directory: Path,
    genders: list[str] | None = None,
    ethnicities_self: list[str] | None = None,
    ethnicities_perceived: list[str] | None = None,
    minimum_perceived_probability: float | None = None,
    ethnicity_selection_mode: ManifestMode = "self",
) -> pd.DataFrame:
    """Index images, attach them to metadata, and apply study filters."""

    image_index = build_image_index(image_directory)
    stimuli = attach_images_to_manifest(manifest, image_index)
    print("Manifest rows:", len(manifest))
    print("Indexed images:", len(image_index))

    print("\nFirst manifest IDs:")
    print(manifest["face_id"].head(10).tolist())

    print("\nFirst image model IDs:")
    print(
        image_index[
            [
                "image_filename",
                "image_model_id",
                "image_face_id_normalised",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )
    print(
        "\nMatched images:",
        int(stimuli["image_exists"].sum()),
    )
    return filter_stimuli(
        stimuli=stimuli,
        genders=genders,
        ethnicities_self=ethnicities_self,
        ethnicities_perceived=ethnicities_perceived,
        minimum_perceived_probability=minimum_perceived_probability,
        ethnicity_selection_mode=ethnicity_selection_mode,
        require_existing_image=True,
    )
