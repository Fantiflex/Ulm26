"""Content-level validation for pipeline inputs and outputs."""

from pathlib import Path

import pandas as pd

from .schemas import PAIR_COLUMNS, RESPONSE_COLUMNS, require_columns


def validate_probability_range(
    dataframe: pd.DataFrame,
    probability_columns: list[str],
) -> None:
    """Check that probability columns are numeric and lie in [0, 1]."""

    require_columns(dataframe, probability_columns, "probability table")
    for column in probability_columns:
        numeric = pd.to_numeric(dataframe[column], errors="coerce")
        invalid = dataframe[column].notna() & numeric.isna()
        if invalid.any():
            raise ValueError(f"Column '{column}' contains non-numeric values.")
        outside = numeric.notna() & ~numeric.between(0.0, 1.0)
        if outside.any():
            raise ValueError(f"Column '{column}' contains values outside [0, 1].")


def validate_image_paths(stimuli: pd.DataFrame) -> None:
    """Check that all declared image paths exist and point to files."""

    require_columns(stimuli, ["face_id", "image_path"], "stimuli table")
    missing: list[str] = []
    for path_value in stimuli["image_path"].dropna():
        path = Path(path_value)
        if not path.exists() or not path.is_file():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image paths are invalid. Examples: {missing[:5]}"
        )


def validate_stimuli(stimuli: pd.DataFrame) -> None:
    """Validate identifiers, metadata, image availability, and paths."""

    require_columns(
        stimuli,
        [
            "face_id",
            "gender_self",
            "ethnicity_self",
            "ethnicity_perceived",
            "ethnicity_perceived_probability",
            "image_path",
            "image_exists",
        ],
        "stimuli table",
    )
    if stimuli.empty:
        raise ValueError("Stimuli table is empty.")
    if stimuli["face_id"].isna().any():
        raise ValueError("Some face IDs are missing.")
    if stimuli["face_id"].duplicated().any():
        duplicates = stimuli.loc[
            stimuli["face_id"].duplicated(keep=False), "face_id"
        ].unique().tolist()
        raise ValueError(f"Duplicate face IDs found: {duplicates[:10]}")
    if not stimuli["image_exists"].fillna(False).all():
        raise ValueError("Some selected stimuli do not have an associated image.")
    if stimuli["image_path"].duplicated().any():
        raise ValueError("The same image path is assigned to several face IDs.")
    validate_probability_range(
        stimuli,
        ["ethnicity_perceived_probability"],
    )
    validate_image_paths(stimuli)


def validate_pairs(
    pairs: pd.DataFrame,
    valid_face_ids: set[str],
) -> None:
    """Validate pair IDs, face membership, uniqueness, and self-comparisons."""

    require_columns(pairs, PAIR_COLUMNS, "pair table")
    if pairs.empty:
        raise ValueError("Pair table is empty.")
    if pairs["pair_id"].duplicated().any():
        raise ValueError("pair_id must be unique.")
    if (pairs["model_left"] == pairs["model_right"]).any():
        raise ValueError("Self-comparisons are not allowed.")

    observed = set(pairs["model_left"]) | set(pairs["model_right"])
    unknown = sorted(observed - set(valid_face_ids))
    if unknown:
        raise ValueError(f"Unknown face IDs found in pairs: {unknown[:10]}")

    unordered = pairs.apply(
        lambda row: tuple(sorted((row["model_left"], row["model_right"]))),
        axis=1,
    )
    if unordered.duplicated().any():
        raise ValueError("Duplicate unordered pairs were found.")


def validate_responses(responses: pd.DataFrame) -> None:
    """Validate participant binary-choice responses."""

    require_columns(responses, RESPONSE_COLUMNS, "response table")
    if responses.empty:
        raise ValueError("Response table is empty.")
    if responses[["participant_id", "trial_index"]].duplicated().any():
        raise ValueError("Each participant/trial combination must be unique.")

    valid_selection = (
        (responses["selected_model"] == responses["model_left"])
        | (responses["selected_model"] == responses["model_right"])
    )
    if not valid_selection.all():
        raise ValueError(
            "Every selected_model must equal model_left or model_right."
        )

    if (pd.to_numeric(responses["response_time_ms"], errors="coerce") < 0).any():
        raise ValueError("response_time_ms cannot be negative.")
