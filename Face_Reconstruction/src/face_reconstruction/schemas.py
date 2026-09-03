"""Data schemas and generic structural validation utilities."""

from collections.abc import Iterable

import pandas as pd


CFD_REQUIRED_COLUMNS = [
    "Model",
    "GenderSelf",
    "EthnicitySelf",
    "AsianProb",
    "MiddleEasternProb",
    "BlackProb",
    "LatinoProb",
    "MultiProb",
    "OtherProb",
    "WhiteProb",
]

STIMULUS_REQUIRED_COLUMNS = [
    "face_id",
    "gender_self",
    "ethnicity_self",
    "ethnicity_perceived",
    "ethnicity_perceived_probability",
    "image_path",
    "image_exists",
]

PAIR_COLUMNS = [
    "pair_id",
    "model_left",
    "model_right",
    "pair_order",
]

RESPONSE_COLUMNS = [
    "participant_id",
    "trial_index",
    "pair_id",
    "model_left",
    "model_right",
    "selected_model",
    "response_time_ms",
]

SCORE_COLUMNS = [
    "participant_id",
    "face_id",
    "n_appearances",
    "n_wins",
    "theta",
    "rank",
    "weight",
]


def require_columns(
    dataframe: pd.DataFrame,
    required_columns: Iterable[str],
    dataframe_name: str,
) -> None:
    """Check that a DataFrame contains all required columns.

    Parameters
    ----------
    dataframe:
        DataFrame to inspect.
    required_columns:
        Column names that must be present.
    dataframe_name:
        Human-readable name used in error messages.

    Raises
    ------
    TypeError
        If ``dataframe`` is not a pandas DataFrame.
    ValueError
        If one or more required columns are missing.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"{dataframe_name} must be a pandas DataFrame.")

    missing = sorted(set(required_columns) - set(dataframe.columns))
    if missing:
        raise ValueError(
            f"{dataframe_name} is missing required columns: {missing}"
        )
