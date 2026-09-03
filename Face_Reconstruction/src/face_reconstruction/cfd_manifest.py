"""
Utilities for loading and harmonising Chicago Face Database metadata.

This module reads the official CFD Excel workbook and converts the original
norming data into a standardised manifest used by the reconstruction pipeline.
"""

from pathlib import Path

import pandas as pd

from face_reconstruction.schemas import require_columns


CFD_SHEET_NAME = "CFD U.S. Norming Data"

CFD_COLUMN_MAPPING = {
    "Model": "face_id",
    "GenderSelf": "gender_self",
    "EthnicitySelf": "ethnicity_self",
    "AgeSelf": "age_self",
    "AsianProb": "asian_prob",
    "ChineseAsianProb": "chinese_asian_prob",
    "JapaneseAsianProb": "japanese_asian_prob",
    "IndianAsianProb": "indian_asian_prob",
    "OtherAsianProb": "other_asian_prob",
    "MiddleEasternProb": "middle_eastern_prob",
    "BlackProb": "black_prob",
    "LatinoProb": "latino_prob",
    "MultiProb": "multi_prob",
    "OtherProb": "other_prob",
    "WhiteProb": "white_prob",
}

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

PERCEIVED_ETHNICITY_COLUMNS = {
    "Asian": "asian_prob",
    "Middle Eastern": "middle_eastern_prob",
    "Black": "black_prob",
    "Latino": "latino_prob",
    "Multiracial": "multi_prob",
    "Other": "other_prob",
    "White": "white_prob",
}


def load_cfd_data(
    workbook_path: Path,
    sheet_name: str = CFD_SHEET_NAME,
    header: int = 7,
) -> pd.DataFrame:
    """
    Load CFD norming data from the official Excel workbook.

    Parameters
    ----------
    workbook_path:
        Path to the official CFD Excel workbook.

    sheet_name:
        Name of the worksheet containing the norming data.

    header:
        Zero-based row index containing the column names.

    Returns
    -------
    pd.DataFrame
        Raw CFD norming data.

    Raises
    ------
    FileNotFoundError
        If the workbook does not exist.

    ValueError
        If the requested worksheet cannot be read.
    """

    if not workbook_path.exists():
        raise FileNotFoundError(
            f"CFD workbook not found: {workbook_path}"
        )

    try:
        cfd_data = pd.read_excel(
            workbook_path,
            sheet_name=sheet_name,
            header=header,
        )
    except ValueError as error:
        raise ValueError(
            f"Could not read worksheet '{sheet_name}' "
            f"from {workbook_path}."
        ) from error


    return cfd_data


def harmonize_cfd_manifest(
    cfd_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert raw CFD metadata into a standardised research manifest.

    The function preserves self-reported ethnicity, keeps perceived-ethnicity
    probability columns, and derives the dominant perceived-ethnicity
    category without removing self/perceived mismatches.

    Parameters
    ----------
    cfd_data:
        Raw table loaded from the CFD norming worksheet.

    Returns
    -------
    pd.DataFrame
        Harmonised CFD manifest with one row per model.

    Raises
    ------
    ValueError
        If required CFD columns are missing.
    """

    require_columns(
        dataframe=cfd_data,
        required_columns=CFD_REQUIRED_COLUMNS,
        dataframe_name="raw CFD data",
    )

    available_columns = [
        column
        for column in CFD_COLUMN_MAPPING
        if column in cfd_data.columns
    ]

    manifest = (
        cfd_data[available_columns]
        .rename(columns=CFD_COLUMN_MAPPING)
        .copy()
    )

    manifest = manifest.dropna(
        subset=["face_id"]
    )

    string_columns = [
        "face_id",
        "gender_self",
        "ethnicity_self",
    ]

    for column in string_columns:
        if column in manifest.columns:
            manifest[column] = (
                manifest[column]
                .astype("string")
                .str.strip()
            )

    probability_columns = [
        column
        for column in PERCEIVED_ETHNICITY_COLUMNS.values()
        if column in manifest.columns
    ]

    for column in probability_columns:
        manifest[column] = pd.to_numeric(
            manifest[column],
            errors="coerce",
        )

    has_perceived_data = (
        manifest[probability_columns]
        .notna()
        .any(axis=1)
    )

    dominant_probability_column = (
        manifest[probability_columns]
        .idxmax(axis=1)
    )

    probability_to_label = {
        probability_column: ethnicity_label
        for ethnicity_label, probability_column
        in PERCEIVED_ETHNICITY_COLUMNS.items()
    }

    manifest["ethnicity_perceived"] = (
        dominant_probability_column
        .map(probability_to_label)
        .where(has_perceived_data)
    )

    manifest["ethnicity_perceived_probability"] = (
        manifest[probability_columns]
        .max(axis=1)
        .where(has_perceived_data)
    )

    manifest["ethnicity_perceived_available"] = (
        has_perceived_data
    )

    manifest = manifest.sort_values(
        by="face_id"
    )

    return manifest.reset_index(drop=True)