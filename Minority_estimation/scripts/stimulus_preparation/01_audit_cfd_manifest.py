"""
Audit the original Chicago Face Database manifest.

This script loads and harmonises the CFD norming data, reports the available
self-reported demographic categories, examines perceived-ethnicity
probabilities, and saves audit tables for reproducibility.
"""
from face_reconstruction.stimuli import build_stimuli_table
from pathlib import Path

import pandas as pd

from face_reconstruction.cfd_manifest import (
    harmonize_cfd_manifest,
    load_cfd_data,
)


WORKBOOK_PATH = Path(
    "data/raw/cfd/CFD_codebook.xlsx"
)

OUTPUT_DIRECTORY = Path(
    "data/interim/cfd_audit"
)


def build_demographic_counts(
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """
    Count the number of CFD models in each gender-by-ethnicity group.

    Parameters
    ----------
    manifest:
        Harmonised CFD manifest containing self-reported gender and ethnicity.

    Returns
    -------
    pd.DataFrame
        Table with one row per demographic combination and a model count.
    """

    counts = (
        manifest.groupby(
            [
                "gender_self",
                "ethnicity_self",
            ],
            dropna=False,
        )
        .size()
        .rename("n_models")
        .reset_index()
        .sort_values(
            by=[
                "gender_self",
                "ethnicity_self",
            ]
        )
    )

    return counts.reset_index(drop=True)


def build_self_perceived_crosstab(
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare self-reported ethnicity with dominant perceived ethnicity.

    Parameters
    ----------
    manifest:
        Harmonised CFD manifest.

    Returns
    -------
    pd.DataFrame
        Cross-tabulation of self-reported and perceived ethnicity categories.
    """

    crosstab = pd.crosstab(
        index=manifest["ethnicity_self"],
        columns=manifest["ethnicity_perceived"],
        margins=True,
        dropna=False,
    )

    return crosstab


def build_missingness_report(
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarise missing values for every manifest column.

    Parameters
    ----------
    manifest:
        Harmonised CFD manifest.

    Returns
    -------
    pd.DataFrame
        Missing-value counts and percentages for each column.
    """

    n_rows = len(manifest)

    report = pd.DataFrame(
        {
            "column": manifest.columns,
            "n_missing": [
                manifest[column].isna().sum()
                for column in manifest.columns
            ],
        }
    )

    if n_rows == 0:
        report["missing_percentage"] = 0.0
    else:
        report["missing_percentage"] = (
            report["n_missing"]
            / n_rows
            * 100
        )

    return report.sort_values(
        by="missing_percentage",
        ascending=False,
    ).reset_index(drop=True)


def build_duplicate_report(
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify duplicated CFD face identifiers.

    Parameters
    ----------
    manifest:
        Harmonised CFD manifest.

    Returns
    -------
    pd.DataFrame
        Rows whose face identifier occurs more than once.
    """

    duplicates = manifest.loc[
        manifest["face_id"].duplicated(
            keep=False
        )
    ].copy()

    return duplicates.sort_values(
        by="face_id"
    ).reset_index(drop=True)


def save_audit_outputs(
    manifest: pd.DataFrame,
    output_directory: Path,
) -> None:
    """
    Save the harmonised manifest and audit summary tables.

    Parameters
    ----------
    manifest:
        Harmonised CFD manifest.

    output_directory:
        Directory in which audit outputs will be stored.
    """

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    demographic_counts = build_demographic_counts(
        manifest
    )

    self_perceived_crosstab = (
        build_self_perceived_crosstab(
            manifest
        )
    )

    missingness_report = build_missingness_report(
        manifest
    )

    duplicate_report = build_duplicate_report(
        manifest
    )

    manifest.to_csv(
        output_directory
        / "cfd_manifest_harmonised.csv",
        index=False,
    )

    demographic_counts.to_csv(
        output_directory
        / "demographic_counts.csv",
        index=False,
    )

    self_perceived_crosstab.to_csv(
        output_directory
        / "self_perceived_ethnicity_crosstab.csv",
    )

    missingness_report.to_csv(
        output_directory
        / "missingness_report.csv",
        index=False,
    )

    duplicate_report.to_csv(
        output_directory
        / "duplicate_face_ids.csv",
        index=False,
    )


def print_audit_summary(
    manifest: pd.DataFrame,
) -> None:
    """
    Print the main CFD audit results to the terminal.

    Parameters
    ----------
    manifest:
        Harmonised CFD manifest.
    """

    print("\nCFD manifest audit")
    print("------------------")
    print(f"Number of rows: {len(manifest)}")
    print(
        "Unique face IDs: "
        f"{manifest['face_id'].nunique()}"
    )

    print("\nSelf-reported gender:")
    print(
        manifest["gender_self"]
        .value_counts(dropna=False)
    )

    print("\nSelf-reported ethnicity:")
    print(
        manifest["ethnicity_self"]
        .value_counts(dropna=False)
    )

    print("\nDominant perceived ethnicity:")
    print(
        manifest["ethnicity_perceived"]
        .value_counts(dropna=False)
    )

    n_duplicates = (
        manifest["face_id"]
        .duplicated(keep=False)
        .sum()
    )

    print(
        "\nRows with duplicated face IDs: "
        f"{n_duplicates}"
    )


def main() -> None:
    """
    Run the complete CFD manifest audit.
    """

    raw_cfd_data = load_cfd_data(
        workbook_path=WORKBOOK_PATH
    )

    manifest = harmonize_cfd_manifest(
        raw_cfd_data
    )

    print_audit_summary(
        manifest
    )

    save_audit_outputs(
        manifest=manifest,
        output_directory=OUTPUT_DIRECTORY,
    )

    print(
        "\nAudit outputs saved to: "
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()