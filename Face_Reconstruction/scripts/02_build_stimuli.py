"""
Build the study-specific CFD stimulus table.

This script loads the harmonised CFD metadata, matches metadata rows with
local images, applies demographic and perceptual inclusion criteria, validates
the selected stimuli, and saves the resulting study manifest.
"""

from pathlib import Path

import pandas as pd

from face_reconstruction.cfd_manifest import (
    harmonize_cfd_manifest,
    load_cfd_data,
)
from face_reconstruction.stimuli import (
    build_stimuli_table,
)
from face_reconstruction.validation import (
    validate_stimuli,
)


WORKBOOK_PATH = Path(
    "data/raw/cfd/CFD_codebook.xlsx"
)

HARMONISED_MANIFEST_PATH = Path(
    "data/interim/cfd_audit/cfd_manifest_harmonised.csv"
)

IMAGE_DIRECTORY = Path(
    "data/raw/cfd/images"
)

OUTPUT_PATH = Path(
    "data/processed/stimuli.csv"
)

GENDERS = ["M"] #options are M, F, or None for all self-reported

ETHNICITIES_SELF = None # options are B, W, H, A, O, or None for all self-reported

ETHNICITIES_PERCEIVED = ["Black", "White"] # options are Black, White, Hispanic, Asian, Other, or None for all perceived

ETHNICITY_SELECTION_MODE = "perceived"

MINIMUM_PERCEIVED_PROBABILITY = 0.8


def build_selection_summary(
    stimuli: pd.DataFrame,
) -> pd.DataFrame:
    """
    Count selected stimuli by demographic and perceived categories.

    Parameters
    ----------
    stimuli:
        Final study-specific stimulus table.

    Returns
    -------
    pd.DataFrame
        Group counts for self-reported gender, self-reported ethnicity, and
        dominant perceived ethnicity.
    """

    grouping_columns = [
        "gender_self",
        "ethnicity_self",
        "ethnicity_perceived",
    ]

    available_grouping_columns = [
        column
        for column in grouping_columns
        if column in stimuli.columns
    ]

    summary = (
        stimuli.groupby(
            available_grouping_columns,
            dropna=False,
        )
        .size()
        .rename("n_stimuli")
        .reset_index()
        .sort_values(
            by=available_grouping_columns
        )
    )

    return summary.reset_index(drop=True)


def save_stimuli_outputs(
    stimuli: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save the selected stimuli and their demographic summary.

    Parameters
    ----------
    stimuli:
        Final validated stimulus table.

    output_path:
        CSV path for the final stimulus table.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    stimuli.to_csv(
        output_path,
        index=False,
    )

    selection_summary = (
        build_selection_summary(
            stimuli
        )
    )

    summary_path = (
        output_path.parent
        / "stimuli_selection_summary.csv"
    )

    selection_summary.to_csv(
        summary_path,
        index=False,
    )


def print_stimuli_summary(
    stimuli: pd.DataFrame,
) -> None:
    """
    Print a summary of the selected experimental stimuli.

    Parameters
    ----------
    stimuli:
        Final study-specific stimulus table.
    """

    print("\nStimulus selection")
    print("------------------")
    print(
        f"Selected stimuli: {len(stimuli)}"
    )

    if "gender_self" in stimuli.columns:
        print("\nGender:")
        print(
            stimuli["gender_self"]
            .value_counts(dropna=False)
        )

    if "ethnicity_self" in stimuli.columns:
        print("\nSelf-reported ethnicity:")
        print(
            stimuli["ethnicity_self"]
            .value_counts(dropna=False)
        )

    if (
        "ethnicity_perceived"
        in stimuli.columns
    ):
        print("\nPerceived ethnicity:")
        print(
            stimuli["ethnicity_perceived"]
            .value_counts(dropna=False)
        )


def main() -> None:
    """
    Build, validate, and save the study-specific stimulus table.
    """

    if HARMONISED_MANIFEST_PATH.exists():
        manifest = pd.read_csv(HARMONISED_MANIFEST_PATH)
    else:
        raw_cfd_data = load_cfd_data(
            workbook_path=WORKBOOK_PATH
        )

        manifest = harmonize_cfd_manifest(
            raw_cfd_data
        )

        HARMONISED_MANIFEST_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        manifest.to_csv(
            HARMONISED_MANIFEST_PATH,
            index=False,
        )

    stimuli = build_stimuli_table(
        manifest=manifest,
        image_directory=IMAGE_DIRECTORY,
        genders=GENDERS,
        ethnicities_self=ETHNICITIES_SELF,
        ethnicities_perceived=(
            ETHNICITIES_PERCEIVED
        ),
        minimum_perceived_probability=(
            MINIMUM_PERCEIVED_PROBABILITY
        ),
        ethnicity_selection_mode=(
            ETHNICITY_SELECTION_MODE
        ),
    )
    
    validate_stimuli(
        stimuli
    )

    save_stimuli_outputs(
        stimuli=stimuli,
        output_path=OUTPUT_PATH,
    )

    print_stimuli_summary(
        stimuli
    )

    print(
        "\nStimulus table saved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()