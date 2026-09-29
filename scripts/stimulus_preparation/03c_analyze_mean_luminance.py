"""
Analyse face-region luminance by perceived racial group.

This script merges the selected CFD stimuli with the face-region
luminance measurements produced by 03_extract_faces.py, then reports
descriptive statistics and plots the luminance distributions.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STIMULI_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stimuli_matched.csv"
)

FACE_LUMINANCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "faces_only"
    / "face_luminance.csv"
)

SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "calibration"
    / "face_luminance_summary.csv"
)

FIGURE_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "stimulus_qc"
    / "face_luminance_distribution.png"
)


# ============================================================
# DATA LOADING
# ============================================================

def normalize_group(value: str) -> str:
    """Normalize perceived ethnicity to black or white."""

    value = str(value).strip().lower()

    if value in {"black", "b"}:
        return "black"

    if value in {"white", "w"}:
        return "white"

    raise ValueError(
        f"Unexpected perceived-ethnicity value: {value}"
    )


def load_data() -> pd.DataFrame:
    """
    Load selected stimuli and their face-luminance measurements.

    Every selected stimulus must have exactly one corresponding
    face-processing record, a successful face detection, and valid
    luminance measurements.
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

    if stimuli.empty:
        raise ValueError(
            "Stimulus table is empty."
        )

    if luminance.empty:
        raise ValueError(
            "Face-luminance table is empty."
        )

    # --------------------------------------------------------
    # Create machine-independent matching identifiers
    # --------------------------------------------------------

    stimuli["image_name"] = (
        stimuli["image_path"]
        .astype(str)
        .map(
            lambda path: Path(path).stem
        )
    )

    luminance["image_name"] = (
        luminance["image_path"]
        .astype(str)
        .map(
            lambda path: Path(path).stem
        )
    )

    # --------------------------------------------------------
    # Merge without silently dropping selected stimuli
    # --------------------------------------------------------

    data = stimuli.merge(
        luminance[
            [
                "image_name",
                "face_detected",
                "luminance_mean",
                "luminance_median",
            ]
        ],
        on="image_name",
        how="left",
        validate="one_to_one",
    )

    # A left join should preserve every selected stimulus.
    if len(data) != len(stimuli):
        raise RuntimeError(
            "Merge changed the number of selected stimuli: "
            f"{len(stimuli)} before merge, "
            f"{len(data)} after merge."
        )

    # --------------------------------------------------------
    # Check that every selected stimulus was processed
    # --------------------------------------------------------

    missing_processing = (
        data["face_detected"].isna()
    )

    if missing_processing.any():

        missing_names = (
            data.loc[
                missing_processing,
                "image_name",
            ]
            .tolist()
        )

        raise RuntimeError(
            "Some selected stimuli have no corresponding "
            "face-processing record: "
            f"{missing_names[:10]}"
        )

    # --------------------------------------------------------
    # Check face detection
    # --------------------------------------------------------

    failed_detection = (
        data["face_detected"] != True
    )

    if failed_detection.any():

        failed_names = (
            data.loc[
                failed_detection,
                "image_name",
            ]
            .tolist()
        )

        raise RuntimeError(
            f"Face detection failed for "
            f"{len(failed_names)} selected stimuli: "
            f"{failed_names[:10]}"
        )

    # --------------------------------------------------------
    # Check luminance values
    # --------------------------------------------------------

    missing_luminance = (
        data[
            [
                "luminance_mean",
                "luminance_median",
            ]
        ]
        .isna()
        .any(axis=1)
    )

    if missing_luminance.any():

        missing_names = (
            data.loc[
                missing_luminance,
                "image_name",
            ]
            .tolist()
        )

        raise RuntimeError(
            f"Missing luminance measurements for "
            f"{len(missing_names)} selected stimuli: "
            f"{missing_names[:10]}"
        )

    # --------------------------------------------------------
    # Experimental group
    # --------------------------------------------------------

    if "ethnicity_perceived" not in data.columns:
        raise ValueError(
            "stimuli.csv must contain "
            "'ethnicity_perceived'."
        )

    data["group"] = (
        data["ethnicity_perceived"]
        .map(normalize_group)
    )

    return data

# ============================================================
# ANALYSIS
# ============================================================

def main() -> None:

    data = load_data()

    print("\n=== FACE LUMINANCE DISTRIBUTIONS ===")

    summary = (
        data
        .groupby("group")["luminance_mean"]
        .agg(
            [
                "count",
                "mean",
                "std",
                "median",
                "min",
                "max",
            ]
        )
    )

    print(summary)

    SUMMARY_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        SUMMARY_OUTPUT_PATH
    )

    black_luminance = data.loc[
        data["group"] == "black",
        "luminance_mean",
    ]

    white_luminance = data.loc[
        data["group"] == "white",
        "luminance_mean",
    ]

    plt.figure(figsize=(9, 5))

    plt.hist(
        black_luminance,
        bins=20,
        alpha=0.5,
        label="Black-perceived faces",
    )

    plt.hist(
        white_luminance,
        bins=20,
        alpha=0.5,
        label="White-perceived faces",
    )

    plt.axvline(
        black_luminance.mean(),
        linestyle="--",
        label=f"Black mean = {black_luminance.mean():.1f}",
    )

    plt.axvline(
        white_luminance.mean(),
        linestyle="--",
        label=f"White mean = {white_luminance.mean():.1f}",
    )

    plt.xlabel("Mean face-region luminance")
    plt.ylabel("Number of faces")
    plt.title(
        "Distribution of face-region luminance "
        "by perceived group"
    )

    plt.legend()
    plt.tight_layout()
    FIGURE_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        FIGURE_OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )
    print(
        f"\nSummary saved to: "
        f"{SUMMARY_OUTPUT_PATH}"
    )

    print(
        f"Figure saved to: "
        f"{FIGURE_OUTPUT_PATH}"
    )

    plt.show()



if __name__ == "__main__":
    main()