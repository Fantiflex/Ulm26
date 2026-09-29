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
    / "stimuli.csv"
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
    """Load stimuli and corresponding face-luminance measurements."""

    stimuli = pd.read_csv(STIMULI_PATH)
    luminance = pd.read_csv(FACE_LUMINANCE_PATH)

    # Keep only successfully detected faces
    if "face_detected" in luminance.columns:
        luminance = luminance.loc[
            luminance["face_detected"] == True
        ].copy()

    # Match using filename rather than machine-specific absolute path
    stimuli["image_name"] = (
        stimuli["image_path"]
        .astype(str)
        .map(lambda path: Path(path).stem)
    )

    luminance["image_name"] = (
        luminance["image_path"]
        .astype(str)
        .map(lambda path: Path(path).stem)
    )

    data = stimuli.merge(
        luminance[
            [
                "image_name",
                "luminance_mean",
                "luminance_median",
            ]
        ],
        on="image_name",
        how="inner",
        validate="one_to_one",
    )

    if data.empty:
        raise RuntimeError(
            "No stimuli matched the face-luminance measurements."
        )

    if "ethnicity_perceived" not in data.columns:
        raise ValueError(
            "stimuli.csv must contain 'ethnicity_perceived'."
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