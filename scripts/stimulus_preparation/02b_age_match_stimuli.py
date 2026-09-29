"""
Perform optimal one-to-one age matching between eligible
Black- and White-perceived face stimuli.

All eligible Black stimuli are retained. The same number of
White stimuli is selected without replacement so as to minimize
the total absolute age difference across matched pairs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stimuli.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "stimuli_matched.csv"
)

MATCHING_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "matching"
)

PAIRS_OUTPUT_PATH = (
    MATCHING_DIRECTORY
    / "age_matching_pairs.csv"
)

SUMMARY_OUTPUT_PATH = (
    MATCHING_DIRECTORY
    / "age_matching_summary.json"
)


# ============================================================
# MATCHING PARAMETERS
# ============================================================

AGE_COLUMN = "age_rated"

GROUP_COLUMN = "ethnicity_perceived"

BLACK_LABEL = "Black"
WHITE_LABEL = "White"


# ============================================================
# HELPERS
# ============================================================

def standardized_mean_difference(
    group_a: pd.Series,
    group_b: pd.Series,
) -> float:
    """
    Compute the standardized mean difference between two groups.
    """

    mean_difference = (
        float(group_a.mean())
        - float(group_b.mean())
    )

    pooled_sd = np.sqrt(
        (
            float(group_a.var(ddof=1))
            + float(group_b.var(ddof=1))
        )
        / 2
    )

    if pooled_sd == 0:
        return 0.0

    return float(
        mean_difference / pooled_sd
    )


def load_stimuli() -> pd.DataFrame:
    """
    Load and validate the eligible stimulus pool.
    """

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Stimulus table not found: {INPUT_PATH}"
        )

    stimuli = pd.read_csv(
        INPUT_PATH
    )

    required_columns = {
        "face_id",
        GROUP_COLUMN,
        AGE_COLUMN,
    }

    missing_columns = (
        required_columns
        - set(stimuli.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if stimuli.empty:
        raise ValueError(
            "Stimulus table is empty."
        )

    if stimuli["face_id"].duplicated().any():
        duplicates = (
            stimuli.loc[
                stimuli["face_id"].duplicated(
                    keep=False
                ),
                "face_id",
            ]
            .tolist()
        )

        raise ValueError(
            "Duplicate face IDs found: "
            f"{duplicates[:10]}"
        )

    if stimuli[AGE_COLUMN].isna().any():
        missing_age_ids = (
            stimuli.loc[
                stimuli[AGE_COLUMN].isna(),
                "face_id",
            ]
            .tolist()
        )

        raise ValueError(
            "Missing age values for selected stimuli: "
            f"{missing_age_ids[:10]}"
        )

    stimuli[AGE_COLUMN] = pd.to_numeric(
        stimuli[AGE_COLUMN],
        errors="raise",
    )

    return stimuli


def split_groups(
    stimuli: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split stimuli into Black- and White-perceived pools.
    """

    black = (
        stimuli.loc[
            stimuli[GROUP_COLUMN]
            .astype(str)
            .str.strip()
            .str.lower()
            == BLACK_LABEL.lower()
        ]
        .copy()
        .reset_index(drop=True)
    )

    white = (
        stimuli.loc[
            stimuli[GROUP_COLUMN]
            .astype(str)
            .str.strip()
            .str.lower()
            == WHITE_LABEL.lower()
        ]
        .copy()
        .reset_index(drop=True)
    )

    if black.empty:
        raise RuntimeError(
            "No Black-perceived stimuli found."
        )

    if white.empty:
        raise RuntimeError(
            "No White-perceived stimuli found."
        )

    if len(white) < len(black):
        raise RuntimeError(
            "The White candidate pool is smaller "
            "than the Black candidate pool."
        )

    return black, white


def perform_optimal_matching(
    black: pd.DataFrame,
    white: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform minimum-cost one-to-one matching on absolute age difference.
    """

    black_ages = (
        black[AGE_COLUMN]
        .to_numpy(dtype=float)
    )

    white_ages = (
        white[AGE_COLUMN]
        .to_numpy(dtype=float)
    )

    cost_matrix = np.abs(
        black_ages[:, None]
        - white_ages[None, :]
    )

    black_indices, white_indices = (
        linear_sum_assignment(
            cost_matrix
        )
    )

    matched_black = (
        black.iloc[
            black_indices
        ]
        .copy()
        .reset_index(drop=True)
    )

    matched_white = (
        white.iloc[
            white_indices
        ]
        .copy()
        .reset_index(drop=True)
    )

    pairs = pd.DataFrame(
        {
            "pair_id": np.arange(
                1,
                len(matched_black) + 1,
            ),
            "black_face_id":
                matched_black["face_id"].to_numpy(),
            "white_face_id":
                matched_white["face_id"].to_numpy(),
            "black_age":
                matched_black[AGE_COLUMN].to_numpy(),
            "white_age":
                matched_white[AGE_COLUMN].to_numpy(),
        }
    )

    pairs["absolute_age_difference"] = (
        pairs["black_age"]
        - pairs["white_age"]
    ).abs()

    final_stimuli = pd.concat(
        [
            matched_black,
            matched_white,
        ],
        ignore_index=True,
    )

    return final_stimuli, pairs


def build_summary(
    black_candidates: pd.DataFrame,
    white_candidates: pd.DataFrame,
    final_stimuli: pd.DataFrame,
    pairs: pd.DataFrame,
) -> dict:
    """
    Build matching diagnostics for reproducibility.
    """

    final_black = final_stimuli.loc[
        final_stimuli[GROUP_COLUMN]
        .astype(str)
        .str.lower()
        == BLACK_LABEL.lower(),
        AGE_COLUMN,
    ]

    final_white = final_stimuli.loc[
        final_stimuli[GROUP_COLUMN]
        .astype(str)
        .str.lower()
        == WHITE_LABEL.lower(),
        AGE_COLUMN,
    ]

    return {
        "matching_method":
            "minimum_cost_one_to_one_assignment",
        "matching_variable":
            AGE_COLUMN,
        "distance_metric":
            "absolute_age_difference",
        "replacement":
            False,

        "n_black_candidates":
            int(len(black_candidates)),
        "n_white_candidates":
            int(len(white_candidates)),

        "n_black_final":
            int(len(final_black)),
        "n_white_final":
            int(len(final_white)),

        "black_age_mean":
            float(final_black.mean()),
        "black_age_sd":
            float(final_black.std(ddof=1)),

        "white_age_mean":
            float(final_white.mean()),
        "white_age_sd":
            float(final_white.std(ddof=1)),

        "mean_absolute_pair_age_difference":
            float(
                pairs[
                    "absolute_age_difference"
                ].mean()
            ),

        "median_absolute_pair_age_difference":
            float(
                pairs[
                    "absolute_age_difference"
                ].median()
            ),

        "max_absolute_pair_age_difference":
            float(
                pairs[
                    "absolute_age_difference"
                ].max()
            ),

        "standardized_mean_difference":
            standardized_mean_difference(
                final_black,
                final_white,
            ),
    }


def validate_final_matching(
    final_stimuli: pd.DataFrame,
    pairs: pd.DataFrame,
) -> None:
    """
    Validate structural properties of the matched stimulus set.
    """

    black_count = int(
        (
            final_stimuli[GROUP_COLUMN]
            .astype(str)
            .str.lower()
            == BLACK_LABEL.lower()
        ).sum()
    )

    white_count = int(
        (
            final_stimuli[GROUP_COLUMN]
            .astype(str)
            .str.lower()
            == WHITE_LABEL.lower()
        ).sum()
    )

    if black_count != white_count:
        raise RuntimeError(
            "Matched groups are not balanced: "
            f"Black={black_count}, "
            f"White={white_count}."
        )

    if len(final_stimuli) != 2 * black_count:
        raise RuntimeError(
            "Unexpected final stimulus count."
        )

    if final_stimuli["face_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate face IDs exist in the "
            "matched stimulus set."
        )

    if pairs["black_face_id"].duplicated().any():
        raise RuntimeError(
            "A Black stimulus was matched more than once."
        )

    if pairs["white_face_id"].duplicated().any():
        raise RuntimeError(
            "A White stimulus was matched more than once."
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Run the complete age-matching procedure.
    """

    stimuli = load_stimuli()

    black, white = split_groups(
        stimuli
    )

    print("\nCandidate pools")
    print("---------------")
    print(
        f"Black-perceived: {len(black)}"
    )
    print(
        f"White-perceived: {len(white)}"
    )

    final_stimuli, pairs = (
        perform_optimal_matching(
            black=black,
            white=white,
        )
    )

    validate_final_matching(
        final_stimuli=final_stimuli,
        pairs=pairs,
    )

    summary = build_summary(
        black_candidates=black,
        white_candidates=white,
        final_stimuli=final_stimuli,
        pairs=pairs,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    MATCHING_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_stimuli.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    pairs.to_csv(
        PAIRS_OUTPUT_PATH,
        index=False,
    )

    with SUMMARY_OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    print("\nFinal matched stimulus set")
    print("--------------------------")
    print(
        f"Total stimuli: "
        f"{len(final_stimuli)}"
    )

    print(
        final_stimuli[
            GROUP_COLUMN
        ].value_counts()
    )

    print(
        "\nMean absolute pair age difference: "
        f"{summary['mean_absolute_pair_age_difference']:.3f}"
    )

    print(
        "Standardized mean difference: "
        f"{summary['standardized_mean_difference']:.3f}"
    )

    print(
        f"\nMatched stimuli saved to: "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Pair audit saved to: "
        f"{PAIRS_OUTPUT_PATH}"
    )

    print(
        f"Matching summary saved to: "
        f"{SUMMARY_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()