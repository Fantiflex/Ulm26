"""
Generate pairwise face comparisons for the participant experiment.

This script loads the selected stimuli, creates a reproducible set of unique
face pairs, randomises left-right presentation, validates the comparison
design, and saves pair-level diagnostics.
"""

from pathlib import Path

import pandas as pd

from face_reconstruction.diagnostics import (
    compute_appearance_counts,
    compute_exposure_imbalance,
)
from face_reconstruction.pairs import (
    add_pair_ids,
    generate_all_pairs,
    generate_balanced_pairs,
    randomise_pair_sides,
    sample_pairs,
)
from face_reconstruction.validation import (
    validate_pairs,
)


STIMULI_PATH = Path(
    "data/processed/stimuli.csv"
)

OUTPUT_PATH = Path(
    "data/processed/pairs.csv"
)

PAIR_GENERATION_METHOD = "balanced"

N_PAIRS = 300

RANDOM_STATE = 42


def load_stimuli(
    stimuli_path: Path,
) -> pd.DataFrame:
    """
    Load the final stimulus table used for pair generation.

    Parameters
    ----------
    stimuli_path:
        Path to the selected stimulus CSV file.

    Returns
    -------
    pd.DataFrame
        Study-specific stimulus table.
    """

    if not stimuli_path.exists():
        raise FileNotFoundError(
            "Stimulus table not found: "
            f"{stimuli_path}"
        )

    stimuli = pd.read_csv(
        stimuli_path
    )

    if "face_id" not in stimuli.columns:
        raise ValueError(
            "Stimulus table must contain "
            "a 'face_id' column."
        )

    return stimuli


def generate_pairs(
    face_ids: list[str],
    method: str,
    n_pairs: int | None,
    random_state: int,
) -> pd.DataFrame:
    """
    Generate pairs using the requested experimental-design method.

    Parameters
    ----------
    face_ids:
        Unique face identifiers available for the experiment.

    method:
        Pair-generation strategy. Supported values are ``all``, ``sample``,
        and ``balanced``.

    n_pairs:
        Number of pairs requested for sampled or balanced designs.

    random_state:
        Seed used for reproducible random generation.

    Returns
    -------
    pd.DataFrame
        Generated unordered face pairs.

    Raises
    ------
    ValueError
        If the method is unsupported or if ``n_pairs`` is missing when needed.
    """

    if method == "all":
        return generate_all_pairs(
            face_ids
        )

    if n_pairs is None:
        raise ValueError(
            "n_pairs is required for sampled "
            "and balanced pair generation."
        )

    if method == "sample":
        return sample_pairs(
            face_ids=face_ids,
            n_pairs=n_pairs,
            random_state=random_state,
        )

    if method == "balanced":
        return generate_balanced_pairs(
            face_ids=face_ids,
            n_pairs=n_pairs,
            random_state=random_state,
        )

    raise ValueError(
        "Unsupported pair-generation method: "
        f"{method}"
    )


def build_pair_diagnostics(
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a face-level exposure diagnostic table.

    Parameters
    ----------
    pairs:
        Final pair table with left and right face assignments.

    Returns
    -------
    pd.DataFrame
        Number of appearances for each face.
    """

    appearance_counts = (
        compute_appearance_counts(
            pairs
        )
        .rename("n_appearances")
        .rename_axis("face_id")
        .reset_index()
        .sort_values(
            by=[
                "n_appearances",
                "face_id",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )

    return appearance_counts.reset_index(
        drop=True
    )


def save_pair_outputs(
    pairs: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save the final pair table and exposure diagnostics.

    Parameters
    ----------
    pairs:
        Validated pair table.

    output_path:
        CSV path for the pair table.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pairs.to_csv(
        output_path,
        index=False,
    )

    diagnostics = build_pair_diagnostics(
        pairs
    )

    diagnostics.to_csv(
        output_path.parent
        / "pair_exposure_diagnostics.csv",
        index=False,
    )


def print_pair_summary(
    pairs: pd.DataFrame,
) -> None:
    """
    Print pair-design statistics to the terminal.

    Parameters
    ----------
    pairs:
        Final validated pair table.
    """

    imbalance = compute_exposure_imbalance(
        pairs
    )

    print("\nPair generation")
    print("---------------")
    print(
        f"Number of pairs: {len(pairs)}"
    )

    print(
        "Number of represented faces: "
        f"{pd.unique(
            pd.concat(
                [
                    pairs['model_left'],
                    pairs['model_right'],
                ]
            )
        ).size}"
    )

    print("\nExposure imbalance:")
    for metric, value in imbalance.items():
        print(f"- {metric}: {value}")


def main() -> None:
    """
    Generate, validate, randomise, and save experimental face pairs.
    """

    stimuli = load_stimuli(
        STIMULI_PATH
    )

    face_ids = (
        stimuli["face_id"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    if len(face_ids) < 2:
        raise ValueError(
            "At least two unique stimuli are "
            "required to generate pairs."
        )

    pairs = generate_pairs(
        face_ids=face_ids,
        method=PAIR_GENERATION_METHOD,
        n_pairs=N_PAIRS,
        random_state=RANDOM_STATE,
    )

    pairs = randomise_pair_sides(
        pairs=pairs,
        random_state=RANDOM_STATE,
    )

    pairs = add_pair_ids(
        pairs
    )

    validate_pairs(
        pairs=pairs,
        valid_face_ids=set(face_ids),
    )

    save_pair_outputs(
        pairs=pairs,
        output_path=OUTPUT_PATH,
    )

    print_pair_summary(
        pairs
    )

    print(
        "\nPair table saved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()