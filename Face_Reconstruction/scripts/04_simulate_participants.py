"""
Simulate participant choices under a Bradley-Terry model.

This script generates latent face-preference scores, simulates pairwise
participant responses, and saves both the ground truth and synthetic response
data for model-validation experiments.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from face_reconstruction.simulation import (
    generate_latent_scores,
    simulate_responses,
)


PAIRS_PATH = Path(
    "data/processed/pairs.csv"
)

OUTPUT_DIRECTORY = Path(
    "data/simulated"
)

N_PARTICIPANTS = 20

RANDOM_STATE = 42


def load_pairs(
    pairs_path: Path,
) -> pd.DataFrame:
    """
    Load the pair table used for response simulation.

    Parameters
    ----------
    pairs_path:
        Path to the experimental pair table.

    Returns
    -------
    pd.DataFrame
        Pair table containing left and right face identifiers.
    """

    if not pairs_path.exists():
        raise FileNotFoundError(
            f"Pair table not found: {pairs_path}"
        )

    pairs = pd.read_csv(
        pairs_path
    )

    required_columns = {
        "model_left",
        "model_right",
    }

    missing_columns = (
        required_columns
        - set(pairs.columns)
    )

    if missing_columns:
        raise ValueError(
            "Pair table is missing columns: "
            f"{sorted(missing_columns)}"
        )

    return pairs


def extract_face_ids(
    pairs: pd.DataFrame,
) -> list[str]:
    """
    Extract all unique face identifiers represented in the pair table.

    Parameters
    ----------
    pairs:
        Experimental pair table.

    Returns
    -------
    list[str]
        Sorted list of unique face identifiers.
    """

    face_ids = pd.concat(
        [
            pairs["model_left"],
            pairs["model_right"],
        ],
        ignore_index=True,
    )

    return sorted(
        face_ids
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


def simulate_multiple_participants(
    pairs: pd.DataFrame,
    latent_scores: pd.DataFrame,
    n_participants: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Simulate pairwise responses for multiple participants.

    Parameters
    ----------
    pairs:
        Pair table presented to each simulated participant.

    latent_scores:
        Ground-truth latent score table.

    n_participants:
        Number of participants to simulate.

    random_state:
        Master seed used to derive participant-level random seeds.

    Returns
    -------
    pd.DataFrame
        Combined synthetic response table.
    """

    if n_participants <= 0:
        raise ValueError(
            "n_participants must be positive."
        )

    random_generator = (
        np.random.default_rng(
            random_state
        )
    )

    participant_seeds = (
        random_generator.integers(
            low=0,
            high=np.iinfo(np.int32).max,
            size=n_participants,
        )
    )

    response_tables = []

    for participant_index, seed in enumerate(
        participant_seeds,
        start=1,
    ):
        participant_id = (
            f"SIM_{participant_index:03d}"
        )

        participant_responses = (
            simulate_responses(
                pairs=pairs,
                latent_scores=latent_scores,
                participant_id=participant_id,
                random_state=int(seed),
            )
        )

        response_tables.append(
            participant_responses
        )

    return pd.concat(
        response_tables,
        ignore_index=True,
    )


def save_simulation_outputs(
    latent_scores: pd.DataFrame,
    responses: pd.DataFrame,
    output_directory: Path,
) -> None:
    """
    Save ground-truth scores and simulated participant responses.

    Parameters
    ----------
    latent_scores:
        Ground-truth latent face scores.

    responses:
        Combined simulated participant responses.

    output_directory:
        Directory in which simulation outputs are stored.
    """

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    latent_scores.to_csv(
        output_directory
        / "true_latent_scores.csv",
        index=False,
    )

    responses.to_csv(
        output_directory
        / "simulated_responses.csv",
        index=False,
    )


def print_simulation_summary(
    latent_scores: pd.DataFrame,
    responses: pd.DataFrame,
) -> None:
    """
    Print the main characteristics of the simulated dataset.

    Parameters
    ----------
    latent_scores:
        Ground-truth score table.

    responses:
        Simulated participant responses.
    """

    print("\nParticipant simulation")
    print("----------------------")
    print(
        f"Number of faces: "
        f"{latent_scores['face_id'].nunique()}"
    )

    print(
        f"Number of participants: "
        f"{responses['participant_id'].nunique()}"
    )

    print(
        f"Number of responses: "
        f"{len(responses)}"
    )

    print("\nTrue score summary:")
    print(
        latent_scores["theta_true"]
        .describe()
    )


def main() -> None:
    """
    Generate and save a reproducible synthetic Bradley-Terry dataset.
    """

    pairs = load_pairs(
        PAIRS_PATH
    )

    face_ids = extract_face_ids(
        pairs
    )

    latent_scores = (
        generate_latent_scores(
            face_ids=face_ids,
            random_state=RANDOM_STATE,
        )
    )

    responses = (
        simulate_multiple_participants(
            pairs=pairs,
            latent_scores=latent_scores,
            n_participants=N_PARTICIPANTS,
            random_state=RANDOM_STATE,
        )
    )

    save_simulation_outputs(
        latent_scores=latent_scores,
        responses=responses,
        output_directory=OUTPUT_DIRECTORY,
    )

    print_simulation_summary(
        latent_scores=latent_scores,
        responses=responses,
    )

    print(
        "\nSimulation outputs saved to: "
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()