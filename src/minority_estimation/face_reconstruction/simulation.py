"""Participant-response simulation for validating pairwise models."""

import numpy as np
import pandas as pd

from .schemas import require_columns


def generate_latent_scores(
    face_ids: list[str],
    random_state: int,
    scale: float = 1.0,
) -> pd.DataFrame:
    """Generate centred ground-truth latent scores for a set of faces."""

    if len(set(face_ids)) != len(face_ids):
        raise ValueError("face_ids must be unique.")
    if len(face_ids) < 2:
        raise ValueError("At least two faces are required.")
    if scale <= 0:
        raise ValueError("scale must be positive.")

    rng = np.random.default_rng(random_state)
    theta = rng.normal(loc=0.0, scale=scale, size=len(face_ids))
    theta -= theta.mean()
    return pd.DataFrame({"face_id": face_ids, "theta_true": theta})


def choice_probability(theta_left: float, theta_right: float) -> float:
    """Compute the Bradley-Terry probability of choosing the left face."""

    difference = np.clip(theta_left - theta_right, -700.0, 700.0)
    return float(1.0 / (1.0 + np.exp(-difference)))


def simulate_responses(
    pairs: pd.DataFrame,
    latent_scores: pd.DataFrame,
    participant_id: str,
    random_state: int,
) -> pd.DataFrame:
    """Simulate one participant's choices from known latent scores."""

    require_columns(
        pairs,
        ["pair_id", "model_left", "model_right", "pair_order"],
        "pair table",
    )
    require_columns(
        latent_scores,
        ["face_id", "theta_true"],
        "latent score table",
    )

    theta_map = latent_scores.set_index("face_id")["theta_true"].to_dict()
    required_faces = set(pairs["model_left"]) | set(pairs["model_right"])
    missing = sorted(required_faces - set(theta_map))
    if missing:
        raise ValueError(f"Latent scores missing for faces: {missing[:10]}")

    rng = np.random.default_rng(random_state)
    rows: list[dict[str, object]] = []
    for row in pairs.itertuples(index=False):
        probability_left = choice_probability(
            theta_map[row.model_left], theta_map[row.model_right]
        )
        choose_left = bool(rng.random() < probability_left)
        rows.append(
            {
                "participant_id": participant_id,
                "trial_index": int(row.pair_order),
                "pair_id": row.pair_id,
                "model_left": row.model_left,
                "model_right": row.model_right,
                "selected_model": (
                    row.model_left if choose_left else row.model_right
                ),
                "response_time_ms": np.nan,
                "choice_probability_left": probability_left,
            }
        )

    return pd.DataFrame(rows)


def simulate_participants(
    pairs: pd.DataFrame,
    latent_scores: pd.DataFrame,
    n_participants: int,
    random_state: int,
) -> pd.DataFrame:
    """Simulate several participants using reproducible independent seeds."""

    if n_participants <= 0:
        raise ValueError("n_participants must be positive.")
    seed_sequence = np.random.SeedSequence(random_state)
    child_seeds = seed_sequence.spawn(n_participants)

    responses = []
    for index, child_seed in enumerate(child_seeds, start=1):
        seed = int(child_seed.generate_state(1)[0])
        responses.append(
            simulate_responses(
                pairs=pairs,
                latent_scores=latent_scores,
                participant_id=f"P{index:03d}",
                random_state=seed,
            )
        )
    return pd.concat(responses, ignore_index=True)


def inject_response_noise(
    responses: pd.DataFrame,
    error_probability: float,
    random_state: int,
) -> pd.DataFrame:
    """Flip a reproducible proportion of choices to model response lapses."""

    require_columns(
        responses,
        ["model_left", "model_right", "selected_model"],
        "response table",
    )
    if not 0.0 <= error_probability <= 1.0:
        raise ValueError("error_probability must be between 0 and 1.")

    rng = np.random.default_rng(random_state)
    result = responses.copy()
    flip = rng.random(len(result)) < error_probability
    selected_is_left = result["selected_model"] == result["model_left"]
    result.loc[flip & selected_is_left, "selected_model"] = result.loc[
        flip & selected_is_left, "model_right"
    ]
    result.loc[flip & ~selected_is_left, "selected_model"] = result.loc[
        flip & ~selected_is_left, "model_left"
    ]
    result["response_flipped"] = flip
    return result
