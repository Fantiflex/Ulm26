"""Pair-generation utilities for pairwise face-comparison experiments."""

from itertools import combinations

import numpy as np
import pandas as pd

from .schemas import require_columns


def _validate_face_ids(face_ids: list[str]) -> list[str]:
    cleaned = [str(face_id).strip() for face_id in face_ids]
    if len(cleaned) < 2:
        raise ValueError("At least two face IDs are required.")
    if any(not face_id for face_id in cleaned):
        raise ValueError("Face IDs cannot be empty.")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("Face IDs must be unique.")
    return cleaned


def generate_all_pairs(face_ids: list[str]) -> pd.DataFrame:
    """Generate every unique unordered pair of face identifiers."""

    cleaned = _validate_face_ids(face_ids)
    rows = [
        {"model_a": left, "model_b": right}
        for left, right in combinations(cleaned, 2)
    ]
    return pd.DataFrame(rows)


def sample_pairs(
    face_ids: list[str],
    n_pairs: int,
    random_state: int,
) -> pd.DataFrame:
    """Sample a reproducible subset of unique unordered pairs."""

    all_pairs = generate_all_pairs(face_ids)
    if n_pairs <= 0:
        raise ValueError("n_pairs must be positive.")
    if n_pairs > len(all_pairs):
        raise ValueError(
            f"Requested {n_pairs} pairs but only {len(all_pairs)} exist."
        )
    return all_pairs.sample(n=n_pairs, random_state=random_state).reset_index(
        drop=True
    )


def generate_balanced_pairs(
    face_ids: list[str],
    n_pairs: int,
    random_state: int,
) -> pd.DataFrame:
    """Generate a unique pair set with approximately balanced exposure."""

    cleaned = _validate_face_ids(face_ids)
    all_pairs = list(combinations(cleaned, 2))
    if n_pairs <= 0:
        raise ValueError("n_pairs must be positive.")
    if n_pairs > len(all_pairs):
        raise ValueError(
            f"Requested {n_pairs} pairs but only {len(all_pairs)} exist."
        )

    rng = np.random.default_rng(random_state)
    rng.shuffle(all_pairs)
    exposure = {face_id: 0 for face_id in cleaned}
    selected: list[tuple[str, str]] = []
    remaining = all_pairs.copy()

    while len(selected) < n_pairs:
        minimum_cost = min(
            exposure[left] + exposure[right] for left, right in remaining
        )
        candidates = [
            pair
            for pair in remaining
            if exposure[pair[0]] + exposure[pair[1]] == minimum_cost
        ]
        chosen = candidates[int(rng.integers(0, len(candidates)))]
        remaining.remove(chosen)
        selected.append(chosen)
        exposure[chosen[0]] += 1
        exposure[chosen[1]] += 1

    return pd.DataFrame(selected, columns=["model_a", "model_b"])


def randomise_pair_sides(
    pairs: pd.DataFrame,
    random_state: int,
) -> pd.DataFrame:
    """Randomly assign each pair member to the left or right position."""

    require_columns(pairs, ["model_a", "model_b"], "pair table")
    rng = np.random.default_rng(random_state)
    swap = rng.random(len(pairs)) < 0.5

    result = pairs.copy().reset_index(drop=True)
    result["model_left"] = np.where(
        swap, result["model_b"], result["model_a"]
    )
    result["model_right"] = np.where(
        swap, result["model_a"], result["model_b"]
    )
    return result.drop(columns=["model_a", "model_b"])


def add_pair_ids(pairs: pd.DataFrame) -> pd.DataFrame:
    """Assign stable sequential identifiers and presentation order."""

    require_columns(pairs, ["model_left", "model_right"], "pair table")
    result = pairs.copy().reset_index(drop=True)
    result["pair_order"] = np.arange(1, len(result) + 1)
    result["pair_id"] = result["pair_order"].map(lambda x: f"PAIR_{x:05d}")
    return result[["pair_id", "model_left", "model_right", "pair_order"]]


def compute_appearance_counts(pairs: pd.DataFrame) -> pd.Series:
    """Count how often each face appears across all comparisons."""

    require_columns(pairs, ["model_left", "model_right"], "pair table")
    return pd.concat(
        [pairs["model_left"], pairs["model_right"]],
        ignore_index=True,
    ).value_counts().sort_index()
