"""Diagnostics for comparison designs and Bradley-Terry estimates."""


from collections import defaultdict, deque

import numpy as np
import pandas as pd

from .schemas import require_columns


def compute_appearance_counts(comparisons: pd.DataFrame) -> pd.Series:
    """Count how many times each face appeared in comparisons."""

    require_columns(
        comparisons,
        ["model_left", "model_right"],
        "comparison table",
    )
    return pd.concat(
        [comparisons["model_left"], comparisons["model_right"]],
        ignore_index=True,
    ).value_counts().sort_index()


def compute_win_counts(responses: pd.DataFrame) -> pd.Series:
    """Count how many times each face was selected."""

    require_columns(responses, ["selected_model"], "response table")
    return responses["selected_model"].value_counts().sort_index()


def find_disconnected_components(
    comparisons: pd.DataFrame,
) -> list[set[str]]:
    """Return connected components of the undirected comparison graph."""

    require_columns(
        comparisons,
        ["model_left", "model_right"],
        "comparison table",
    )
    adjacency: dict[str, set[str]] = defaultdict(set)
    for row in comparisons.itertuples(index=False):
        adjacency[row.model_left].add(row.model_right)
        adjacency[row.model_right].add(row.model_left)

    components: list[set[str]] = []
    unvisited = set(adjacency)
    while unvisited:
        start = next(iter(unvisited))
        queue: deque[str] = deque([start])
        component: set[str] = set()
        while queue:
            node = queue.popleft()
            if node in component:
                continue
            component.add(node)
            queue.extend(adjacency[node] - component)
        components.append(component)
        unvisited -= component

    return sorted(components, key=len, reverse=True)


def check_comparison_graph_connectivity(comparisons: pd.DataFrame) -> bool:
    """Return True when all compared faces form one connected graph."""

    return len(find_disconnected_components(comparisons)) == 1


def compute_exposure_imbalance(
    comparisons: pd.DataFrame,
) -> dict[str, float]:
    """Summarise imbalance in face exposure counts."""

    counts = compute_appearance_counts(comparisons).astype(float)
    mean = float(counts.mean())
    return {
        "minimum": float(counts.min()),
        "maximum": float(counts.max()),
        "mean": mean,
        "standard_deviation": float(counts.std(ddof=0)),
        "range": float(counts.max() - counts.min()),
        "coefficient_of_variation": (
            float(counts.std(ddof=0) / mean) if mean else np.nan
        ),
    }


def build_diagnostic_table(
    stimuli: pd.DataFrame,
    responses: pd.DataFrame,
    scores: pd.DataFrame,
) -> pd.DataFrame:
    """Combine metadata, counts, model estimates, and reconstruction weights."""

    require_columns(stimuli, ["face_id", "image_path"], "stimuli table")
    require_columns(
        responses,
        ["model_left", "model_right", "selected_model"],
        "response table",
    )
    require_columns(scores, ["face_id", "theta", "rank"], "score table")

    appearances = compute_appearance_counts(responses).rename("n_appearances")
    wins = compute_win_counts(responses).rename("n_wins")

    metadata_columns = ["face_id", "image_path"]
    if "landmark_file_exists" in stimuli.columns:
        metadata_columns.append("landmark_file_exists")

    result = stimuli[metadata_columns].drop_duplicates("face_id")
    result = result.merge(scores, on="face_id", how="left", validate="one_to_one")
    result = result.merge(
        appearances,
        left_on="face_id",
        right_index=True,
        how="left",
    )
    result = result.merge(
        wins,
        left_on="face_id",
        right_index=True,
        how="left",
    )
    result[["n_appearances", "n_wins"]] = result[
        ["n_appearances", "n_wins"]
    ].fillna(0).astype(int)

    preferred_order = [
        "face_id",
        "n_appearances",
        "n_wins",
        "theta",
        "standard_error",
        "rank",
        "weight",
        "image_path",
        "landmark_file_exists",
    ]
    return result[[column for column in preferred_order if column in result.columns]]


def evaluate_score_recovery(
    true_scores: pd.DataFrame,
    estimated_scores: pd.DataFrame,
    top_k: int = 5,
) -> dict[str, float]:
    """Compare estimated scores with known simulated ground truth."""

    require_columns(true_scores, ["face_id", "theta_true"], "true scores")
    require_columns(estimated_scores, ["face_id", "theta"], "estimated scores")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    merged = true_scores.merge(
        estimated_scores[["face_id", "theta"]],
        on="face_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) < 2:
        raise ValueError("At least two overlapping faces are required.")

    pearson = merged["theta_true"].corr(merged["theta"], method="pearson")
    spearman = merged["theta_true"].corr(merged["theta"], method="spearman")
    mse = float(np.mean((merged["theta_true"] - merged["theta"]) ** 2))

    k = min(top_k, len(merged))
    true_top = set(merged.nlargest(k, "theta_true")["face_id"])
    estimated_top = set(merged.nlargest(k, "theta")["face_id"])

    return {
        "pearson_correlation": float(pearson),
        "spearman_correlation": float(spearman),
        "mean_squared_error": mse,
        "top_k_recall": len(true_top & estimated_top) / k,
    }
