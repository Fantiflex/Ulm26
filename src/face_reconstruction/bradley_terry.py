"""Bradley-Terry estimation for pairwise face-choice data."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from .schemas import require_columns


def build_design_matrix(
    responses: pd.DataFrame,
    face_ids: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Convert pairwise responses into a logistic-regression design matrix."""

    require_columns(
        responses,
        ["model_left", "model_right", "selected_model"],
        "response table",
    )
    if face_ids is None:
        face_ids = sorted(
            set(responses["model_left"]) | set(responses["model_right"])
        )
    if len(set(face_ids)) != len(face_ids):
        raise ValueError("face_ids must be unique.")

    index = {face_id: position for position, face_id in enumerate(face_ids)}
    design = np.zeros((len(responses), len(face_ids)), dtype=float)
    outcomes = np.zeros(len(responses), dtype=float)

    for row_index, row in enumerate(responses.itertuples(index=False)):
        if row.model_left not in index or row.model_right not in index:
            raise ValueError("A response contains an unknown face ID.")
        if row.selected_model not in {row.model_left, row.model_right}:
            raise ValueError("selected_model must be the left or right face.")

        design[row_index, index[row.model_left]] = 1.0
        design[row_index, index[row.model_right]] = -1.0
        outcomes[row_index] = float(row.selected_model == row.model_left)

    return design, outcomes, face_ids


def negative_log_likelihood(
    theta: np.ndarray,
    design_matrix: np.ndarray,
    outcomes: np.ndarray,
    regularization: float,
) -> float:
    """Compute the L2-regularised Bradley-Terry negative log-likelihood."""

    logits = design_matrix @ theta
    log_likelihood = np.sum(
        outcomes * np.logaddexp(0.0, -logits)
        + (1.0 - outcomes) * np.logaddexp(0.0, logits)
    )
    penalty = 0.5 * regularization * float(theta @ theta)
    return float(log_likelihood + penalty)


def _gradient(
    theta: np.ndarray,
    design_matrix: np.ndarray,
    outcomes: np.ndarray,
    regularization: float,
) -> np.ndarray:
    probabilities = expit(design_matrix @ theta)
    return design_matrix.T @ (probabilities - outcomes) + regularization * theta


def centre_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Centre Bradley-Terry scores so that their mean is zero."""

    require_columns(scores, ["theta"], "score table")
    result = scores.copy()
    result["theta"] = result["theta"] - result["theta"].mean()
    return result


def add_ranks(scores: pd.DataFrame) -> pd.DataFrame:
    """Rank faces from highest to lowest estimated preference."""

    require_columns(scores, ["theta"], "score table")
    result = scores.copy()
    result["rank"] = (
        result["theta"].rank(method="min", ascending=False).astype(int)
    )
    return result


def add_softmax_weights(
    scores: pd.DataFrame,
    temperature: float = 1.0,
) -> pd.DataFrame:
    """Convert latent scores into positive weights that sum to one."""

    require_columns(scores, ["theta"], "score table")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    result = scores.copy()
    scaled = result["theta"].to_numpy(dtype=float) / temperature
    scaled -= scaled.max()
    exponentials = np.exp(scaled)
    result["weight"] = exponentials / exponentials.sum()
    return result


def estimate_theta(
    responses: pd.DataFrame,
    regularization: float = 1.0,
    participant_id: str | None = None,
    max_iterations: int = 1_000,
) -> pd.DataFrame:
    """Estimate one Bradley-Terry score per face.

    Regularisation makes the optimisation identifiable and more stable for
    sparse designs. Scores are additionally centred for interpretability.
    """

    if regularization < 0:
        raise ValueError("regularization cannot be negative.")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")

    design, outcomes, face_ids = build_design_matrix(responses)
    initial = np.zeros(len(face_ids), dtype=float)
    result = minimize(
        negative_log_likelihood,
        initial,
        args=(design, outcomes, regularization),
        jac=_gradient,
        method="L-BFGS-B",
        options={"maxiter": max_iterations},
    )
    if not result.success:
        raise RuntimeError(f"Bradley-Terry optimisation failed: {result.message}")

    scores = pd.DataFrame(
        {
            "face_id": face_ids,
            "theta": result.x,
        }
    )
    scores = add_ranks(centre_scores(scores))
    if participant_id is not None:
        scores.insert(0, "participant_id", participant_id)
    scores.attrs["converged"] = bool(result.success)
    scores.attrs["objective"] = float(result.fun)
    scores.attrs["n_iterations"] = int(result.nit)
    return scores


def estimate_standard_errors(
    responses: pd.DataFrame,
    scores: pd.DataFrame,
    regularization: float = 1.0,
) -> pd.DataFrame:
    """Approximate score uncertainty from the observed Hessian."""

    require_columns(scores, ["face_id", "theta"], "score table")
    design, _, face_ids = build_design_matrix(
        responses,
        face_ids=scores["face_id"].tolist(),
    )
    theta = scores.set_index("face_id").loc[face_ids, "theta"].to_numpy()
    probabilities = expit(design @ theta)
    weights = probabilities * (1.0 - probabilities)
    hessian = design.T @ (design * weights[:, None])
    hessian += regularization * np.eye(len(face_ids))
    covariance = np.linalg.pinv(hessian)

    result = scores.copy()
    result["standard_error"] = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    return result
