from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd




def _as_1d_float_array(
    values: Iterable[float] | np.ndarray | pd.Series,
    *,
    name: str,
) -> np.ndarray:
    """
    Convertit une séquence numérique en tableau NumPy 1D de float64.
    """
    array = np.asarray(values, dtype=np.float64)

    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")

    if array.size == 0:
        raise ValueError(f"{name} cannot be empty.")

    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")

    return array


def normalize_weights(
    weights: Iterable[float] | np.ndarray | pd.Series,
) -> np.ndarray:
    """
    Valide et normalise des poids pour que leur somme soit égale à 1.

    Parameters
    ----------
    weights:
        Séquence de poids non négatifs.

    Returns
    -------
    np.ndarray
        Poids normalisés de forme ``(n_items,)``.

    Raises
    ------
    ValueError
        Si les poids sont vides, multidimensionnels, non finis,
        négatifs ou de somme nulle.
    """
    array = _as_1d_float_array(weights, name="weights")

    if np.any(array < 0):
        raise ValueError("weights must be non-negative.")

    total = float(array.sum())

    if total <= 0:
        raise ValueError("weights must have a strictly positive sum.")

    return array / total


def uniform_weights(n_items: int) -> np.ndarray:
    """
    Génère des poids uniformes.

    Parameters
    ----------
    n_items:
        Nombre d'éléments à pondérer.

    Returns
    -------
    np.ndarray
        Tableau contenant ``1 / n_items`` pour chaque élément.
    """
    if isinstance(n_items, bool) or not isinstance(n_items, (int, np.integer)):
        raise TypeError("n_items must be an integer.")

    if n_items <= 0:
        raise ValueError("n_items must be strictly positive.")

    return np.full(
        shape=n_items,
        fill_value=1.0 / n_items,
        dtype=np.float64,
    )


def softmax_weights(
    scores: Iterable[float] | np.ndarray | pd.Series,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Transforme des scores en poids softmax.

    La formule utilisée est :

        w_i = exp(score_i / temperature)
              / sum_j exp(score_j / temperature)

    Une faible température concentre davantage les poids sur les scores
    les plus élevés. Une température élevée rapproche les poids de
    l'uniforme.

    Parameters
    ----------
    scores:
        Scores latents, par exemple les coefficients Bradley-Terry.
    temperature:
        Température strictement positive.

    Returns
    -------
    np.ndarray
        Poids softmax normalisés.
    """
    score_array = _as_1d_float_array(scores, name="scores")

    if not np.isscalar(temperature):
        raise TypeError("temperature must be a scalar.")

    temperature = float(temperature)

    if not np.isfinite(temperature):
        raise ValueError("temperature must be finite.")

    if temperature <= 0:
        raise ValueError("temperature must be strictly positive.")

    scaled_scores = score_array / temperature

    # Stabilisation numérique :
    # soustraire le maximum ne change pas le résultat du softmax.
    scaled_scores = scaled_scores - np.max(scaled_scores)

    exponentials = np.exp(scaled_scores)

    return normalize_weights(exponentials)


def select_top_k(
    item_ids: Iterable[str] | np.ndarray | pd.Series,
    weights: Iterable[float] | np.ndarray | pd.Series,
    top_k: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sélectionne les ``top_k`` éléments ayant les poids les plus élevés.

    Les poids sélectionnés sont renormalisés pour sommer à 1.

    Parameters
    ----------
    item_ids:
        Identifiants associés aux poids.
    weights:
        Poids des éléments.
    top_k:
        Nombre d'éléments à conserver. Si ``None`` ou supérieur au nombre
        d'éléments, tous les éléments sont conservés.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Identifiants sélectionnés et poids renormalisés, triés du poids
        le plus élevé au poids le plus faible.
    """
    ids = np.asarray(item_ids)

    if ids.ndim != 1:
        raise ValueError("item_ids must be one-dimensional.")

    if ids.size == 0:
        raise ValueError("item_ids cannot be empty.")

    normalized = normalize_weights(weights)

    if ids.size != normalized.size:
        raise ValueError(
            "item_ids and weights must contain the same number of elements."
        )

    if top_k is None:
        selected_indices = np.argsort(
            -normalized,
            kind="stable",
        )
    else:
        if isinstance(top_k, bool) or not isinstance(
            top_k,
            (int, np.integer),
        ):
            raise TypeError("top_k must be an integer or None.")

        if top_k <= 0:
            raise ValueError("top_k must be strictly positive.")

        selected_indices = np.argsort(
            -normalized,
            kind="stable",
        )[:top_k]

    selected_ids = ids[selected_indices]
    selected_weights = normalize_weights(
        normalized[selected_indices]
    )

    return selected_ids, selected_weights


def effective_sample_size(
    weights: Iterable[float] | np.ndarray | pd.Series,
) -> float:
    """
    Calcule le nombre effectif d'éléments contribuant à la reconstruction.

    La formule est :

        N_eff = 1 / sum_i(w_i ** 2)

    Pour des poids uniformes sur n éléments, ``N_eff = n``.
    Si un seul élément porte tout le poids, ``N_eff = 1``.
    """
    normalized = normalize_weights(weights)

    return float(1.0 / np.sum(normalized**2))


def weight_entropy(
    weights: Iterable[float] | np.ndarray | pd.Series,
    *,
    normalized: bool = False,
) -> float:
    """
    Calcule l'entropie de Shannon des poids.

    Parameters
    ----------
    weights:
        Poids non négatifs.
    normalized:
        Si ``True``, divise l'entropie par ``log(n)`` afin d'obtenir
        une valeur entre 0 et 1 lorsque plusieurs éléments sont présents.

    Returns
    -------
    float
        Entropie brute ou normalisée.
    """
    probability_weights = normalize_weights(weights)

    positive_weights = probability_weights[
        probability_weights > 0
    ]

    entropy = float(
        -np.sum(
            positive_weights * np.log(positive_weights)
        )
    )

    if not normalized:
        return entropy

    n_items = probability_weights.size

    if n_items == 1:
        return 0.0

    return float(entropy / np.log(n_items))


def build_weight_table(
    face_ids: Iterable[str] | np.ndarray | pd.Series,
    scores: Iterable[float] | np.ndarray | pd.Series,
    *,
    method: str = "softmax",
    temperature: float = 1.0,
    top_k: int | None = None,
) -> pd.DataFrame:
    """
    Construit un tableau de diagnostic des scores et des poids.

    Cette fonction est facultative, mais pratique pour relier les scores
    Bradley-Terry au pipeline de morphing.

    Returns
    -------
    pd.DataFrame
        Colonnes :
        - face_id
        - theta
        - rank
        - weight
        - selected
    """
    ids = np.asarray(face_ids)
    theta = _as_1d_float_array(scores, name="scores")

    if ids.ndim != 1:
        raise ValueError("face_ids must be one-dimensional.")

    if ids.size != theta.size:
        raise ValueError(
            "face_ids and scores must have the same length."
        )

    if method == "uniform":
        weights = uniform_weights(len(ids))
    elif method == "softmax":
        weights = softmax_weights(
            theta,
            temperature=temperature,
        )
    else:
        raise ValueError(
            "method must be either 'uniform' or 'softmax'."
        )

    order = np.argsort(-weights, kind="stable")

    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)

    table = pd.DataFrame(
        {
            "face_id": ids,
            "theta": theta,
            "rank": ranks,
            "weight_before_top_k": weights,
        }
    )

    selected_ids, selected_weights = select_top_k(
        item_ids=ids,
        weights=weights,
        top_k=top_k,
    )

    selected_weight_map = dict(
        zip(
            selected_ids.tolist(),
            selected_weights.tolist(),
            strict=True,
        )
    )

    table["selected"] = table["face_id"].isin(
        selected_weight_map
    )
    table["weight"] = (
        table["face_id"]
        .map(selected_weight_map)
        .fillna(0.0)
        .astype(float)
    )

    return table.sort_values(
        by="rank",
        ascending=True,
    ).reset_index(drop=True)