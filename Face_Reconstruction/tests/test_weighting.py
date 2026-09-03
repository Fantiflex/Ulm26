import numpy as np
import pytest

from face_reconstruction.weighting import (
    build_weight_table,
    effective_sample_size,
    normalize_weights,
    select_top_k,
    softmax_weights,
    uniform_weights,
    weight_entropy,
)


def test_normalize_weights_sums_to_one():
    weights = normalize_weights([2.0, 3.0, 5.0])

    assert np.isclose(weights.sum(), 1.0)
    assert np.allclose(weights, [0.2, 0.3, 0.5])


def test_normalize_weights_rejects_negative_values():
    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        normalize_weights([0.5, -0.2, 0.7])


def test_normalize_weights_rejects_zero_sum():
    with pytest.raises(
        ValueError,
        match="strictly positive sum",
    ):
        normalize_weights([0.0, 0.0])


def test_uniform_weights():
    weights = uniform_weights(4)

    assert np.allclose(weights, [0.25] * 4)
    assert np.isclose(weights.sum(), 1.0)


def test_uniform_weights_rejects_invalid_size():
    with pytest.raises(ValueError):
        uniform_weights(0)


def test_softmax_weights_sum_to_one():
    scores = np.array([-1.0, 0.0, 1.0])

    weights = softmax_weights(
        scores,
        temperature=1.0,
    )

    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= 0)
    assert weights[2] > weights[1] > weights[0]


def test_softmax_is_invariant_to_constant_shift():
    scores = np.array([-1.0, 0.0, 1.0])

    weights_1 = softmax_weights(scores)
    weights_2 = softmax_weights(scores + 1000)

    assert np.allclose(weights_1, weights_2)


def test_low_temperature_concentrates_weights():
    scores = np.array([0.0, 1.0, 2.0])

    low_temperature = softmax_weights(
        scores,
        temperature=0.1,
    )
    high_temperature = softmax_weights(
        scores,
        temperature=10.0,
    )

    assert low_temperature.max() > high_temperature.max()


def test_high_temperature_approaches_uniform():
    scores = np.array([-2.0, 0.0, 2.0])

    weights = softmax_weights(
        scores,
        temperature=1_000_000.0,
    )

    assert np.allclose(
        weights,
        np.full(3, 1 / 3),
        atol=1e-6,
    )


def test_softmax_rejects_non_positive_temperature():
    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        softmax_weights([1.0, 2.0], temperature=0.0)


def test_select_top_k_returns_highest_weights():
    ids = np.array(["A", "B", "C", "D"])
    weights = np.array([0.1, 0.5, 0.3, 0.1])

    selected_ids, selected_weights = select_top_k(
        ids,
        weights,
        top_k=2,
    )

    assert selected_ids.tolist() == ["B", "C"]
    assert np.isclose(selected_weights.sum(), 1.0)
    assert np.allclose(
        selected_weights,
        [0.625, 0.375],
    )


def test_select_top_k_none_keeps_all_items():
    ids = np.array(["A", "B", "C"])
    weights = np.array([0.2, 0.5, 0.3])

    selected_ids, selected_weights = select_top_k(
        ids,
        weights,
        top_k=None,
    )

    assert selected_ids.tolist() == ["B", "C", "A"]
    assert np.allclose(
        selected_weights,
        [0.5, 0.3, 0.2],
    )


def test_select_top_k_rejects_mismatched_lengths():
    with pytest.raises(
        ValueError,
        match="same number",
    ):
        select_top_k(
            ["A", "B"],
            [0.2, 0.3, 0.5],
            top_k=2,
        )


def test_effective_sample_size_uniform():
    weights = uniform_weights(5)

    assert np.isclose(
        effective_sample_size(weights),
        5.0,
    )


def test_effective_sample_size_one_dominant_item():
    weights = np.array([1.0, 0.0, 0.0])

    assert np.isclose(
        effective_sample_size(weights),
        1.0,
    )


def test_weight_entropy_uniform_is_log_n():
    weights = uniform_weights(4)

    assert np.isclose(
        weight_entropy(weights),
        np.log(4),
    )


def test_normalized_entropy_uniform_is_one():
    weights = uniform_weights(4)

    assert np.isclose(
        weight_entropy(
            weights,
            normalized=True,
        ),
        1.0,
    )


def test_normalized_entropy_one_hot_is_zero():
    weights = np.array([1.0, 0.0, 0.0])

    assert np.isclose(
        weight_entropy(
            weights,
            normalized=True,
        ),
        0.0,
    )


def test_build_weight_table():
    table = build_weight_table(
        face_ids=["A", "B", "C"],
        scores=[0.0, 2.0, 1.0],
        method="softmax",
        temperature=1.0,
        top_k=2,
    )

    assert table["face_id"].tolist() == ["B", "C", "A"]
    assert table["rank"].tolist() == [1, 2, 3]
    assert table["selected"].tolist() == [
        True,
        True,
        False,
    ]
    assert np.isclose(table["weight"].sum(), 1.0)
    assert table.loc[
        table["face_id"] == "A",
        "weight",
    ].iloc[0] == 0.0