import pandas as pd

from face_reconstruction.bradley_terry import estimate_theta


def test_winning_face_gets_higher_score():
    responses = pd.DataFrame(
        {
            "model_left": ["A", "A", "B"],
            "model_right": ["B", "C", "C"],
            "selected_model": ["A", "A", "B"],
        }
    )

    scores = estimate_theta(
        responses=responses,
        regularization=1.0,
    )

    theta = scores.set_index("face_id")["theta"]

    assert theta["A"] > theta["B"]
    assert theta["B"] > theta["C"]