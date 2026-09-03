from face_reconstruction.pairs import generate_all_pairs


def test_generate_all_pairs():
    pairs = generate_all_pairs(
        ["A", "B", "C"]
    )

    assert len(pairs) == 3