from pathlib import Path

from face_reconstruction.stimuli import build_image_index


def test_build_image_index(tmp_path: Path):
    image_path = tmp_path / "CFD-BF-007-001-N.jpg"
    image_path.write_bytes(b"test")

    index = build_image_index(tmp_path)

    assert len(index) == 1
    assert index.loc[0, "image_filename"] == (
        "CFD-BF-007-001-N.jpg"
    )
    assert index.loc[0, "image_model_id"] == "BF-007"
    assert index.loc[0, "image_path"] == image_path.as_posix()