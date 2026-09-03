"""High-level orchestration for the scientific pipeline."""

from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

from .bradley_terry import (
    add_softmax_weights,
    estimate_standard_errors,
    estimate_theta,
)
from .diagnostics import build_diagnostic_table
from .pairs import (
    add_pair_ids,
    generate_balanced_pairs,
    randomise_pair_sides,
)
from .stimuli import build_stimuli_table
from .validation import validate_pairs, validate_responses, validate_stimuli


def load_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("The configuration file must contain a mapping.")
    return config


def prepare_stimuli(
    manifest: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Build, validate, and optionally save the study stimulus table."""

    selection = config.get("stimuli", {})
    stimuli = build_stimuli_table(
        manifest=manifest,
        image_directory=Path(selection["image_directory"]),
        genders=selection.get("genders"),
        ethnicities_self=selection.get("ethnicities_self"),
        ethnicities_perceived=selection.get("ethnicities_perceived"),
        minimum_perceived_probability=selection.get(
            "minimum_perceived_probability"
        ),
        ethnicity_selection_mode=selection.get(
            "ethnicity_selection_mode", "self"
        ),
    )
    validate_stimuli(stimuli)

    output_path = selection.get("output_path")
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        stimuli.to_csv(path, index=False)
    return stimuli


def prepare_pairs(
    stimuli: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Generate, randomise, validate, and optionally save face pairs."""

    settings = config.get("pairs", {})
    face_ids = stimuli["face_id"].tolist()
    pairs = generate_balanced_pairs(
        face_ids=face_ids,
        n_pairs=int(settings["n_pairs"]),
        random_state=int(settings.get("random_state", 0)),
    )
    pairs = randomise_pair_sides(
        pairs,
        random_state=int(settings.get("side_random_state", 1)),
    )
    pairs = add_pair_ids(pairs)
    validate_pairs(pairs, set(face_ids))

    output_path = settings.get("output_path")
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pairs.to_csv(path, index=False)
    return pairs


def analyse_participant(
    responses: pd.DataFrame,
    stimuli: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit one participant model and build its diagnostic table."""

    validate_responses(responses)
    settings = config.get("model", {})
    participant_ids = responses["participant_id"].dropna().unique()
    if len(participant_ids) != 1:
        raise ValueError("analyse_participant expects exactly one participant.")
    participant_id = str(participant_ids[0])

    regularization = float(settings.get("regularization", 1.0))
    scores = estimate_theta(
        responses,
        regularization=regularization,
        participant_id=participant_id,
    )
    scores = estimate_standard_errors(
        responses,
        scores,
        regularization=regularization,
    )
    scores = add_softmax_weights(
        scores,
        temperature=float(settings.get("temperature", 1.0)),
    )
    diagnostics = build_diagnostic_table(stimuli, responses, scores)
    return scores, diagnostics


def run_pipeline(
    config_path: Path,
    manifest_loader: Callable[[dict[str, Any]], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run stimulus and pair preparation from a configuration file.

    ``manifest_loader`` is injected so this module remains independent of a
    specific database loader. For CFD, pass a function that reads and
    harmonises the official workbook.
    """

    config = load_config(config_path)
    manifest = manifest_loader(config)
    stimuli = prepare_stimuli(manifest, config)
    pairs = prepare_pairs(stimuli, config)
    return stimuli, pairs
