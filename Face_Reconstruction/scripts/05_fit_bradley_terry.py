"""
Fit participant-level Bradley-Terry models.

This script loads pairwise participant responses, estimates latent face
preferences for each participant, computes ranks and softmax reconstruction
weights, builds diagnostic tables, and saves all model outputs.
"""

from pathlib import Path

import pandas as pd

from face_reconstruction.bradley_terry import (
    add_softmax_weights,
    estimate_standard_errors,
    estimate_theta,
)
from face_reconstruction.diagnostics import (
    build_diagnostic_table,
    evaluate_score_recovery,
)
from face_reconstruction.validation import (
    validate_responses,
)


RESPONSES_PATH = Path(
    "data/simulated/simulated_responses.csv"
)

STIMULI_PATH = Path(
    "data/processed/stimuli.csv"
)

TRUE_SCORES_PATH = Path(
    "data/simulated/true_latent_scores.csv"
)

OUTPUT_DIRECTORY = Path(
    "data/results/bradley_terry"
)

REGULARIZATION = 1.0

SOFTMAX_TEMPERATURE = 1.0


def load_csv(
    path: Path,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Load a required CSV dataset.

    Parameters
    ----------
    path:
        Path to the CSV file.

    dataset_name:
        Human-readable dataset name used in error messages.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"{dataset_name} not found: {path}"
        )

    return pd.read_csv(
        path
    )


def fit_participant_model(
    participant_responses: pd.DataFrame,
    regularization: float,
    softmax_temperature: float,
) -> pd.DataFrame:
    """
    Fit one Bradley-Terry model for a single participant.

    Parameters
    ----------
    participant_responses:
        Pairwise responses from one participant.

    regularization:
        L2 penalty applied to latent face scores.

    softmax_temperature:
        Temperature used to convert scores into reconstruction weights.

    Returns
    -------
    pd.DataFrame
        Estimated scores, ranks, standard errors, and softmax weights.
    """

    participant_ids = (
        participant_responses[
            "participant_id"
        ]
        .dropna()
        .unique()
    )

    if len(participant_ids) != 1:
        raise ValueError(
            "fit_participant_model expects "
            "responses from exactly one participant."
        )

    participant_id = str(
        participant_ids[0]
    )

    scores = estimate_theta(
        responses=participant_responses,
        regularization=regularization,
        participant_id=participant_id,
    )

    scores = estimate_standard_errors(
        responses=participant_responses,
        scores=scores,
        regularization=regularization,
    )

    scores = add_softmax_weights(
        scores=scores,
        temperature=softmax_temperature,
    )

    return scores


def fit_all_participants(
    responses: pd.DataFrame,
    regularization: float,
    softmax_temperature: float,
) -> pd.DataFrame:
    """
    Fit participant-specific Bradley-Terry models for an entire dataset.

    Parameters
    ----------
    responses:
        Response table containing one or more participants.

    regularization:
        L2 penalty applied to each participant model.

    softmax_temperature:
        Softmax temperature used to calculate reconstruction weights.

    Returns
    -------
    pd.DataFrame
        Combined participant-level score table.
    """

    score_tables = []

    grouped_responses = responses.groupby(
        "participant_id",
        sort=True,
    )

    for participant_id, participant_data in grouped_responses:
        print(
            f"Fitting participant: "
            f"{participant_id}"
        )

        participant_scores = (
            fit_participant_model(
                participant_responses=(
                    participant_data.copy()
                ),
                regularization=regularization,
                softmax_temperature=(
                    softmax_temperature
                ),
            )
        )

        score_tables.append(
            participant_scores
        )

    if not score_tables:
        raise ValueError(
            "No participant models were fitted."
        )

    return pd.concat(
        score_tables,
        ignore_index=True,
    )


def build_all_diagnostics(
    stimuli: pd.DataFrame,
    responses: pd.DataFrame,
    scores: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build face-level diagnostics separately for each participant.

    Parameters
    ----------
    stimuli:
        Study-specific stimulus metadata.

    responses:
        Complete participant response table.

    scores:
        Combined participant-level Bradley-Terry estimates.

    Returns
    -------
    pd.DataFrame
        Combined participant diagnostic table.
    """

    diagnostic_tables = []

    for participant_id, participant_scores in (
        scores.groupby(
            "participant_id",
            sort=True,
        )
    ):
        participant_responses = responses.loc[
            responses["participant_id"]
            == participant_id
        ].copy()

        diagnostics = (
            build_diagnostic_table(
                stimuli=stimuli,
                responses=participant_responses,
                scores=participant_scores,
            )
        )

        diagnostics[
            "participant_id"
        ] = participant_id

        diagnostic_tables.append(
            diagnostics
        )

    return pd.concat(
        diagnostic_tables,
        ignore_index=True,
    )


def build_recovery_report(
    scores: pd.DataFrame,
    true_scores: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate recovery of known latent scores for each simulated participant.

    Parameters
    ----------
    scores:
        Estimated participant-level Bradley-Terry scores.

    true_scores:
        Ground-truth latent scores used during simulation.

    Returns
    -------
    pd.DataFrame
        One row of recovery metrics per participant.
    """

    reports = []

    for participant_id, participant_scores in (
        scores.groupby(
            "participant_id",
            sort=True,
        )
    ):
        metrics = evaluate_score_recovery(
            true_scores=true_scores,
            estimated_scores=participant_scores,
        )

        reports.append(
            {
                "participant_id": (
                    participant_id
                ),
                **metrics,
            }
        )

    return pd.DataFrame(
        reports
    )


def save_model_outputs(
    scores: pd.DataFrame,
    diagnostics: pd.DataFrame,
    recovery_report: pd.DataFrame | None,
    output_directory: Path,
) -> None:
    """
    Save Bradley-Terry estimates and diagnostic outputs.

    Parameters
    ----------
    scores:
        Participant-level score estimates.

    diagnostics:
        Face-level diagnostic table.

    recovery_report:
        Optional score-recovery report for simulated data.

    output_directory:
        Directory in which outputs are saved.
    """

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    scores.to_csv(
        output_directory
        / "scores.csv",
        index=False,
    )

    diagnostics.to_csv(
        output_directory
        / "diagnostics.csv",
        index=False,
    )

    if recovery_report is not None:
        recovery_report.to_csv(
            output_directory
            / "score_recovery.csv",
            index=False,
        )


def print_model_summary(
    scores: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> None:
    """
    Print the main Bradley-Terry fitting results.

    Parameters
    ----------
    scores:
        Combined participant-level score estimates.

    diagnostics:
        Combined diagnostic table.
    """

    print("\nBradley-Terry estimation")
    print("------------------------")
    print(
        "Participants fitted: "
        f"{scores['participant_id'].nunique()}"
    )

    print(
        "Total score rows: "
        f"{len(scores)}"
    )

    print(
        "Total diagnostic rows: "
        f"{len(diagnostics)}"
    )

    print("\nTheta summary:")
    print(
        scores["theta"].describe()
    )

    print("\nWeight-sum check:")
    print(
        scores.groupby(
            "participant_id"
        )["weight"]
        .sum()
        .describe()
    )

def summarise_recovery(
    recovery_report: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarise Bradley-Terry recovery metrics across simulated participants.
    """

    metric_columns = [
        "pearson_correlation",
        "spearman_correlation",
        "mean_squared_error",
        "top_k_recall",
    ]

    summary = (
        recovery_report[metric_columns]
        .agg(
            [
                "mean",
                "std",
                "min",
                "median",
                "max",
            ]
        )
        .transpose()
        .reset_index()
        .rename(columns={"index": "metric"})
    )

    return summary

def print_recovery_summary(
    recovery_report: pd.DataFrame,
) -> None:
    """
    Print score-recovery metrics across simulated participants.

    Parameters
    ----------
    recovery_report:
        Participant-level comparison between true and estimated scores.
    """

    metric_columns = [
        "pearson_correlation",
        "spearman_correlation",
        "mean_squared_error",
        "top_k_recall",
    ]

    print("\nScore recovery")
    print("--------------")

    print(
        recovery_report[metric_columns]
        .agg(
            [
                "mean",
                "std",
                "min",
                "median",
                "max",
            ]
        )
        .transpose()
        .round(3)
    )


def main() -> None:
    """
    Fit all participant models and save estimates and diagnostics.
    """

    responses = load_csv(
        path=RESPONSES_PATH,
        dataset_name="Response table",
    )

    stimuli = load_csv(
        path=STIMULI_PATH,
        dataset_name="Stimulus table",
    )

    validate_responses(
        responses
    )

    scores = fit_all_participants(
        responses=responses,
        regularization=REGULARIZATION,
        softmax_temperature=(
            SOFTMAX_TEMPERATURE
        ),
    )

    diagnostics = (
        build_all_diagnostics(
            stimuli=stimuli,
            responses=responses,
            scores=scores,
        )
    )

    recovery_report = None

    if TRUE_SCORES_PATH.exists():
        true_scores = load_csv(
            path=TRUE_SCORES_PATH,
            dataset_name=(
                "Ground-truth score table"
            ),
        )

        recovery_report = (
            build_recovery_report(
                scores=scores,
                true_scores=true_scores,
            )
        )
    if recovery_report is not None:
        print_recovery_summary(
            recovery_report
        )
    save_model_outputs(
        scores=scores,
        diagnostics=diagnostics,
        recovery_report=recovery_report,
        output_directory=OUTPUT_DIRECTORY,
    )

    print_model_summary(
        scores=scores,
        diagnostics=diagnostics,
    )

    print(
        "\nModel outputs saved to: "
        f"{OUTPUT_DIRECTORY}"
    )
    
    recovery_summary = summarise_recovery(
    recovery_report,
    )

    recovery_summary.to_csv(
        OUTPUT_DIRECTORY / "score_recovery_summary.csv",
        index=False,
    )



if __name__ == "__main__":
    main()