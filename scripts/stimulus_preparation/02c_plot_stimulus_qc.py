"""Generate aggregate QC tables and figures after pipeline steps 01 and 02."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from minority_estimation.config import MINIMUM_PERCEIVED_PROBABILITY


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = PROJECT_ROOT / "results" / "stimulus_qc" / "audit_and_selection"

MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "interim" / "cfd_audit"
    / "cfd_manifest_harmonised.csv"
)
STIMULI_PATH = PROJECT_ROOT / "data" / "processed" / "stimuli.csv"

GROUPS = ["Black", "White"]
COLORS = {"Black": "#0072B2", "White": "#D55E00"}

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlelocation": "left",
    "savefig.dpi": 300,
})


def save_figure(fig, name):
    """Save a readable PNG and a vector PDF."""
    fig.savefig(OUTPUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUTPUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def save_table(table, name, index=False):
    table.to_csv(OUTPUT / f"{name}.csv", index=index)


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    selected = pd.read_csv(STIMULI_PATH)

    for name, frame in [("manifest", manifest), ("stimuli", selected)]:
        if frame.empty:
            raise ValueError(f"{name} is empty.")
        if frame["face_id"].isna().any() or frame["face_id"].duplicated().any():
            raise ValueError(f"{name} has missing or duplicate face IDs.")

    if not selected["face_id"].isin(manifest["face_id"]).all():
        raise ValueError("Some selected IDs are absent from the manifest.")

    OUTPUT.mkdir(parents=True, exist_ok=True)

    # 1. Composition of the original database.
    demographics = pd.crosstab(
        manifest["ethnicity_self"].fillna("Missing"),
        manifest["gender_self"].fillna("Missing"),
    ).sort_index()

    save_table(demographics, "01_demographic_counts", index=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    demographics.plot.bar(ax=ax, color=["#0072B2", "#D55E00", "#009E73"])
    ax.set(
        title="CFD composition by self-reported category and gender",
        xlabel="Self-reported ethnicity category",
        ylabel="Number of models",
    )
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="Self-reported gender")
    fig.tight_layout()
    save_figure(fig, "01_demographic_counts")

    # 2. Self-reported versus dominant perceived category.
    cross = pd.crosstab(
        manifest["ethnicity_self"].fillna("Missing"),
        manifest["ethnicity_perceived"].fillna("Missing"),
    ).sort_index()

    percentages = cross.div(cross.sum(axis=1), axis=0) * 100
    save_table(cross, "02_self_perceived_counts", index=True)
    save_table(percentages, "02_self_perceived_row_percentages", index=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    image = ax.imshow(percentages.to_numpy(), cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(cross.columns)), labels=cross.columns)
    ax.set_yticks(range(len(cross.index)), labels=cross.index)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")

    for row in range(len(cross.index)):
        for col in range(len(cross.columns)):
            value = percentages.iloc[row, col]
            ax.text(
                col, row,
                f"{value:.1f}%\n(n={cross.iloc[row, col]})",
                ha="center", va="center", fontsize=9,
                color="white" if value > 55 else "black",
            )

    ax.set(
        title="Self-reported versus dominant perceived category",
        xlabel="Dominant perceived category",
        ylabel="Self-reported category",
    )
    fig.colorbar(image, ax=ax, label="Percentage within self-reported category")
    fig.tight_layout()
    save_figure(fig, "02_self_perceived_categories")

    # 3. Missing metadata, including fully observed columns in the table.
    missing = pd.DataFrame({
        "column": manifest.columns,
        "n_missing": manifest.isna().sum().to_numpy(),
        "missing_percentage": manifest.isna().mean().to_numpy() * 100,
    }).sort_values("missing_percentage", ascending=False)

    save_table(missing, "03_missingness")

    nonzero = missing.loc[missing["n_missing"] > 0].sort_values(
        "missing_percentage"
    )
    fig, ax = plt.subplots(figsize=(9, max(4, len(nonzero) * 0.35)))
    if nonzero.empty:
        ax.text(
            0.5, 0.5, "No missing values in the harmonised manifest",
            transform=ax.transAxes, ha="center", va="center",
        )
        ax.set_axis_off()
    else:
        ax.barh(nonzero["column"], nonzero["missing_percentage"], color="#0072B2")
        ax.set(xlabel="Missing values (%)", xlim=(0, 100))
    ax.set_title("Missing metadata in the harmonised CFD manifest")
    fig.tight_layout()
    save_figure(fig, "03_missingness")

    # 4. Category counts before and after selection.
    counts = pd.concat([
        manifest["ethnicity_perceived"].fillna("Missing").value_counts().rename(
            "original_manifest"
        ),
        selected["ethnicity_perceived"].fillna("Missing").value_counts().rename(
            "selected_before_age_matching"
        ),
    ], axis=1).fillna(0).astype(int).sort_index()

    counts["retained_percentage"] = (
        100 * counts["selected_before_age_matching"]
        / counts["original_manifest"]
    )
    save_table(counts, "04_selection_counts", index=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    counts[["original_manifest", "selected_before_age_matching"]].plot.bar(
        ax=ax, color=["#999999", "#0072B2"]
    )
    ax.set(
        title="CFD models before and after study selection",
        xlabel="Dominant perceived category",
        ylabel="Number of models",
    )
    ax.tick_params(axis="x", rotation=35)
    ax.legend(["Original manifest", "Selected, before age matching"])
    fig.tight_layout()
    save_figure(fig, "04_selection_counts")

    # 5. Perceptual response proportions within the selected pool.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    bins = np.linspace(0, 1, 21)

    for ax, group in zip(axes, GROUPS):
        values = pd.to_numeric(
            selected.loc[
                selected["ethnicity_perceived"] == group,
                "ethnicity_perceived_probability",
            ],
            errors="coerce",
        ).dropna()

        ax.hist(values, bins=bins, color=COLORS[group], edgecolor="white")
        ax.axvline(
            MINIMUM_PERCEIVED_PROBABILITY,
            color="#333333", linestyle="--", label="Selection threshold",
        )
        ax.set(
            title=f"{group}-perceived (n={len(values)})",
            xlabel="Proportion of norming responses in dominant category",
            xlim=(0, 1),
        )
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Number of selected models")
    fig.tight_layout()
    save_figure(fig, "05_selected_perceptual_proportions")

    # 6. Rated age in the selected pool, before age matching.
    age = selected[["ethnicity_perceived", "age_rated"]].copy()
    age["age_rated"] = pd.to_numeric(age["age_rated"], errors="coerce")

    age_summary = age.groupby("ethnicity_perceived")["age_rated"].agg(
        n_valid="count",
        mean="mean",
        sd="std",
        median="median",
        minimum="min",
        maximum="max",
    )
    age_summary["n_missing"] = (
        age.groupby("ethnicity_perceived").size() - age_summary["n_valid"]
    )
    save_table(age_summary, "06_selected_rated_age_summary", index=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for group in GROUPS:
        values = np.sort(
            age.loc[age["ethnicity_perceived"] == group, "age_rated"]
            .dropna().to_numpy()
        )
        if len(values):
            cumulative = np.arange(1, len(values) + 1) / len(values)
            ax.step(
                values, cumulative, where="post",
                color=COLORS[group], linewidth=2,
                label=f"{group}-perceived (n={len(values)})",
            )

    ax.set(
        title="Rated age before age matching",
        xlabel="Mean rated age (years)",
        ylabel="Cumulative proportion of selected models",
        ylim=(0, 1.02),
    )
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "06_selected_rated_age")

    (OUTPUT / "README.md").write_text(
        "# CFD audit and stimulus selection QC\n\n"
        f"Original manifest: {len(manifest)} models.\n\n"
        f"Selected pool: {len(selected)} models, before age matching.\n\n"
        "Generated by `scripts/stimulus_preparation/02c_plot_stimulus_qc.py`.\n\n"
        "Inputs: `data/interim/cfd_audit/cfd_manifest_harmonised.csv` "
        "and `data/processed/stimuli.csv`.\n\n"
        "Figures are saved as PNG and PDF; aggregate tables as CSV.\n\n"
        "The self/perceived heatmap is normalised within each "
        "self-reported category. It describes category correspondence, "
        "not classification accuracy.\n\n"
        "Perceptual proportions are CFD norming response proportions, "
        "not classifier confidence scores.\n\n"
        "Selection counts compare the full database with the eligible "
        "pool; they do not isolate the effect of individual filters.\n\n"
        "These results do not describe the final age-matched pool "
        "or establish population representativeness.\n",
        encoding="utf-8",
    )

    print(f"Original manifest: {len(manifest)} models")
    print(f"Selected pool: {len(selected)} models")
    print(f"QC outputs saved to: {OUTPUT}")


if __name__ == "__main__":
    main()