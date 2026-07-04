from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "outputs"
FIG = OUT / "figures"

LABELS = {
    "copd": "COPD",
    "respiratory_failure": "Respiratory failure",
    "ischemic_heart_disease": "Ischemic heart disease",
    "pneumonia_influenza": "Pneumonia/influenza",
    "diabetes": "Diabetes",
    "heart_failure": "Heart failure",
    "atrial_fibrillation": "Atrial fibrillation",
    "cerebrovascular": "Cerebrovascular disease",
    "pulmonary_embolism": "Pulmonary embolism",
    "ckd": "CKD",
    "ild": "ILD",
    "depression_anxiety": "Depression/anxiety",
    "autoimmune_broad": "Autoimmune broad",
    "non_tobacco_substance_opioid": "Non-tobacco substance/opioid",
    "serious_mental_illness": "Serious mental illness",
}


def savefig(name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "svg"]:
        plt.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close()


def heatmap(df: pd.DataFrame, title: str, output_name: str, figsize: tuple[float, float]) -> None:
    matrix = df.pivot(index="stratum_label", columns="group", values="proportion_among_stratum_lung_cancer_deaths")
    matrix = matrix.fillna(0)
    matrix = matrix.rename(columns={c: LABELS.get(c, c) for c in matrix.columns})

    values = matrix.values * 100
    plt.figure(figsize=figsize)
    ax = plt.gca()
    im = ax.imshow(values, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(np.arange(values.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(values.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=8)
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Co-mentioned (%)")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if values[i, j] >= 0.5:
                ax.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center", fontsize=6, color="black")
    savefig(output_name)


def main() -> None:
    counts = pd.read_csv(OUT / "P1_mcod_2018_2024_stratified_comorbidity_counts.csv")
    national = pd.read_csv(OUT / "P1_mcod_2018_2024_comorbidity_counts_long.csv")

    top_groups = (
        national[(national["year"] == 2024) & (~national["scope"].isin(["all_deaths_in_file", "underlying_cause_C34"]))]
        .sort_values("proportion_among_lung_cancer_deaths", ascending=False)
        .head(10)["scope"]
        .tolist()
    )

    sex_df = counts[(counts["year"] == 2024) & (counts["dimension"] == "sex") & (counts["group"].isin(top_groups))].copy()
    sex_df["stratum_label"] = pd.Categorical(sex_df["stratum_label"], ["Male", "Female"], ordered=True)
    sex_df = sex_df.sort_values("stratum_label")
    heatmap(
        sex_df,
        "2024 comorbidity proportions by sex",
        "Figure4_P1_2024_sex_stratified_heatmap",
        (8.8, 2.6),
    )

    age_order = ["45-54 years", "55-64 years", "65-74 years", "75-84 years", "85+ years"]
    age_df = counts[
        (counts["year"] == 2024)
        & (counts["dimension"] == "age12")
        & (counts["stratum_label"].isin(age_order))
        & (counts["group"].isin(top_groups))
    ].copy()
    age_df["stratum_label"] = pd.Categorical(age_df["stratum_label"], age_order, ordered=True)
    age_df = age_df.sort_values("stratum_label")
    heatmap(
        age_df,
        "2024 comorbidity proportions by age",
        "Figure5_P1_2024_age_stratified_heatmap",
        (8.8, 4.2),
    )

    race_order = [
        "Non-Hispanic White only",
        "Non-Hispanic Black only",
        "Non-Hispanic Asian only",
        "Mexican",
        "Puerto Rican",
        "Cuban",
        "Other/unknown Hispanic",
        "Non-Hispanic AIAN only",
        "Non-Hispanic more than one race",
    ]
    race_groups = top_groups[:8]
    race_df = counts[
        (counts["year"] == 2024)
        & (counts["dimension"] == "race_ethnicity_2022plus")
        & (counts["stratum_label"].isin(race_order))
        & (counts["group"].isin(race_groups))
    ].copy()
    race_df["stratum_label"] = pd.Categorical(race_df["stratum_label"], race_order, ordered=True)
    race_df = race_df.sort_values("stratum_label")
    heatmap(
        race_df,
        "2024 comorbidity proportions by race/Hispanic origin",
        "SuppFigS1_P1_2024_race_ethnicity_stratified_heatmap",
        (8.4, 5.6),
    )
    print(f"Wrote stratified figures to {FIG}")


if __name__ == "__main__":
    main()
