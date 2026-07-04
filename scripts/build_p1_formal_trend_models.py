from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "outputs"
FIG = OUT / "figures"

EXCLUDE_SCOPES = {"all_deaths_in_file", "underlying_cause_C34", "underlying_cause_C34_no_record_axis_U071", "covid_u071"}

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
    "autoimmune_broad": "Autoimmune/rheumatic",
    "non_tobacco_substance_opioid": "Substance/opioid",
    "serious_mental_illness": "Serious mental illness",
}


def label(scope: str) -> str:
    return LABELS.get(scope, scope.replace("_", " ").title())


def savefig(name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "svg"]:
        plt.savefig(FIG / f"{name}.{ext}", dpi=400, bbox_inches="tight")
    plt.close()


def fit_scope_trend(df: pd.DataFrame, scope: str, analysis_set: str) -> dict:
    part = df[df["scope"] == scope].sort_values("year").copy()
    part["year_centered"] = part["year"] - part["year"].min()
    y = part["deaths"].astype(float).to_numpy()
    denom = part["denominator_lung_cancer_ucd_deaths"].astype(float).to_numpy()
    x = sm.add_constant(part["year_centered"].astype(float).to_numpy())

    poisson = sm.GLM(y, x, family=sm.families.Poisson(), offset=np.log(denom))
    poisson_fit = poisson.fit(cov_type="HC0")
    beta = float(poisson_fit.params[1])
    se = float(poisson_fit.bse[1])
    lcl = beta - 1.96 * se
    ucl = beta + 1.96 * se
    apc = (np.exp(beta) - 1.0) * 100
    apc_lcl = (np.exp(lcl) - 1.0) * 100
    apc_ucl = (np.exp(ucl) - 1.0) * 100

    prop_pct = y / denom * 100
    wls = sm.WLS(prop_pct, x, weights=denom)
    wls_fit = wls.fit(cov_type="HC0")
    slope = float(wls_fit.params[1])
    slope_se = float(wls_fit.bse[1])
    slope_lcl = slope - 1.96 * slope_se
    slope_ucl = slope + 1.96 * slope_se

    first = part.iloc[0]
    last = part.iloc[-1]
    return {
        "analysis_set": analysis_set,
        "scope": scope,
        "label": label(scope),
        "start_year": int(first["year"]),
        "end_year": int(last["year"]),
        "start_deaths": int(first["deaths"]),
        "end_deaths": int(last["deaths"]),
        "start_denominator": int(first["denominator_lung_cancer_ucd_deaths"]),
        "end_denominator": int(last["denominator_lung_cancer_ucd_deaths"]),
        "start_proportion_pct": float(first["deaths"] / first["denominator_lung_cancer_ucd_deaths"] * 100),
        "end_proportion_pct": float(last["deaths"] / last["denominator_lung_cancer_ucd_deaths"] * 100),
        "absolute_change_pct_points": float(last["deaths"] / last["denominator_lung_cancer_ucd_deaths"] * 100 - first["deaths"] / first["denominator_lung_cancer_ucd_deaths"] * 100),
        "apc_pct_per_year": apc,
        "apc_lcl": apc_lcl,
        "apc_ucl": apc_ucl,
        "apc_p_value": float(poisson_fit.pvalues[1]),
        "absolute_slope_pct_points_per_year": slope,
        "absolute_slope_lcl": slope_lcl,
        "absolute_slope_ucl": slope_ucl,
        "absolute_slope_p_value": float(wls_fit.pvalues[1]),
    }


def build_trends() -> tuple[pd.DataFrame, pd.DataFrame]:
    main = pd.read_csv(OUT / "P1_mcod_2018_2024_comorbidity_counts_long.csv")
    main = main[~main["scope"].isin(EXCLUDE_SCOPES)].copy()
    main = main.rename(columns={"proportion_among_lung_cancer_deaths": "proportion"})
    main["analysis_set"] = "main_all_lung_cancer_ucd_deaths"
    main = main[["analysis_set", "year", "scope", "deaths", "denominator_lung_cancer_ucd_deaths"]]

    covid_counts_path = OUT / "P1_covid_sensitivity_counts_2018_2024.csv"
    noncovid = pd.DataFrame()
    if covid_counts_path.exists():
        covid = pd.read_csv(covid_counts_path)
        noncovid = covid[covid["analysis_set"] == "non_covid_lung_cancer_ucd_deaths"].copy()
        noncovid = noncovid[~noncovid["scope"].isin(EXCLUDE_SCOPES)].copy()
        noncovid["analysis_set"] = "non_covid_lung_cancer_ucd_deaths"
        noncovid = noncovid[["analysis_set", "year", "scope", "deaths", "denominator_lung_cancer_ucd_deaths"]]

    all_sets = [main]
    if not noncovid.empty:
        all_sets.append(noncovid)
    combined = pd.concat(all_sets, ignore_index=True)
    for col in ["year", "deaths", "denominator_lung_cancer_ucd_deaths"]:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    rows = []
    for analysis_set, data in combined.groupby("analysis_set"):
        for scope in sorted(data["scope"].unique()):
            try:
                rows.append(fit_scope_trend(data, scope, analysis_set))
            except Exception as exc:
                rows.append({"analysis_set": analysis_set, "scope": scope, "label": label(scope), "model_error": str(exc)})
    trends = pd.DataFrame(rows)
    for analysis_set, idx in trends.groupby("analysis_set").groups.items():
        p = pd.to_numeric(trends.loc[idx, "apc_p_value"], errors="coerce").fillna(1)
        trends.loc[idx, "apc_fdr_q_value"] = multipletests(p, method="fdr_bh")[1]
        p2 = pd.to_numeric(trends.loc[idx, "absolute_slope_p_value"], errors="coerce").fillna(1)
        trends.loc[idx, "absolute_slope_fdr_q_value"] = multipletests(p2, method="fdr_bh")[1]
    return combined, trends


def draw_apc_figure(trends: pd.DataFrame) -> None:
    main = trends[trends["analysis_set"] == "main_all_lung_cancer_ucd_deaths"].copy()
    main = main.dropna(subset=["apc_pct_per_year"]).sort_values("absolute_change_pct_points", ascending=True)
    colors = np.where(main["apc_fdr_q_value"] < 0.05, "#2a9d8f", "#98a2b3")
    y = np.arange(main.shape[0])
    plt.figure(figsize=(8.2, 6.0))
    ax = plt.gca()
    ax.errorbar(
        main["apc_pct_per_year"],
        y,
        xerr=[main["apc_pct_per_year"] - main["apc_lcl"], main["apc_ucl"] - main["apc_pct_per_year"]],
        fmt="none",
        ecolor="#98a2b3",
        elinewidth=1.0,
        capsize=2,
        zorder=1,
    )
    ax.scatter(main["apc_pct_per_year"], y, c=colors, s=28, zorder=2)
    ax.axvline(0, color="#667085", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(main["label"], fontsize=8)
    ax.set_xlabel("Annual percent change in co-mention proportion (%)")
    ax.set_title("Formal trend models for co-mentioned conditions, 2018-2024")
    ax.grid(axis="x", alpha=0.22)
    savefig("Figure9_P1_formal_trend_models_apc_2018_2024")


def draw_covid_sensitivity_figure(combined: pd.DataFrame) -> None:
    groups = ["respiratory_failure", "pneumonia_influenza", "copd", "heart_failure", "atrial_fibrillation", "diabetes"]
    data = combined[combined["scope"].isin(groups)].copy()
    data["proportion_pct"] = data["deaths"] / data["denominator_lung_cancer_ucd_deaths"] * 100
    plt.figure(figsize=(9.2, 5.6))
    ax = plt.gca()
    color_map = {
        "respiratory_failure": "#d95f02",
        "pneumonia_influenza": "#e7298a",
        "copd": "#1f77b4",
        "heart_failure": "#e6ab02",
        "atrial_fibrillation": "#1b9e77",
        "diabetes": "#66a61e",
    }
    for group in groups:
        main = data[(data["scope"] == group) & (data["analysis_set"] == "main_all_lung_cancer_ucd_deaths")].sort_values("year")
        noncovid = data[(data["scope"] == group) & (data["analysis_set"] == "non_covid_lung_cancer_ucd_deaths")].sort_values("year")
        ax.plot(main["year"], main["proportion_pct"], color=color_map[group], linewidth=2.0, marker="o", markersize=3.5, label=label(group))
        if not noncovid.empty:
            ax.plot(noncovid["year"], noncovid["proportion_pct"], color=color_map[group], linewidth=1.5, linestyle="--", alpha=0.75)
    ax.set_xlabel("Year")
    ax.set_ylabel("Co-mentioned deaths (%)")
    ax.set_title("COVID sensitivity: solid main analysis, dashed excluding record-axis U07.1")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    savefig("Figure10_P1_covid_sensitivity_trends")


def write_report(trends: pd.DataFrame) -> None:
    main = trends[trends["analysis_set"] == "main_all_lung_cancer_ucd_deaths"].copy()
    main = main.sort_values("absolute_change_pct_points", ascending=False)
    noncovid = trends[trends["analysis_set"] == "non_covid_lung_cancer_ucd_deaths"].copy()

    lines = [
        "# P1 Formal Trend Model Report",
        "",
        "## Model",
        "",
        "For each co-mentioned condition, annual trends were modeled using a Poisson GLM with log lung-cancer-death denominator as an offset. Annual percent change (APC) is reported as exp(beta_year)-1. A weighted least-squares model of annual percentages was also fitted to estimate absolute percentage-point change per year.",
        "",
        "## Main 2018-2024 APC results",
        "",
    ]
    for _, row in main.head(10).iterrows():
        lines.append(
            f"- {row['label']}: APC {row['apc_pct_per_year']:+.2f}%/year "
            f"({row['apc_lcl']:+.2f} to {row['apc_ucl']:+.2f}), "
            f"absolute change {row['absolute_change_pct_points']:+.2f} percentage points; "
            f"FDR q={row['apc_fdr_q_value']:.3g}."
        )

    if not noncovid.empty:
        merged = main[["scope", "label", "apc_pct_per_year"]].rename(columns={"apc_pct_per_year": "main_apc"}).merge(
            noncovid[["scope", "apc_pct_per_year"]].rename(columns={"apc_pct_per_year": "noncovid_apc"}),
            on="scope",
            how="left",
        )
        merged["apc_difference_noncovid_minus_main"] = merged["noncovid_apc"] - merged["main_apc"]
        merged = merged.sort_values("apc_difference_noncovid_minus_main")
        lines += [
            "",
            "## COVID sensitivity comparison",
            "",
        ]
        for _, row in merged.head(8).iterrows():
            lines.append(
                f"- {row['label']}: main APC {row['main_apc']:+.2f}%/year; "
                f"non-COVID APC {row['noncovid_apc']:+.2f}%/year; "
                f"difference {row['apc_difference_noncovid_minus_main']:+.2f}."
            )

    lines += [
        "",
        "## Interpretation",
        "",
        "The formal models convert the descriptive 2018-2024 changes into annualized trend estimates. These results should be presented as trends in death-certificate co-mentions, not incidence or clinical prevalence.",
    ]
    (OUT / "P1_formal_trend_model_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    combined, trends = build_trends()
    combined.to_csv(OUT / "P1_trend_model_input_counts_2018_2024.csv", index=False, encoding="utf-8-sig")
    trends.to_csv(OUT / "P1_formal_trend_model_results_2018_2024.csv", index=False, encoding="utf-8-sig")
    draw_apc_figure(trends)
    draw_covid_sensitivity_figure(combined)
    write_report(trends)
    print("Wrote formal trend model outputs.")


if __name__ == "__main__":
    main()
