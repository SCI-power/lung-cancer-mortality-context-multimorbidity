from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "outputs"
FIG = OUT / "figures"


def load_results() -> pd.DataFrame:
    df = pd.read_csv(OUT / "P1_external_triangulation_weighted_estimates.csv")
    for col in ["estimate_display", "ci_low_display", "ci_high_display", "unweighted_n", "unweighted_events"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def pick_estimate(df: pd.DataFrame, dataset: str, domain: str, metric: str, cycle: str | None = None) -> dict:
    sub = df[(df["dataset"] == dataset) & (df["domain"] == domain) & (df["metric"] == metric)]
    if cycle is not None:
        sub = sub[sub["cycle"] == cycle]
    if sub.empty:
        return {}
    row = sub.iloc[0].to_dict()
    return row


def build_concordance(est: pd.DataFrame) -> pd.DataFrame:
    node = pd.read_csv(OUT / "P1_node_table_1999_2024.csv")
    node_2024 = node[node["year"] == 2024].set_index("group")

    rows = []

    def add(signal, mcod_group, external_source, external_metric, interpretation, action, external_value=None, external_ci=None, external_flag=None):
        mcod_pct = float(node_2024.loc[mcod_group, "node_proportion_pct"]) if mcod_group in node_2024.index else np.nan
        mcod_class = node_2024.loc[mcod_group, "node_class"] if mcod_group in node_2024.index else ""
        if external_value is None:
            external_level = "not measured"
        elif external_value >= 10:
            external_level = "high"
        elif external_value >= 5:
            external_level = "moderate"
        else:
            external_level = "low"
        if pd.isna(mcod_pct):
            mcod_level = "not measured"
        elif mcod_pct >= 10:
            mcod_level = "high"
        elif mcod_pct >= 5:
            mcod_level = "moderate"
        else:
            mcod_level = "low"
        rows.append(
            {
                "signal": signal,
                "mcod_group": mcod_group,
                "mcod_2024_pct": mcod_pct,
                "mcod_level": mcod_level,
                "mcod_class": mcod_class,
                "external_source": external_source,
                "external_metric": external_metric,
                "external_pct_or_mean": external_value,
                "external_ci": external_ci,
                "external_reliability": external_flag,
                "external_level": external_level,
                "triangulated_interpretation": interpretation,
                "manuscript_action": action,
            }
        )

    nhis_lung = est[(est["dataset"] == "NHIS") & (est["domain"] == "Lung cancer history")]
    meps_lung = est[(est["dataset"] == "MEPS") & (est["domain"] == "Lung cancer history")]
    nhanes_cancer_2021 = est[(est["dataset"] == "NHANES") & (est["domain"] == "Any cancer history") & (est["cycle"] == "2021-2023")]

    def metric_value(sub: pd.DataFrame, metric: str):
        row = sub[sub["metric"] == metric]
        if row.empty:
            return None, None, None
        r = row.iloc[0]
        return (
            float(r["estimate_display"]),
            f'{float(r["ci_low_display"]):.1f}-{float(r["ci_high_display"]):.1f}',
            r["reliability_flag"],
        )

    v, ci, flag = metric_value(nhis_lung, "COPD")
    add("COPD", "copd", "NHIS lung-cancer-history adults", "COPD", "High-confidence chronic-core concordance: high MCOD co-mention and high premortem survey burden.", "Primary triangulated chronic-core signal", v, ci, flag)

    v, ci, flag = metric_value(nhis_lung, "Diabetes")
    add("Diabetes", "diabetes", "NHIS lung-cancer-history adults", "Diabetes", "Externally common chronic burden with moderate MCOD death-context representation.", "Chronic-core secondary signal; emphasize underrepresentation on death certificates", v, ci, flag)

    v, ci, flag = metric_value(nhis_lung, "Coronary heart disease")
    add("Coronary heart disease / IHD", "ischemic_heart_disease", "NHIS lung-cancer-history adults", "Coronary heart disease", "Externally common cardiovascular comorbidity with moderate MCOD representation.", "Cardiovascular chronic-core secondary signal", v, ci, flag)

    v, ci, flag = metric_value(nhis_lung, "Depression")
    add("Depression/anxiety", "depression_anxiety", "NHIS lung-cancer-history adults", "Depression", "High external burden but low MCOD co-mention; likely death-certificate under-ascertainment for mental health burden.", "Use as death-silent burden example, not mortality-core claim", v, ci, flag)

    v, ci, flag = metric_value(nhanes_cancer_2021, "Kidney disease")
    add("Kidney disease / CKD", "ckd", "NHANES any-cancer adults, 2021-2023", "Kidney disease", "Moderate external cancer-context burden but low MCOD co-mention.", "Contextual signal; not a primary MCOD network claim", v, ci, flag)

    v, ci, flag = metric_value(nhis_lung, "Ever smoked")
    add("Smoking exposure", "non_tobacco_substance_opioid", "NHIS lung-cancer-history adults", "Ever smoked", "High etiologic exposure burden; not equivalent to comorbidity and not directly captured as MCOD tobacco dependence.", "Report as exposure-context validation only", v, ci, flag)

    add("Respiratory failure", "respiratory_failure", "External survey data", "Not a premortem chronic condition construct", "High MCOD signal without matching external chronic construct; likely terminal pathway/death-process signal.", "Keep separate from chronic-core comorbidity interpretation")
    add("Pneumonia/influenza", "pneumonia_influenza", "External survey data", "Not a stable premortem chronic construct", "Moderate MCOD signal without chronic external analogue; likely acute/terminal pathway.", "Keep as terminal/acute pathway sensitivity signal")

    concordance = pd.DataFrame(rows)
    concordance.to_csv(OUT / "P1_external_triangulation_mcod_external_concordance.csv", index=False, encoding="utf-8-sig")
    return concordance


def fig_nhis_lung(est: pd.DataFrame) -> None:
    sub = est[(est["dataset"] == "NHIS") & (est["domain"] == "Lung cancer history")]
    order = ["COPD", "Diabetes", "Hypertension", "Coronary heart disease", "Depression", "Ever smoked", "Current smoking"]
    sub = sub[sub["metric"].isin(order)].copy()
    sub["metric"] = pd.Categorical(sub["metric"], categories=order, ordered=True)
    sub = sub.sort_values("metric")
    y = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.errorbar(
        sub["estimate_display"],
        y,
        xerr=[sub["estimate_display"] - sub["ci_low_display"], sub["ci_high_display"] - sub["estimate_display"]],
        fmt="o",
        color="#1f6f78",
        ecolor="#7aa6ad",
        elinewidth=2,
        capsize=3,
    )
    for idx, row in enumerate(sub.itertuples(index=False)):
        ax.text(row.estimate_display + 1.2, idx, f"n={int(row.unweighted_n)}", va="center", fontsize=8.5, color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels(sub["metric"])
    ax.set_xlabel("Weighted prevalence, % (95% CI)")
    ax.set_xlim(0, min(100, max(sub["ci_high_display"].max() + 10, 45)))
    ax.grid(axis="x", color="#dddddd", linewidth=0.8)
    ax.set_title("NHIS 2023-2024: premortem burden among adults with lung cancer history")
    ax.text(0, -1.05, "Complex survey estimates using pooled WTFA_A/2; reliability flags are stable for displayed estimates.", fontsize=8.5, color="#444444")
    fig.tight_layout()
    for ext in ["png", "svg"]:
        fig.savefig(FIG / f"Figure29_P1_NHIS_lung_cancer_history_weighted_prevalence.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)


def fig_concordance(concordance: pd.DataFrame) -> None:
    plot = concordance.dropna(subset=["external_pct_or_mean", "mcod_2024_pct"]).copy()
    keep = ["COPD", "Diabetes", "Coronary heart disease / IHD", "Depression/anxiety", "Kidney disease / CKD"]
    plot = plot[plot["signal"].isin(keep)]
    colors = {
        "Primary triangulated chronic-core signal": "#1f6f78",
        "Chronic-core secondary signal; emphasize underrepresentation on death certificates": "#498c5a",
        "Cardiovascular chronic-core secondary signal": "#498c5a",
        "Use as death-silent burden example, not mortality-core claim": "#bf7b30",
        "Contextual signal; not a primary MCOD network claim": "#8a8a8a",
    }
    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    for _, row in plot.iterrows():
        ax.scatter(row["external_pct_or_mean"], row["mcod_2024_pct"], s=105, color=colors.get(row["manuscript_action"], "#444444"), edgecolor="white", linewidth=0.8)
        ax.text(row["external_pct_or_mean"] + 0.8, row["mcod_2024_pct"] + 0.15, row["signal"], fontsize=8.5)
    ax.axhline(10, color="#999999", linestyle="--", linewidth=0.8)
    ax.axvline(10, color="#999999", linestyle="--", linewidth=0.8)
    ax.set_xlabel("External premortem/context estimate, %")
    ax.set_ylabel("MCOD 2024 co-mention among lung cancer deaths, %")
    ax.set_title("MCOD-external triangulation: chronic core vs death-silent burden")
    ax.grid(color="#e0e0e0", linewidth=0.7)
    ax.set_xlim(0, max(50, plot["external_pct_or_mean"].max() + 5))
    ax.set_ylim(0, max(22, plot["mcod_2024_pct"].max() + 2))
    fig.tight_layout()
    for ext in ["png", "svg"]:
        fig.savefig(FIG / f"Figure30_P1_MCOD_external_concordance_scatter.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)


def fig_meps_burden(est: pd.DataFrame) -> None:
    metrics = ["Total annual expenditure", "Office-based visits", "Outpatient visits", "Emergency room visits", "Inpatient discharges", "Prescription fills"]
    sub = est[(est["dataset"] == "MEPS") & (est["domain"].isin(["Any cancer history", "Lung cancer history"])) & (est["metric"].isin(metrics))].copy()
    sub["metric"] = pd.Categorical(sub["metric"], categories=metrics, ordered=True)
    sub = sub.sort_values(["metric", "domain"])
    colors = {"Any cancer history": "#5b8c5a", "Lung cancer history": "#1f6f78"}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={"width_ratios": [0.75, 1.45]})

    exp = sub[sub["metric"] == "Total annual expenditure"].copy()
    x = np.arange(len(exp))
    axes[0].bar(x, exp["estimate_display"], color=[colors[d] for d in exp["domain"]])
    axes[0].errorbar(x, exp["estimate_display"], yerr=[exp["estimate_display"] - exp["ci_low_display"], exp["ci_high_display"] - exp["estimate_display"]], fmt="none", ecolor="#333333", capsize=3, linewidth=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["Any cancer", "Lung cancer"], fontsize=9)
    axes[0].set_ylabel("Weighted mean (95% CI)")
    axes[0].set_title("Annual expenditure")
    axes[0].grid(axis="y", color="#e0e0e0", linewidth=0.7)

    util_metrics = ["Office-based visits", "Outpatient visits", "Emergency room visits", "Inpatient discharges", "Prescription fills"]
    short = {
        "Office-based visits": "Office",
        "Outpatient visits": "Outpatient",
        "Emergency room visits": "ER",
        "Inpatient discharges": "Inpatient",
        "Prescription fills": "Rx fills",
    }
    positions = np.arange(len(util_metrics))
    width = 0.36
    for i, domain in enumerate(["Any cancer history", "Lung cancer history"]):
        p = sub[(sub["domain"] == domain) & (sub["metric"].isin(util_metrics))].copy()
        p["metric"] = pd.Categorical(p["metric"], categories=util_metrics, ordered=True)
        p = p.sort_values("metric")
        offset = (i - 0.5) * width
        axes[1].bar(positions + offset, p["estimate_display"], width=width, color=colors[domain], label="Any cancer" if domain == "Any cancer history" else "Lung cancer")
        axes[1].errorbar(positions + offset, p["estimate_display"], yerr=[p["estimate_display"] - p["ci_low_display"], p["ci_high_display"] - p["estimate_display"]], fmt="none", ecolor="#333333", capsize=2, linewidth=0.9)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([short[m] for m in util_metrics], fontsize=9)
    axes[1].set_title("Annual utilization counts")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", color="#e0e0e0", linewidth=0.7)
    fig.suptitle("MEPS 2023: health-care burden context for cancer and lung cancer history", y=1.02)
    fig.tight_layout()
    for ext in ["png", "svg"]:
        fig.savefig(FIG / f"Figure31_P1_MEPS_healthcare_burden_context.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)


def fig_nhanes_context(est: pd.DataFrame) -> None:
    metrics = ["COPD/emphysema/chronic bronchitis", "Diabetes", "Hypertension", "Kidney disease", "Current smoking"]
    sub = est[(est["dataset"] == "NHANES") & (est["domain"] == "Any cancer history") & (est["metric"].isin(metrics))].copy()
    sub["metric"] = pd.Categorical(sub["metric"], categories=metrics, ordered=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    width = 0.36
    positions = np.arange(len(metrics))
    for i, cycle in enumerate(["2017-2018", "2021-2023"]):
        p = sub[sub["cycle"] == cycle].sort_values("metric")
        offset = (i - 0.5) * width
        ax.bar(positions + offset, p["estimate_display"], width=width, label=cycle, color="#89a7c2" if i == 0 else "#315f72")
        ax.errorbar(positions + offset, p["estimate_display"], yerr=[p["estimate_display"] - p["ci_low_display"], p["ci_high_display"] - p["estimate_display"]], fmt="none", ecolor="#333333", capsize=2, linewidth=0.9)
    ax.set_xticks(positions)
    ax.set_xticklabels(["COPD/emphysema", "Diabetes", "Hypertension", "Kidney disease", "Current smoking"], rotation=20, ha="right")
    ax.set_ylabel("Weighted prevalence, % (95% CI)")
    ax.set_title("NHANES cancer-history context: biomarker/exposure-adjacent burden by cycle")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.7)
    fig.tight_layout()
    for ext in ["png", "svg"]:
        fig.savefig(FIG / f"Figure32_P1_NHANES_cancer_context_by_cycle.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)


def write_report(est: pd.DataFrame, concordance: pd.DataFrame) -> None:
    nhis_lung = est[(est["dataset"] == "NHIS") & (est["domain"] == "Lung cancer history")]
    rows = []
    for metric in ["COPD", "Diabetes", "Hypertension", "Coronary heart disease", "Depression", "Ever smoked", "Current smoking"]:
        r = nhis_lung[nhis_lung["metric"] == metric].iloc[0]
        rows.append(f"- {metric}: {r.estimate_display:.1f}% (95% CI {r.ci_low_display:.1f}-{r.ci_high_display:.1f}); unweighted n={int(r.unweighted_n)}, events={int(r.unweighted_events)}.")
    concordance_brief = []
    for _, r in concordance.iterrows():
        ext = "not externally measured" if pd.isna(r["external_pct_or_mean"]) else f'{r["external_pct_or_mean"]:.1f}%'
        concordance_brief.append(f"- {r['signal']}: MCOD 2024 {r['mcod_2024_pct']:.1f}% vs external {ext}; {r['triangulated_interpretation']}")
    body = f"""# P1 external public-data triangulation results and standards

## Methods standard used

1. NHIS 2023-2024 was analyzed as a pooled two-year public-use Sample Adult dataset. The adult weight was divided by two, and year-specific strata/PSU identifiers were used to avoid cross-year design collisions.
2. MEPS 2023 was analyzed with `PERWT23F`, `VARSTR`, and `VARPSU`; HC-249 condition-file ICD-10 clusters were merged to HC-251 by `DUPERSID`.
3. NHANES 2017-2018 and August 2021-August 2023 were analyzed by cycle rather than pooled, because the 2021-2023 cycle is a special pre-pandemic/post-pandemic combined cycle. Interview weights were used for questionnaire variables and MEC weights for examination/laboratory variables.
4. All estimates used R `survey`: `svydesign`, Taylor-linearized standard errors, and `svyciprop(..., method="beta")` for proportions.
5. Reliability flags use conservative publication screening: suppress if unweighted denominator <30, events <10 for proportions, or RSE >50%; unstable if denominator <50 or RSE >30%.
6. External datasets are used for contextual triangulation only. They do not convert MCOD death-certificate co-mentions into clinical comorbidity prevalence.

## Key NHIS lung-cancer-history results

{chr(10).join(rows)}

## MCOD-external interpretation

{chr(10).join(concordance_brief)}

## Files generated

- `P1_external_triangulation_weighted_estimates.csv`
- `P1_external_triangulation_mcod_external_concordance.csv`
- `P1_external_triangulation_results_v1.xlsx`
- `Figure29_P1_NHIS_lung_cancer_history_weighted_prevalence.png/svg`
- `Figure30_P1_MCOD_external_concordance_scatter.png/svg`
- `Figure31_P1_MEPS_healthcare_burden_context.png/svg`
- `Figure32_P1_NHANES_cancer_context_by_cycle.png/svg`
"""
    (OUT / "P1_external_triangulation_results_and_methods_report.md").write_text(body, encoding="utf-8")


def write_excel(est: pd.DataFrame, concordance: pd.DataFrame) -> None:
    with pd.ExcelWriter(OUT / "P1_external_triangulation_results_v1.xlsx", engine="openpyxl") as writer:
        est.to_excel(writer, sheet_name="weighted_estimates", index=False)
        concordance.to_excel(writer, sheet_name="concordance", index=False)
        est[(est["dataset"] == "NHIS") & (est["domain"] == "Lung cancer history")].to_excel(writer, sheet_name="nhis_lung", index=False)
        est[(est["dataset"] == "MEPS") & (est["domain"] == "Lung cancer history")].to_excel(writer, sheet_name="meps_lung", index=False)
        est[(est["dataset"] == "NHANES") & (est["domain"] == "Any cancer history")].to_excel(writer, sheet_name="nhanes_cancer", index=False)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    est = load_results()
    concordance = build_concordance(est)
    fig_nhis_lung(est)
    fig_concordance(concordance)
    fig_meps_burden(est)
    fig_nhanes_context(est)
    write_excel(est, concordance)
    write_report(est, concordance)
    print("Triangulation summary complete.")


if __name__ == "__main__":
    main()

