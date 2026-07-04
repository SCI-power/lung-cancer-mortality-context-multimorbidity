from pathlib import Path
import re
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT / "outputs"
FIGURES = OUTPUTS / "figures"
ASSETS = OUTPUTS / "nhis_comparator_assets"

FIGURES.mkdir(parents=True, exist_ok=True)

PREV_PATH = OUTPUTS / "P1_nhis_lung_vs_comparator_weighted_prevalence.csv"
PR_PATH = OUTPUTS / "P1_nhis_lung_vs_comparator_adjusted_pr.csv"
QC_PATH = OUTPUTS / "P1_nhis_lung_vs_comparator_qc.csv"

prev = pd.read_csv(PREV_PATH)
pr = pd.read_csv(PR_PATH)
qc = pd.read_csv(QC_PATH)

metric_order = [
    "COPD",
    "Coronary heart disease",
    "Hypertension",
    "Depression",
    "Diabetes",
    "Ever smoked",
    "Current smoking",
]

metric_labels = {
    "COPD": "COPD",
    "Coronary heart disease": "Coronary\nheart disease",
    "Hypertension": "Hypertension",
    "Depression": "Depression",
    "Diabetes": "Diabetes",
    "Ever smoked": "Ever smoked",
    "Current smoking": "Current\nsmoking",
}

group_order = ["No cancer history", "Other cancer history", "Lung cancer history"]
group_colors = {
    "No cancer history": "#4C78A8",
    "Other cancer history": "#59A14F",
    "Lung cancer history": "#C44E52",
}
ref_colors = {
    "No cancer history": "#4C78A8",
    "Other cancer history": "#59A14F",
}


def fmt_ci(row, est_col, lo_col, hi_col, digits=2):
    return f"{row[est_col]:.{digits}f} ({row[lo_col]:.{digits}f}-{row[hi_col]:.{digits}f})"


def save_figure(fig, basename):
    paths = []
    for ext in ("png", "svg", "pdf"):
        path = FIGURES / f"{basename}.{ext}"
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def make_pr_forest():
    data = pr.copy()
    data["metric"] = pd.Categorical(data["metric"], categories=metric_order, ordered=True)
    data = data.sort_values(["metric", "reference_group"])

    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    y_base = np.arange(len(metric_order))[::-1]
    offset = {"No cancer history": 0.13, "Other cancer history": -0.13}

    for ref in ["No cancer history", "Other cancer history"]:
        sub = data[data["reference_group"] == ref].set_index("metric").loc[metric_order].reset_index()
        y = y_base + offset[ref]
        x = sub["adjusted_pr"].to_numpy()
        xerr = np.vstack([
            x - sub["ci_low"].to_numpy(),
            sub["ci_high"].to_numpy() - x,
        ])
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            fmt="o",
            color=ref_colors[ref],
            ecolor=ref_colors[ref],
            elinewidth=1.8,
            capsize=3,
            markersize=5.8,
            label=f"vs {ref}",
            zorder=3,
        )

    ax.axvline(1.0, color="#222222", linewidth=1.1, linestyle="--", zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(0.65, 5.2)
    ax.set_xticks([0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_yticks(y_base)
    ax.set_yticklabels([metric_labels[m].replace("\n", " ") for m in metric_order])
    ax.set_xlabel("Adjusted prevalence ratio for lung cancer history")
    ax.set_title("Adjusted PRs by comparator group", loc="left", fontsize=12, pad=18)
    fig.text(
        0.16,
        0.035,
        "Disease outcomes are adjusted for age, sex, race, Hispanic origin, region, and smoking status; smoking outcomes are demographic-adjusted.",
        fontsize=8.5,
        color="#4A4A4A",
    )
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.13), ncol=1, fontsize=9)
    fig.subplots_adjust(top=0.84, bottom=0.18, left=0.26, right=0.98)
    return save_figure(fig, "Figure34_P1_NHIS_lung_vs_comparator_adjusted_PR")


def make_prevalence_plot():
    data = prev.copy()
    data["metric"] = pd.Categorical(data["metric"], categories=metric_order, ordered=True)
    data["group"] = pd.Categorical(data["group"], categories=group_order, ordered=True)

    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    y_base = np.arange(len(metric_order))[::-1]
    offsets = {
        "No cancer history": 0.18,
        "Other cancer history": 0.00,
        "Lung cancer history": -0.18,
    }

    for group in group_order:
        sub = (
            data[data["group"] == group]
            .set_index("metric")
            .loc[metric_order]
            .reset_index()
        )
        y = y_base + offsets[group]
        x = sub["prevalence_pct"].to_numpy()
        xerr = np.vstack([
            x - sub["ci_low_pct"].to_numpy(),
            sub["ci_high_pct"].to_numpy() - x,
        ])
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            fmt="o",
            color=group_colors[group],
            ecolor=group_colors[group],
            elinewidth=1.7,
            capsize=3,
            markersize=5.6,
            label=group,
            zorder=3,
        )

    ax.set_xlim(0, 88)
    ax.set_yticks(y_base)
    ax.set_yticklabels([metric_labels[m].replace("\n", " ") for m in metric_order])
    ax.set_xlabel("Weighted prevalence, %")
    ax.set_title("NHIS 2023-2024 weighted burden by cancer-history group", loc="left", fontsize=12, pad=12)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    return save_figure(fig, "Figure35_P1_NHIS_lung_vs_comparator_weighted_prevalence")


def copy_asset(src, folder, target_name, asset_type, note, rows):
    dst_dir = ASSETS / folder
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / target_name
    shutil.copy2(src, dst)
    rows.append(
        {
            "section": folder,
            "asset_name": target_name,
            "source_name": src.name,
            "asset_type": asset_type,
            "copied": True,
            "note": note,
            "relative_path": str(dst.relative_to(ASSETS)).replace("\\", "/"),
        }
    )
    return dst


def write_report():
    def row(metric, reference):
        return pr[(pr["metric"] == metric) & (pr["reference_group"] == reference)].iloc[0]

    def p(metric, group):
        return prev[(prev["metric"] == metric) & (prev["group"] == group)].iloc[0]

    lines = [
        "# NHIS lung cancer history comparator analysis",
        "",
        "## Objective",
        "",
        "This add-on analysis addresses a key reviewer concern: whether the high external multimorbidity burden in adults with a history of lung cancer is only a function of age, smoking, or general cancer survivorship rather than a lung-cancer-specific signal.",
        "",
        "## Data and design",
        "",
        "- Dataset: NHIS 2023-2024 pooled public-use sample.",
        "- Groups: no cancer history, other cancer history, and lung cancer history.",
        "- Survey design: pooled two-year weights divided by two, with year-specific strata and PSU identifiers.",
        "- Outcomes: COPD, coronary heart disease, diabetes, hypertension, depression, ever smoking, and current smoking.",
        "- Models: disease outcomes used demographic-and-smoking-adjusted survey-weighted quasipoisson log-link models; smoking outcomes used demographic-adjusted models.",
        "",
        "## Key results",
        "",
        f"- The weighted lung-cancer-history population was approximately {qc.loc[qc['group'] == 'Lung cancer history', 'weighted_population'].iloc[0]:,.0f} adults, with {int(qc.loc[qc['group'] == 'Lung cancer history', 'unweighted_n'].iloc[0])} unweighted respondents.",
        f"- COPD prevalence was {fmt_ci(p('COPD', 'Lung cancer history'), 'prevalence_pct', 'ci_low_pct', 'ci_high_pct', 1)}% in adults with lung cancer history, compared with {fmt_ci(p('COPD', 'Other cancer history'), 'prevalence_pct', 'ci_low_pct', 'ci_high_pct', 1)}% in other cancer history and {fmt_ci(p('COPD', 'No cancer history'), 'prevalence_pct', 'ci_low_pct', 'ci_high_pct', 1)}% in no cancer history.",
        f"- After adjustment, COPD remained strongly higher in lung cancer history versus no cancer history (PR {fmt_ci(row('COPD', 'No cancer history'), 'adjusted_pr', 'ci_low', 'ci_high')}) and versus other cancer history (PR {fmt_ci(row('COPD', 'Other cancer history'), 'adjusted_pr', 'ci_low', 'ci_high')}).",
        f"- Coronary heart disease also remained higher versus no cancer history (PR {fmt_ci(row('Coronary heart disease', 'No cancer history'), 'adjusted_pr', 'ci_low', 'ci_high')}) and other cancer history (PR {fmt_ci(row('Coronary heart disease', 'Other cancer history'), 'adjusted_pr', 'ci_low', 'ci_high')}).",
        f"- Ever smoking and current smoking were consistently higher in lung cancer history versus both comparator groups: ever smoking PR {fmt_ci(row('Ever smoked', 'Other cancer history'), 'adjusted_pr', 'ci_low', 'ci_high')} versus other cancer history; current smoking PR {fmt_ci(row('Current smoking', 'Other cancer history'), 'adjusted_pr', 'ci_low', 'ci_high')} versus other cancer history.",
        f"- Diabetes did not remain elevated after adjustment versus no cancer history (PR {fmt_ci(row('Diabetes', 'No cancer history'), 'adjusted_pr', 'ci_low', 'ci_high')}) or other cancer history (PR {fmt_ci(row('Diabetes', 'Other cancer history'), 'adjusted_pr', 'ci_low', 'ci_high')}).",
        f"- Hypertension and depression were higher versus no cancer history but attenuated versus other cancer history: hypertension PR {fmt_ci(row('Hypertension', 'Other cancer history'), 'adjusted_pr', 'ci_low', 'ci_high')}; depression PR {fmt_ci(row('Depression', 'Other cancer history'), 'adjusted_pr', 'ci_low', 'ci_high')}.",
        "",
        "## Interpretation for the manuscript",
        "",
        "- This analysis strengthens the external-triangulation argument by separating lung-cancer-specific signals from broader cancer-history burden.",
        "- COPD, coronary heart disease, and smoking-related metrics are the most robust lung-cancer-history-enriched signals because they remain elevated even against other cancer history.",
        "- Diabetes appears to represent a broad cardiometabolic burden rather than a lung-cancer-specific external signal.",
        "- Hypertension and depression should be framed as clinically relevant survivorship or multimorbidity burden signals, but not as strongly lung-cancer-specific after comparison with other cancer history.",
        "- The death-certificate visibility-gap argument remains valid, but it should be framed as an ecological contrast between mortality-context co-mention and survey-measured burden, not as clinical prevalence estimated from death certificates.",
        "",
        "## Recommended manuscript placement",
        "",
        "- Promote the adjusted PR forest plot to a main figure panel or candidate main figure, because it directly addresses confounding and specificity.",
        "- Keep the weighted prevalence plot and detailed estimates in supplementary material.",
        "- Add a short paragraph in Results titled 'External specificity analysis using NHIS cancer-history comparators'.",
        "",
        "## Limitations to state explicitly",
        "",
        "- NHIS lung cancer history is self-reported and does not distinguish active disease from remote history.",
        "- The lung-cancer-history subgroup is modest in size, so estimates should be interpreted as external triangulation rather than definitive causal estimates.",
        "- The analysis is cross-sectional and cannot determine whether comorbidities preceded, followed, or were detected because of lung cancer care.",
    ]
    path = OUTPUTS / "P1_nhis_lung_vs_comparator_results_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_reassessment_report():
    lines = [
        "# eClinicalMedicine post-supplement reassessment",
        "",
        "## Added material",
        "",
        "- NHIS 2023-2024 survey-weighted comparator analysis contrasting adults with lung cancer history against adults with no cancer history and adults with other cancer history.",
        "- New adjusted prevalence-ratio forest plot and weighted prevalence plot.",
        "- Three new supplementary tables covering weighted prevalence, adjusted PRs, and group-level QC.",
        "",
        "## Effect on reviewer risk",
        "",
        "- The previous major concern was that external survey burden could be explained by age, smoking, or cancer survivorship generally.",
        "- The new NHIS comparator analysis partly resolves this concern: COPD, coronary heart disease, ever smoking, and current smoking remain elevated against other cancer history; diabetes does not; hypertension and depression attenuate.",
        "- This gives the paper a sharper interpretation: not all external comorbidity signals are lung-cancer-specific, but the respiratory, cardiovascular, and smoking-related signals are robust enough for a clinically interpretable triangulation claim.",
        "",
        "## Revised suitability for eClinicalMedicine",
        "",
        "- Before this supplement: approximately 7.2/10 for eClinicalMedicine readiness.",
        "- After this supplement, if the manuscript compresses figures and writes the death-certificate low-visibility burden carefully: approximately 7.8-8.2/10.",
        "- Remaining barrier to a confident >8-tier submission is the absence of individual-level linked mortality-to-clinical histories. This is acceptable only if the manuscript repeatedly frames the study as mortality-context surveillance plus external public-data triangulation rather than true clinical comorbidity prevalence among decedents.",
        "",
        "## Required wording changes",
        "",
        "- Use 'mortality-context co-mention' rather than 'clinical comorbidity prevalence' for MCOD results.",
        "- Use 'ecological visibility ratio' or 'death-certificate low-mention burden' rather than 'underdiagnosis' or 'underreporting' unless supported by linked clinical data.",
        "- State that NHIS comparator analysis identifies specificity of burden among adults with a history of lung cancer, not among lung cancer decedents.",
    ]
    path = OUTPUTS / "P1_eClinicalMedicine_post_NHIS_supplement_reassessment.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def update_manifest(new_rows):
    manifest_path = ASSETS / "asset_manifest.csv"
    if manifest_path.exists():
        old = pd.read_csv(manifest_path)
    else:
        old = pd.DataFrame(columns=["asset_type", "asset_name", "source_name", "note", "relative_path"])
    names = {r["asset_name"] for r in new_rows}
    old = old[~old["asset_name"].isin(names)]
    combined = pd.concat([old, pd.DataFrame(new_rows)], ignore_index=True)
    combined.to_csv(manifest_path, index=False)


def update_readme():
    readme_path = ASSETS / "README_submission_assets.md"
    text = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# NHIS comparator assets\n\nCopied assets: 0\nPDF figure files now included: 0\n\n"
    start = "<!-- NHIS_COMPARATOR_START -->"
    end = "<!-- NHIS_COMPARATOR_END -->"
    file_count = sum(1 for p in ASSETS.rglob("*") if p.is_file())
    pdf_count = sum(1 for p in ASSETS.rglob("*.pdf") if p.is_file())
    manifest_count = len(pd.read_csv(ASSETS / "asset_manifest.csv")) if (ASSETS / "asset_manifest.csv").exists() else 0
    text = re.sub(r"Copied assets: \d+", f"Copied assets: {manifest_count}", text)
    text = re.sub(r"PDF figure files now included: \d+", f"PDF figure files now included: {pdf_count}", text)
    section = "\n".join(
        [
            start,
            "",
            "## NHIS lung cancer comparator add-on",
            "",
            "- Added candidate `Main_Figure_9_NHIS_lung_vs_comparator_adjusted_PR` as PNG, SVG, and PDF.",
            "- Added `Supplementary_Figure_S20_NHIS_lung_vs_comparator_adjusted_PR` as PNG, SVG, and PDF.",
            "- Added `Supplementary_Figure_S21_NHIS_lung_vs_comparator_weighted_prevalence` as PNG, SVG, and PDF.",
            "- Added `Supplementary_Table_S29_NHIS_lung_vs_comparator_weighted_prevalence.csv`.",
            "- Added `Supplementary_Table_S30_NHIS_lung_vs_comparator_adjusted_PR.csv`.",
            "- Added `Supplementary_Table_S31_NHIS_lung_vs_comparator_QC.csv`.",
            "- Added `Report_10_NHIS_lung_vs_comparator_results.md` and `Report_11_eClinicalMedicine_post_NHIS_supplement_reassessment.md`.",
            "",
            f"Current asset file count: {file_count}.",
            f"Current PDF figure files included: {pdf_count}.",
            "",
            end,
            "",
        ]
    )
    if start in text and end in text:
        prefix = text.split(start)[0].rstrip()
        suffix = text.split(end, 1)[1].lstrip()
        text = prefix + "\n\n" + section + suffix
    else:
        text = text.rstrip() + "\n\n" + section
    readme_path.write_text(text, encoding="utf-8")


def main():
    pr_figs = make_pr_forest()
    prev_figs = make_prevalence_plot()
    report = write_report()
    reassessment = write_reassessment_report()

    new_rows = []

    for src in pr_figs:
        ext = src.suffix
        copy_asset(
            src,
            "main_figures",
            f"Main_Figure_9_NHIS_lung_vs_comparator_adjusted_PR{ext}",
            "figure",
            "Candidate main figure: NHIS adjusted comparator analysis addressing specificity and confounding.",
            new_rows,
        )
        copy_asset(
            src,
            "supplementary_figures",
            f"Supplementary_Figure_S20_NHIS_lung_vs_comparator_adjusted_PR{ext}",
            "figure",
            "Supplementary version of the NHIS adjusted comparator forest plot.",
            new_rows,
        )

    for src in prev_figs:
        ext = src.suffix
        copy_asset(
            src,
            "supplementary_figures",
            f"Supplementary_Figure_S21_NHIS_lung_vs_comparator_weighted_prevalence{ext}",
            "figure",
            "NHIS weighted prevalence by no cancer, other cancer, and lung cancer history.",
            new_rows,
        )

    table_assets = [
        (PREV_PATH, "Supplementary_Table_S29_NHIS_lung_vs_comparator_weighted_prevalence.csv", "NHIS weighted prevalence estimates with confidence intervals."),
        (PR_PATH, "Supplementary_Table_S30_NHIS_lung_vs_comparator_adjusted_PR.csv", "NHIS adjusted prevalence ratios comparing lung cancer history with no cancer and other cancer history."),
        (QC_PATH, "Supplementary_Table_S31_NHIS_lung_vs_comparator_QC.csv", "NHIS comparator group unweighted and weighted sample QC."),
    ]
    for src, name, note in table_assets:
        copy_asset(src, "supplementary_tables", name, "table", note, new_rows)

    copy_asset(report, "reports_and_methods", "Report_10_NHIS_lung_vs_comparator_results.md", "report", "NHIS comparator methods, results, interpretation, and limitations.", new_rows)
    copy_asset(reassessment, "reports_and_methods", "Report_11_eClinicalMedicine_post_NHIS_supplement_reassessment.md", "report", "Updated eClinicalMedicine readiness after NHIS comparator supplement.", new_rows)

    update_manifest(new_rows)
    update_readme()

    print("NHIS comparator summary assets written")
    print(f"Added assets: {len(new_rows)}")
    for row in new_rows:
        print(row["relative_path"])


if __name__ == "__main__":
    main()

