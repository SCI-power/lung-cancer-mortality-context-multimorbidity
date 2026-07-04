from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "outputs"
YEARLY = OUT / "yearly_resident_1999_2024"
FIG = OUT / "figures"

START_YEAR = 1999
END_YEAR = 2024
AGE_REFERENCE_YEAR = 2024

EXCLUDE_SCOPES = {
    "all_deaths_in_file",
    "all_deaths_us_residents",
    "foreign_resident_deaths_excluded",
    "underlying_cause_C34",
    "underlying_cause_C34_no_record_axis_U071",
    "covid_u071",
}
TERMINAL_ACUTE_NODES = {"respiratory_failure", "pneumonia_influenza", "pulmonary_embolism"}
CHRONIC_CORE_EXCLUDE = TERMINAL_ACUTE_NODES

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
    "non_tobacco_substance_opioid": "Non-tobacco substance/opioid",
    "serious_mental_illness": "Serious mental illness",
}

AGE12_LABELS = {
    "01": "Under 1",
    "02": "1-4",
    "03": "5-14",
    "04": "15-24",
    "05": "25-34",
    "06": "35-44",
    "07": "45-54",
    "08": "55-64",
    "09": "65-74",
    "10": "75-84",
    "11": "85+",
    "12": "Unknown",
}
AGE_HEATMAP_CODES = ["07", "08", "09", "10", "11"]


def label(scope: str) -> str:
    return LABELS.get(scope, scope.replace("_", " ").title())


def savefig(name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "svg"]:
        plt.savefig(FIG / f"{name}.{ext}", dpi=400, bbox_inches="tight")
    plt.close()


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    counts = pd.read_csv(OUT / "P1_mcod_1999_2024_comorbidity_counts_long.csv")
    pairs = pd.read_csv(OUT / "P1_edge_enrichment_1999_2024.csv")
    denom = pd.read_csv(OUT / "P1_mcod_1999_2024_age12_denominators.csv", dtype={"age12": str})
    for col in ["year", "deaths", "denominator_lung_cancer_ucd_deaths", "proportion_among_lung_cancer_deaths"]:
        if col in counts.columns:
            counts[col] = pd.to_numeric(counts[col], errors="coerce")
    for col in ["year", "co_mentioned_deaths", "denominator", "lift_vs_independence", "phi_correlation"]:
        if col in pairs.columns:
            pairs[col] = pd.to_numeric(pairs[col], errors="coerce")
    denom["year"] = pd.to_numeric(denom["year"], errors="coerce").astype(int)
    denom["lung_cancer_ucd_deaths"] = pd.to_numeric(denom["lung_cancer_ucd_deaths"], errors="coerce")
    return counts, pairs, denom


def read_age_group_counts() -> pd.DataFrame:
    frames = []
    for year in range(START_YEAR, END_YEAR + 1):
        path = YEARLY / f"P1_mcod_{year}_lung_cancer_group_by_age12.csv"
        df = pd.read_csv(path, dtype={"age12": str})
        frames.append(df)
    age = pd.concat(frames, ignore_index=True)
    age["year"] = pd.to_numeric(age["year"], errors="coerce").astype(int)
    age["deaths"] = pd.to_numeric(age["deaths"], errors="coerce").fillna(0)
    age = age[~age["group"].isin(EXCLUDE_SCOPES)].copy()
    return age


def build_age_standardized(counts: pd.DataFrame, age_counts: pd.DataFrame, denom: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ref = denom[(denom["year"] == AGE_REFERENCE_YEAR) & (~denom["age12"].isin(["12", "Unknown"]))].copy()
    ref = ref[ref["lung_cancer_ucd_deaths"] > 0]
    ref["reference_weight_2024"] = ref["lung_cancer_ucd_deaths"] / ref["lung_cancer_ucd_deaths"].sum()
    ref_weights = ref[["age12", "reference_weight_2024"]]

    groups = sorted(counts[~counts["scope"].isin(EXCLUDE_SCOPES)]["scope"].unique())
    years = list(range(START_YEAR, END_YEAR + 1))
    scaffold = pd.MultiIndex.from_product([years, groups, ref_weights["age12"].tolist()], names=["year", "group", "age12"]).to_frame(index=False)
    age = scaffold.merge(age_counts, on=["year", "group", "age12"], how="left")
    age["deaths"] = age["deaths"].fillna(0)
    age = age.merge(denom[["year", "age12", "lung_cancer_ucd_deaths"]], on=["year", "age12"], how="left")
    age = age.merge(ref_weights, on="age12", how="left")
    age["age12_label"] = age["age12"].map(AGE12_LABELS)
    age["age_specific_proportion"] = np.where(age["lung_cancer_ucd_deaths"] > 0, age["deaths"] / age["lung_cancer_ucd_deaths"], np.nan)
    age["weighted_age_specific_proportion"] = age["age_specific_proportion"] * age["reference_weight_2024"]

    std = (
        age.groupby(["year", "group"], as_index=False)
        .agg(age_standardized_proportion_2024_weights=("weighted_age_specific_proportion", "sum"))
    )
    std["label"] = std["group"].map(label)
    crude = counts[~counts["scope"].isin(EXCLUDE_SCOPES)][
        ["year", "scope", "deaths", "denominator_lung_cancer_ucd_deaths", "proportion_among_lung_cancer_deaths"]
    ].rename(columns={"scope": "group", "proportion_among_lung_cancer_deaths": "crude_proportion"})
    std = std.merge(crude, on=["year", "group"], how="left")
    std["crude_pct"] = std["crude_proportion"] * 100
    std["age_standardized_pct_2024_weights"] = std["age_standardized_proportion_2024_weights"] * 100

    changes = []
    for group, data in std.groupby("group"):
        start = data[data["year"] == START_YEAR].iloc[0]
        end = data[data["year"] == END_YEAR].iloc[0]
        crude_change = end["crude_pct"] - start["crude_pct"]
        std_change = end["age_standardized_pct_2024_weights"] - start["age_standardized_pct_2024_weights"]
        changes.append(
            {
                "group": group,
                "label": label(group),
                "crude_1999_pct": start["crude_pct"],
                "crude_2024_pct": end["crude_pct"],
                "crude_change_pct_points": crude_change,
                "age_standardized_1999_pct_2024_weights": start["age_standardized_pct_2024_weights"],
                "age_standardized_2024_pct_2024_weights": end["age_standardized_pct_2024_weights"],
                "age_standardized_change_pct_points": std_change,
                "age_adjustment_difference_pct_points": std_change - crude_change,
                "age_adjustment_retained_share_pct": std_change / crude_change * 100 if crude_change else np.nan,
            }
        )
    change_df = pd.DataFrame(changes).sort_values("age_standardized_change_pct_points", ascending=False)
    return std, change_df


def fit_poisson_apc(year: pd.Series, deaths: pd.Series, denom: pd.Series) -> tuple[float, float, float, float]:
    year_centered = year.astype(float).to_numpy() - float(year.min())
    y = deaths.astype(float).to_numpy()
    n = denom.astype(float).to_numpy()
    x = sm.add_constant(year_centered)
    model = sm.GLM(y, x, family=sm.families.Poisson(), offset=np.log(n))
    fit = model.fit(cov_type="HC0")
    beta = float(fit.params[1])
    se = float(fit.bse[1])
    return (
        (math.exp(beta) - 1.0) * 100,
        (math.exp(beta - 1.96 * se) - 1.0) * 100,
        (math.exp(beta + 1.96 * se) - 1.0) * 100,
        float(fit.pvalues[1]),
    )


def build_age_specific_apc(age_counts: pd.DataFrame, denom: pd.DataFrame, top_groups: list[str]) -> pd.DataFrame:
    scaffold = pd.MultiIndex.from_product(
        [range(START_YEAR, END_YEAR + 1), top_groups, AGE_HEATMAP_CODES],
        names=["year", "group", "age12"],
    ).to_frame(index=False)
    data = scaffold.merge(age_counts, on=["year", "group", "age12"], how="left")
    data["deaths"] = data["deaths"].fillna(0)
    data = data.merge(denom[["year", "age12", "lung_cancer_ucd_deaths"]], on=["year", "age12"], how="left")
    rows = []
    for (group, age12), part in data.groupby(["group", "age12"]):
        part = part.sort_values("year")
        if part["deaths"].sum() < 50 or (part["deaths"] > 0).sum() < 10:
            continue
        apc, lcl, ucl, p = fit_poisson_apc(part["year"], part["deaths"], part["lung_cancer_ucd_deaths"])
        first = part[part["year"] == START_YEAR].iloc[0]
        last = part[part["year"] == END_YEAR].iloc[0]
        rows.append(
            {
                "group": group,
                "label": label(group),
                "age12": age12,
                "age12_label": AGE12_LABELS.get(age12, age12),
                "start_deaths": int(first["deaths"]),
                "end_deaths": int(last["deaths"]),
                "start_denominator": int(first["lung_cancer_ucd_deaths"]),
                "end_denominator": int(last["lung_cancer_ucd_deaths"]),
                "start_pct": first["deaths"] / first["lung_cancer_ucd_deaths"] * 100 if first["lung_cancer_ucd_deaths"] else np.nan,
                "end_pct": last["deaths"] / last["lung_cancer_ucd_deaths"] * 100 if last["lung_cancer_ucd_deaths"] else np.nan,
                "change_pct_points": (last["deaths"] / last["lung_cancer_ucd_deaths"] - first["deaths"] / first["lung_cancer_ucd_deaths"]) * 100,
                "apc_pct_per_year": apc,
                "apc_lcl": lcl,
                "apc_ucl": ucl,
                "apc_p_value": p,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["apc_fdr_q_value"] = multipletests(out["apc_p_value"].fillna(1), method="fdr_bh")[1]
    return out


def segmented_poisson(counts: pd.DataFrame) -> pd.DataFrame:
    data = counts[~counts["scope"].isin(EXCLUDE_SCOPES)].copy()
    rows = []
    candidate_knots = list(range(2006, 2020))
    for scope, part in data.groupby("scope"):
        part = part.sort_values("year").copy()
        t = part["year"].astype(float).to_numpy() - START_YEAR
        y = part["deaths"].astype(float).to_numpy()
        n = part["denominator_lung_cancer_ucd_deaths"].astype(float).to_numpy()
        x0 = sm.add_constant(t)
        linear_fit = sm.GLM(y, x0, family=sm.families.Poisson(), offset=np.log(n)).fit()
        best = None
        for knot_year in candidate_knots:
            knot_t = knot_year - START_YEAR
            x = np.column_stack([np.ones_like(t), t, np.maximum(0, t - knot_t)])
            try:
                fit = sm.GLM(y, x, family=sm.families.Poisson(), offset=np.log(n)).fit()
            except Exception:
                continue
            if best is None or fit.aic < best["aic"]:
                best = {"knot_year": knot_year, "fit": fit, "aic": fit.aic}
        if best is None:
            continue
        fit = best["fit"]
        b1 = float(fit.params[1])
        b2 = float(fit.params[2])
        cov = fit.cov_params()
        pre_se = math.sqrt(float(cov[1, 1]))
        post_beta = b1 + b2
        post_se = math.sqrt(float(cov[1, 1] + cov[2, 2] + 2 * cov[1, 2]))
        rows.append(
            {
                "group": scope,
                "label": label(scope),
                "best_knot_year": int(best["knot_year"]),
                "linear_aic": float(linear_fit.aic),
                "segmented_aic": float(best["aic"]),
                "aic_improvement_vs_linear": float(linear_fit.aic - best["aic"]),
                "pre_knot_apc_pct_per_year": (math.exp(b1) - 1.0) * 100,
                "pre_knot_apc_lcl": (math.exp(b1 - 1.96 * pre_se) - 1.0) * 100,
                "pre_knot_apc_ucl": (math.exp(b1 + 1.96 * pre_se) - 1.0) * 100,
                "post_knot_apc_pct_per_year": (math.exp(post_beta) - 1.0) * 100,
                "post_knot_apc_lcl": (math.exp(post_beta - 1.96 * post_se) - 1.0) * 100,
                "post_knot_apc_ucl": (math.exp(post_beta + 1.96 * post_se) - 1.0) * 100,
                "knot_term_p_value": float(fit.pvalues[2]),
                "start_pct": float(part.iloc[0]["proportion_among_lung_cancer_deaths"] * 100),
                "end_pct": float(part.iloc[-1]["proportion_among_lung_cancer_deaths"] * 100),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["knot_term_fdr_q_value"] = multipletests(out["knot_term_p_value"].fillna(1), method="fdr_bh")[1]
    return out.sort_values("aic_improvement_vs_linear", ascending=False)


def fixed_period_apc(counts: pd.DataFrame) -> pd.DataFrame:
    periods = [(1999, 2009), (2010, 2019), (2020, 2024)]
    data = counts[~counts["scope"].isin(EXCLUDE_SCOPES)].copy()
    rows = []
    for scope, part in data.groupby("scope"):
        for start, end in periods:
            p = part[(part["year"] >= start) & (part["year"] <= end)].sort_values("year")
            if p.shape[0] < 3:
                continue
            apc, lcl, ucl, pval = fit_poisson_apc(p["year"], p["deaths"], p["denominator_lung_cancer_ucd_deaths"])
            rows.append(
                {
                    "group": scope,
                    "label": label(scope),
                    "period": f"{start}-{end}",
                    "start_year": start,
                    "end_year": end,
                    "apc_pct_per_year": apc,
                    "apc_lcl": lcl,
                    "apc_ucl": ucl,
                    "apc_p_value": pval,
                    "start_pct": float(p.iloc[0]["proportion_among_lung_cancer_deaths"] * 100),
                    "end_pct": float(p.iloc[-1]["proportion_among_lung_cancer_deaths"] * 100),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["apc_fdr_q_value"] = multipletests(out["apc_p_value"].fillna(1), method="fdr_bh")[1]
    return out


def draw_age_standardized_figure(std: pd.DataFrame, change: pd.DataFrame) -> None:
    top_groups = change.head(6)["group"].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), sharex=True)
    axes = axes.ravel()
    for ax, group in zip(axes, top_groups):
        part = std[std["group"] == group].sort_values("year")
        ax.plot(part["year"], part["crude_pct"], color="#667085", linewidth=1.7, marker="o", markersize=2.4, label="Crude")
        ax.plot(part["year"], part["age_standardized_pct_2024_weights"], color="#2a9d8f", linewidth=2.0, marker="o", markersize=2.4, label="Age-standardized")
        ax.set_title(label(group), fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks([1999, 2004, 2009, 2014, 2019, 2024])
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Crude versus 2024-age-standardized co-mention proportions, 1999-2024", fontsize=14)
    fig.supxlabel("Year")
    fig.supylabel("Co-mentioned deaths (%)")
    fig.tight_layout()
    savefig("Figure13_P1_age_standardized_trends_1999_2024")


def draw_age_apc_heatmap(age_apc: pd.DataFrame) -> None:
    groups_order = (
        age_apc.groupby(["group", "label"], as_index=False)["change_pct_points"].mean()
        .sort_values("change_pct_points", ascending=False)["group"]
        .tolist()
    )
    pivot = age_apc.pivot(index="label", columns="age12_label", values="apc_pct_per_year")
    ordered_labels = [label(g) for g in groups_order if label(g) in pivot.index]
    ordered_cols = [AGE12_LABELS[c] for c in AGE_HEATMAP_CODES if AGE12_LABELS[c] in pivot.columns]
    pivot = pivot.loc[ordered_labels, ordered_cols]
    plt.figure(figsize=(9.8, 6.0))
    ax = plt.gca()
    vmax = np.nanmax(np.abs(pivot.to_numpy()))
    im = ax.imshow(pivot.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=7)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("APC (%/year)")
    ax.set_title("Age-specific annual percent change in co-mentions, 1999-2024")
    savefig("Figure14_P1_age_specific_apc_heatmap_1999_2024")


def draw_chronic_core_network(pairs: pd.DataFrame) -> pd.DataFrame:
    edges = pairs[(pairs["year"] == END_YEAR) & (pairs["edge_class"] == "chronic_chronic")].copy()
    edges = edges[~edges["group_a"].isin(CHRONIC_CORE_EXCLUDE) & ~edges["group_b"].isin(CHRONIC_CORE_EXCLUDE)]
    edges = edges.sort_values(["co_mentioned_deaths", "observed_minus_expected"], ascending=False).head(18)
    plot_edges = edges.head(10).copy()
    graph = nx.Graph()
    for _, row in plot_edges.iterrows():
        graph.add_edge(row["group_a"], row["group_b"], weight=float(row["co_mentioned_deaths"]), lift=float(row["lift_vs_independence"]))
    manual_pos = {
        "copd": (0.0, 0.10),
        "ischemic_heart_disease": (-0.95, 0.58),
        "heart_failure": (0.95, 0.58),
        "diabetes": (-0.85, -0.48),
        "atrial_fibrillation": (0.72, -0.48),
        "cerebrovascular": (-1.75, 0.00),
        "ckd": (1.65, -0.02),
        "ild": (0.00, 1.18),
        "depression_anxiety": (1.68, 0.92),
    }
    pos = {node: manual_pos.get(node, (0.0, 0.0)) for node in graph.nodes()}
    weights = np.array([graph[u][v]["weight"] for u, v in graph.edges()], dtype=float)
    lifts = np.array([graph[u][v]["lift"] for u, v in graph.edges()], dtype=float)
    widths = 0.8 + 5.6 * weights / weights.max()
    colors = plt.cm.viridis((lifts - lifts.min()) / (lifts.max() - lifts.min() + 1e-9))
    node_strength = {node: 0.0 for node in graph.nodes()}
    for u, v, data in graph.edges(data=True):
        node_strength[u] += data["weight"]
        node_strength[v] += data["weight"]
    max_strength = max(node_strength.values()) if node_strength else 1
    node_sizes = [500 + 2400 * node_strength[n] / max_strength for n in graph.nodes()]

    plt.figure(figsize=(9.2, 5.8))
    ax = plt.gca()
    nx.draw_networkx_edges(graph, pos, width=widths, edge_color=colors, alpha=0.68, ax=ax)
    nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color="#cde7d8", edgecolors="#344054", linewidths=1.1, ax=ax)
    offsets = {
        "copd": (0.0, 0.00),
        "ischemic_heart_disease": (0.00, 0.20),
        "heart_failure": (0.10, 0.12),
        "diabetes": (-0.05, -0.16),
        "atrial_fibrillation": (0.12, -0.16),
        "cerebrovascular": (-0.08, -0.14),
        "ckd": (0.05, -0.14),
        "ild": (0.0, 0.14),
        "depression_anxiety": (0.02, 0.14),
    }
    for node, (x, y) in pos.items():
        dx, dy = offsets.get(node, (0.0, 0.0))
        ax.text(
            x + dx,
            y + dy,
            label(node),
            ha="center",
            va="center",
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.2},
            zorder=10,
        )
    ax.set_title("2024 chronic-core co-mention network after excluding terminal/acute nodes")
    ax.set_xlim(-2.15, 2.05)
    ax.set_ylim(-0.92, 0.98)
    ax.axis("off")
    savefig("Figure15_P1_2024_chronic_core_network_1999_2024")
    return edges


def draw_segmented_figure(counts: pd.DataFrame, segmented: pd.DataFrame) -> None:
    groups = ["respiratory_failure", "copd", "atrial_fibrillation", "diabetes", "heart_failure", "ckd"]
    data = counts[counts["scope"].isin(groups)].copy()
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), sharex=True)
    axes = axes.ravel()
    for ax, group in zip(axes, groups):
        part = data[data["scope"] == group].sort_values("year")
        row = segmented[segmented["group"] == group].iloc[0]
        ax.plot(part["year"], part["proportion_among_lung_cancer_deaths"] * 100, color="#2a9d8f", marker="o", markersize=2.6, linewidth=1.8)
        ax.axvline(row["best_knot_year"], color="#d95f02", linestyle="--", linewidth=1)
        ax.set_title(f"{label(group)} | knot {int(row['best_knot_year'])}", fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks([1999, 2004, 2009, 2014, 2019, 2024])
    fig.suptitle("Data-driven single-knot trend screening for selected co-mentions", fontsize=14)
    fig.supxlabel("Year")
    fig.supylabel("Co-mentioned deaths (%)")
    fig.tight_layout()
    savefig("Figure16_P1_segmented_trend_screening_1999_2024")


def write_report(
    age_change: pd.DataFrame,
    age_apc: pd.DataFrame,
    chronic_edges_2024: pd.DataFrame,
    segmented: pd.DataFrame,
    fixed_periods: pd.DataFrame,
) -> None:
    lines = [
        "# P1 1999-2024 High-Impact Supplement",
        "",
        "## What was added",
        "",
        "- Direct age standardization of co-mention proportions using the 2024 lung-cancer-death age distribution.",
        "- Age-specific APC models for major co-mentioned conditions in age strata 45-54 through 85+ years.",
        "- Chronic-core network after excluding terminal/acute nodes: respiratory failure, pneumonia/influenza, and pulmonary embolism.",
        "- Exploratory data-driven single-knot Poisson trend screening plus fixed-period APCs for 1999-2009, 2010-2019, and 2020-2024.",
        "",
        "## Age standardization result",
        "",
    ]
    for _, row in age_change.head(8).iterrows():
        lines.append(
            f"- {row['label']}: crude change {row['crude_change_pct_points']:+.2f} pp; "
            f"age-standardized change {row['age_standardized_change_pct_points']:+.2f} pp; "
            f"retained share {row['age_adjustment_retained_share_pct']:.1f}%."
        )
    lines += [
        "",
        "## Chronic-core 2024 edges",
        "",
    ]
    for _, row in chronic_edges_2024.head(8).iterrows():
        lines.append(
            f"- {row['edge_label']}: {int(row['co_mentioned_deaths']):,} deaths; "
            f"lift {row['lift_vs_independence']:.2f}; phi {row['phi_correlation']:.3f}."
        )
    lines += [
        "",
        "## Segmented trend screening",
        "",
    ]
    for _, row in segmented.head(8).iterrows():
        lines.append(
            f"- {row['label']}: best knot {int(row['best_knot_year'])}; "
            f"pre-knot APC {row['pre_knot_apc_pct_per_year']:+.2f}%/year; "
            f"post-knot APC {row['post_knot_apc_pct_per_year']:+.2f}%/year; "
            f"AIC improvement {row['aic_improvement_vs_linear']:.1f}."
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "Age standardization addresses compositional shifts in the lung-cancer-death population, but these analyses remain based on death-certificate co-mentions. Segmented trend knots are exploratory screening results and should be reported as descriptive inflection evidence, not causal change-points.",
        "",
        "CKD and selected late-emerging conditions show abrupt mid-series increases, so coding, certification, and reporting changes should be discussed as an alternative explanation alongside true morbidity shifts.",
    ]
    (OUT / "P1_high_impact_supplement_1999_2024_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    counts, pairs, denom = read_inputs()
    age_counts = read_age_group_counts()
    age_std, age_change = build_age_standardized(counts, age_counts, denom)
    top_age_groups = age_change.head(8)["group"].tolist()
    age_apc = build_age_specific_apc(age_counts, denom, top_age_groups)
    segmented = segmented_poisson(counts)
    fixed_periods = fixed_period_apc(counts)
    chronic_edges_2024 = draw_chronic_core_network(pairs)

    age_std.to_csv(OUT / "P1_age_standardized_trends_1999_2024.csv", index=False, encoding="utf-8-sig")
    age_change.to_csv(OUT / "P1_age_standardized_change_1999_2024.csv", index=False, encoding="utf-8-sig")
    age_apc.to_csv(OUT / "P1_age_specific_apc_1999_2024.csv", index=False, encoding="utf-8-sig")
    segmented.to_csv(OUT / "P1_segmented_trend_models_1999_2024.csv", index=False, encoding="utf-8-sig")
    fixed_periods.to_csv(OUT / "P1_fixed_period_apc_1999_2024.csv", index=False, encoding="utf-8-sig")
    chronic_edges_2024.to_csv(OUT / "P1_2024_chronic_core_network_edges.csv", index=False, encoding="utf-8-sig")

    draw_age_standardized_figure(age_std, age_change)
    draw_age_apc_heatmap(age_apc)
    draw_segmented_figure(counts, segmented)
    write_report(age_change, age_apc, chronic_edges_2024, segmented, fixed_periods)
    print("Wrote P1 high-impact supplement outputs for 1999-2024.")


if __name__ == "__main__":
    main()
