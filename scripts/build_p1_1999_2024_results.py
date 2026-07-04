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
FIG = OUT / "figures"

START_YEAR = 1999
END_YEAR = 2024
TAG = "1999_2024"

EXCLUDE_SCOPES = {
    "all_deaths_in_file",
    "all_deaths_us_residents",
    "foreign_resident_deaths_excluded",
    "underlying_cause_C34",
    "underlying_cause_C34_no_record_axis_U071",
    "covid_u071",
}

TERMINAL_ACUTE_NODES = {
    "respiratory_failure",
    "pneumonia_influenza",
    "pulmonary_embolism",
}

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


def label(scope: str) -> str:
    return LABELS.get(scope, scope.replace("_", " ").title())


def savefig(name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "svg"]:
        plt.savefig(FIG / f"{name}.{ext}", dpi=400, bbox_inches="tight")
    plt.close()


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = pd.read_csv(OUT / f"P1_mcod_{TAG}_comorbidity_counts_long.csv")
    pairs = pd.read_csv(OUT / f"P1_mcod_{TAG}_pair_counts_long.csv")
    for col in ["year", "deaths", "denominator_lung_cancer_ucd_deaths", "proportion_among_lung_cancer_deaths"]:
        if col in counts.columns:
            counts[col] = pd.to_numeric(counts[col], errors="coerce")
    for col in ["year", "co_mentioned_deaths", "denominator_lung_cancer_ucd_deaths", "proportion_among_lung_cancer_deaths"]:
        if col in pairs.columns:
            pairs[col] = pd.to_numeric(pairs[col], errors="coerce")
    return counts, pairs


def build_node_table(counts: pd.DataFrame) -> pd.DataFrame:
    nodes = counts[~counts["scope"].isin(EXCLUDE_SCOPES)].copy()
    nodes = nodes.rename(
        columns={
            "scope": "group",
            "deaths": "node_deaths",
            "denominator_lung_cancer_ucd_deaths": "denominator",
            "proportion_among_lung_cancer_deaths": "node_proportion",
        }
    )
    nodes["label"] = nodes["group"].map(label)
    nodes["node_proportion_pct"] = nodes["node_proportion"] * 100
    nodes["node_class"] = np.where(nodes["group"].isin(TERMINAL_ACUTE_NODES), "terminal_or_acute_pathway", "chronic_or_preexisting")
    nodes = nodes.sort_values(["year", "node_deaths"], ascending=[True, False])
    return nodes


def build_edge_enrichment(pairs: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    lookup = nodes[["year", "group", "node_deaths", "node_proportion"]].copy()
    edges = pairs.merge(
        lookup.rename(columns={"group": "group_a", "node_deaths": "node_a_deaths", "node_proportion": "node_a_proportion"}),
        on=["year", "group_a"],
        how="left",
    ).merge(
        lookup.rename(columns={"group": "group_b", "node_deaths": "node_b_deaths", "node_proportion": "node_b_proportion"}),
        on=["year", "group_b"],
        how="left",
    )
    edges = edges.rename(columns={"denominator_lung_cancer_ucd_deaths": "denominator"})
    n = edges["denominator"].astype(float)
    a = edges["node_a_deaths"].astype(float)
    b = edges["node_b_deaths"].astype(float)
    c = edges["co_mentioned_deaths"].astype(float)
    expected = (a * b) / n
    edges["expected_co_mentions_if_independent"] = expected
    edges["observed_minus_expected"] = c - expected
    edges["lift_vs_independence"] = np.where(expected > 0, c / expected, np.nan)
    edges["jaccard_index"] = np.where((a + b - c) > 0, c / (a + b - c), np.nan)

    n10 = a - c
    n01 = b - c
    n00 = n - a - b + c
    denom = np.sqrt(a * (n - a) * b * (n - b))
    edges["phi_correlation"] = np.where(denom > 0, ((c * n00) - (n10 * n01)) / denom, np.nan)
    edges["edge_class"] = np.where(
        edges["group_a"].isin(TERMINAL_ACUTE_NODES) | edges["group_b"].isin(TERMINAL_ACUTE_NODES),
        "involves_terminal_or_acute_pathway",
        "chronic_chronic",
    )
    edges["label_a"] = edges["group_a"].map(label)
    edges["label_b"] = edges["group_b"].map(label)
    edges["edge_label"] = edges["label_a"] + " + " + edges["label_b"]
    return edges.sort_values(["year", "co_mentioned_deaths"], ascending=[True, False])


def compute_network_metrics(nodes: pd.DataFrame, edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict] = []
    chronic_edges = []
    terminal_rows = []
    for year, nodes_y in nodes.groupby("year"):
        edges_y = edges[edges["year"] == year].copy()
        denominator = float(nodes_y["denominator"].iloc[0])
        core_threshold = max(250.0, denominator * 0.005)
        core_edges = edges_y[edges_y["co_mentioned_deaths"] >= core_threshold].copy()
        chronic_core = core_edges[core_edges["edge_class"] == "chronic_chronic"].copy()
        chronic_edges.append(chronic_core)

        graph = nx.Graph()
        for _, node in nodes_y.iterrows():
            graph.add_node(node["group"])
        for _, edge in core_edges.iterrows():
            graph.add_edge(
                edge["group_a"],
                edge["group_b"],
                weight=float(edge["co_mentioned_deaths"]),
                distance=1.0 / math.log1p(float(edge["co_mentioned_deaths"])),
            )

        degree = nx.degree_centrality(graph) if graph.number_of_nodes() else {}
        between = nx.betweenness_centrality(graph, weight="distance") if graph.number_of_nodes() else {}
        try:
            eigen = nx.eigenvector_centrality_numpy(graph, weight="weight") if graph.number_of_edges() else {}
        except Exception:
            eigen = {}

        top20 = edges_y.sort_values("co_mentioned_deaths", ascending=False).head(20)
        total_top20 = float(top20["co_mentioned_deaths"].sum())
        terminal_top20 = top20[top20["edge_class"] == "involves_terminal_or_acute_pathway"]
        terminal_rows.append(
            {
                "year": int(year),
                "top20_edges_total_co_mentions": int(total_top20),
                "top20_edges_involving_terminal_or_acute_nodes": int(terminal_top20.shape[0]),
                "top20_terminal_or_acute_edge_co_mentions": int(terminal_top20["co_mentioned_deaths"].sum()),
                "top20_terminal_or_acute_share_pct": terminal_top20["co_mentioned_deaths"].sum() / total_top20 * 100 if total_top20 else np.nan,
            }
        )

        for _, node in nodes_y.iterrows():
            group = node["group"]
            all_edges = edges_y[(edges_y["group_a"] == group) | (edges_y["group_b"] == group)]
            core_edges_node = core_edges[(core_edges["group_a"] == group) | (core_edges["group_b"] == group)]
            metric_rows.append(
                {
                    "year": int(year),
                    "group": group,
                    "label": label(group),
                    "node_class": node["node_class"],
                    "node_deaths": int(node["node_deaths"]),
                    "node_proportion_pct": float(node["node_proportion_pct"]),
                    "weighted_strength_all_edges": int(all_edges["co_mentioned_deaths"].sum()),
                    "weighted_strength_all_edges_pct_of_lung_cancer_deaths": float(all_edges["co_mentioned_deaths"].sum() / denominator * 100),
                    "core_edge_threshold_deaths": int(core_threshold),
                    "core_degree": int(core_edges_node.shape[0]),
                    "core_weighted_strength": int(core_edges_node["co_mentioned_deaths"].sum()),
                    "degree_centrality_core": degree.get(group, 0.0),
                    "betweenness_centrality_core": between.get(group, 0.0),
                    "eigenvector_centrality_core": eigen.get(group, 0.0),
                    "mean_lift_all_edges": all_edges["lift_vs_independence"].replace([np.inf, -np.inf], np.nan).mean(),
                    "max_lift_all_edges": all_edges["lift_vs_independence"].replace([np.inf, -np.inf], np.nan).max(),
                }
            )
    chronic_edges_df = pd.concat(chronic_edges, ignore_index=True) if chronic_edges else pd.DataFrame()
    return pd.DataFrame(metric_rows), chronic_edges_df, pd.DataFrame(terminal_rows)


def fit_scope_trend(data: pd.DataFrame, scope: str) -> dict:
    part = data[data["scope"] == scope].sort_values("year").copy()
    part["year_centered"] = part["year"] - part["year"].min()
    y = part["deaths"].astype(float).to_numpy()
    denom = part["denominator_lung_cancer_ucd_deaths"].astype(float).to_numpy()
    x = sm.add_constant(part["year_centered"].astype(float).to_numpy())

    poisson = sm.GLM(y, x, family=sm.families.Poisson(), offset=np.log(denom))
    poisson_fit = poisson.fit(cov_type="HC0")
    beta = float(poisson_fit.params[1])
    se = float(poisson_fit.bse[1])
    apc = (np.exp(beta) - 1.0) * 100
    apc_lcl = (np.exp(beta - 1.96 * se) - 1.0) * 100
    apc_ucl = (np.exp(beta + 1.96 * se) - 1.0) * 100

    prop_pct = y / denom * 100
    wls = sm.WLS(prop_pct, x, weights=denom)
    wls_fit = wls.fit(cov_type="HC0")
    slope = float(wls_fit.params[1])
    slope_se = float(wls_fit.bse[1])

    first = part.iloc[0]
    last = part.iloc[-1]
    return {
        "analysis_set": "main_all_lung_cancer_ucd_deaths",
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
        "relative_change_pct": float((last["deaths"] / last["denominator_lung_cancer_ucd_deaths"]) / (first["deaths"] / first["denominator_lung_cancer_ucd_deaths"]) * 100 - 100),
        "apc_pct_per_year": apc,
        "apc_lcl": apc_lcl,
        "apc_ucl": apc_ucl,
        "apc_p_value": float(poisson_fit.pvalues[1]),
        "absolute_slope_pct_points_per_year": slope,
        "absolute_slope_lcl": slope - 1.96 * slope_se,
        "absolute_slope_ucl": slope + 1.96 * slope_se,
        "absolute_slope_p_value": float(wls_fit.pvalues[1]),
    }


def build_trend_models(counts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = counts[~counts["scope"].isin(EXCLUDE_SCOPES)].copy()
    data = data[["year", "scope", "deaths", "denominator_lung_cancer_ucd_deaths"]].copy()
    rows = []
    for scope in sorted(data["scope"].unique()):
        try:
            rows.append(fit_scope_trend(data, scope))
        except Exception as exc:
            rows.append({"analysis_set": "main_all_lung_cancer_ucd_deaths", "scope": scope, "label": label(scope), "model_error": str(exc)})
    trends = pd.DataFrame(rows)
    p = pd.to_numeric(trends["apc_p_value"], errors="coerce").fillna(1)
    trends["apc_fdr_q_value"] = multipletests(p, method="fdr_bh")[1]
    p2 = pd.to_numeric(trends["absolute_slope_p_value"], errors="coerce").fillna(1)
    trends["absolute_slope_fdr_q_value"] = multipletests(p2, method="fdr_bh")[1]
    data["analysis_set"] = "main_all_lung_cancer_ucd_deaths"
    return data, trends


def draw_trend_figure(counts: pd.DataFrame) -> None:
    trend = counts[~counts["scope"].isin(EXCLUDE_SCOPES)].copy()
    top_2024 = (
        trend[trend["year"] == END_YEAR]
        .sort_values("proportion_among_lung_cancer_deaths", ascending=False)
        .head(10)["scope"]
        .tolist()
    )
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_2024)))
    plt.figure(figsize=(10.6, 6.3))
    ax = plt.gca()
    for color, scope in zip(colors, top_2024):
        part = trend[trend["scope"] == scope].sort_values("year")
        ax.plot(
            part["year"],
            part["proportion_among_lung_cancer_deaths"] * 100,
            marker="o",
            markersize=3.0,
            linewidth=1.9,
            color=color,
            label=label(scope),
        )
    ax.set_xlabel("Year")
    ax.set_ylabel("Deaths with co-mentioned condition (%)")
    ax.set_title("Co-mentioned conditions among lung cancer underlying-cause deaths, 1999-2024")
    ax.set_xticks([1999, 2004, 2009, 2014, 2019, 2024])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    savefig("Figure1_P1_comorbidity_trends_1999_2024")


def draw_network_figure(counts: pd.DataFrame, pairs: pd.DataFrame) -> None:
    nodes_2024 = counts[(counts["year"] == END_YEAR) & (~counts["scope"].isin(EXCLUDE_SCOPES))][["scope", "deaths"]]
    node_counts = dict(zip(nodes_2024["scope"], nodes_2024["deaths"]))
    edges = pairs[pairs["year"] == END_YEAR].sort_values("co_mentioned_deaths", ascending=False).head(20)

    graph = nx.Graph()
    for _, row in edges.iterrows():
        graph.add_edge(row["group_a"], row["group_b"], weight=float(row["co_mentioned_deaths"]))
    pos = nx.spring_layout(graph, seed=20260621, weight="weight", k=0.85)
    weights = [graph[u][v]["weight"] for u, v in graph.edges()]
    max_weight = max(weights) if weights else 1
    max_node = max(node_counts.values()) if node_counts else 1
    node_sizes = [450 + 2200 * node_counts.get(node, 0) / max_node for node in graph.nodes()]
    edge_widths = [0.8 + 5.5 * w / max_weight for w in weights]
    node_colors = ["#f7d488" if n in TERMINAL_ACUTE_NODES else "#b7d7ef" for n in graph.nodes()]

    plt.figure(figsize=(10.4, 8.0))
    ax = plt.gca()
    nx.draw_networkx_edges(graph, pos, width=edge_widths, alpha=0.55, edge_color="#667085")
    nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color=node_colors, edgecolors="#344054", linewidths=1.1)
    offsets = {
        "copd": (0.00, 0.00),
        "respiratory_failure": (0.00, -0.05),
        "pneumonia_influenza": (0.00, -0.08),
        "pulmonary_embolism": (-0.02, -0.07),
        "cerebrovascular": (-0.08, 0.00),
        "diabetes": (-0.05, 0.08),
        "ischemic_heart_disease": (0.10, 0.08),
        "heart_failure": (0.10, 0.00),
        "atrial_fibrillation": (0.12, -0.03),
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
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
            zorder=10,
        )
    xs = [xy[0] for xy in pos.values()]
    ys = [xy[1] for xy in pos.values()]
    x_pad = (max(xs) - min(xs)) * 0.38 if xs else 1
    y_pad = (max(ys) - min(ys)) * 0.25 if ys else 1
    ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
    ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)
    plt.title("Top co-mentioned condition pairs in 2024, resident scope")
    plt.axis("off")
    savefig("Figure2_P1_2024_comorbidity_network_1999_2024_resident")


def draw_change_figure(trends: pd.DataFrame) -> None:
    df = trends.dropna(subset=["absolute_change_pct_points"]).copy()
    df = df.sort_values("absolute_change_pct_points", ascending=True)
    colors = np.where(df["absolute_change_pct_points"] >= 0, "#2a9d8f", "#667085")
    plt.figure(figsize=(8.7, 6.3))
    ax = plt.gca()
    ax.barh(df["label"], df["absolute_change_pct_points"], color=colors)
    ax.axvline(0, color="#475467", linewidth=1)
    ax.set_xlabel("Absolute change, percentage points")
    ax.set_title("Change in co-mentioned conditions, 1999 to 2024")
    ax.grid(axis="x", alpha=0.23)
    savefig("Figure3_P1_change_1999_2024")


def draw_apc_figure(trends: pd.DataFrame) -> None:
    df = trends.dropna(subset=["apc_pct_per_year"]).sort_values("absolute_change_pct_points", ascending=True).copy()
    colors = np.where(df["apc_fdr_q_value"] < 0.05, "#2a9d8f", "#98a2b3")
    y = np.arange(df.shape[0])
    plt.figure(figsize=(8.4, 6.0))
    ax = plt.gca()
    ax.errorbar(
        df["apc_pct_per_year"],
        y,
        xerr=[df["apc_pct_per_year"] - df["apc_lcl"], df["apc_ucl"] - df["apc_pct_per_year"]],
        fmt="none",
        ecolor="#98a2b3",
        elinewidth=1.0,
        capsize=2,
        zorder=1,
    )
    ax.scatter(df["apc_pct_per_year"], y, c=colors, s=30, zorder=2)
    ax.axvline(0, color="#667085", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"], fontsize=8)
    ax.set_xlabel("Annual percent change in co-mention proportion (%)")
    ax.set_title("Formal trend models for co-mentioned conditions, 1999-2024")
    ax.grid(axis="x", alpha=0.22)
    savefig("Figure9_P1_formal_trend_models_apc_1999_2024")


def write_tables_and_report(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    metrics: pd.DataFrame,
    chronic_edges: pd.DataFrame,
    terminal_summary: pd.DataFrame,
    model_input: pd.DataFrame,
    trends: pd.DataFrame,
) -> None:
    nodes.to_csv(OUT / "P1_node_table_1999_2024.csv", index=False, encoding="utf-8-sig")
    edges.to_csv(OUT / "P1_edge_enrichment_1999_2024.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUT / "P1_network_node_metrics_1999_2024.csv", index=False, encoding="utf-8-sig")
    chronic_edges.to_csv(OUT / "P1_chronic_core_edges_1999_2024.csv", index=False, encoding="utf-8-sig")
    terminal_summary.to_csv(OUT / "P1_terminal_pathway_sensitivity_summary_1999_2024.csv", index=False, encoding="utf-8-sig")
    model_input.to_csv(OUT / "P1_trend_model_input_counts_1999_2024.csv", index=False, encoding="utf-8-sig")
    trends.to_csv(OUT / "P1_formal_trend_model_results_1999_2024.csv", index=False, encoding="utf-8-sig")

    table1 = (
        nodes[nodes["year"] == END_YEAR][["group", "label", "node_deaths", "denominator", "node_proportion_pct", "node_class"]]
        .merge(
            trends[["scope", "absolute_change_pct_points", "relative_change_pct", "apc_pct_per_year", "apc_lcl", "apc_ucl", "apc_fdr_q_value"]],
            left_on="group",
            right_on="scope",
            how="left",
        )
        .drop(columns=["scope"])
        .sort_values("node_deaths", ascending=False)
    )
    table2 = edges[edges["year"] == END_YEAR].sort_values("co_mentioned_deaths", ascending=False).head(30)
    table1.to_csv(OUT / "Table1_P1_2024_node_burden_and_1999_2024_change.csv", index=False, encoding="utf-8-sig")
    table2.to_csv(OUT / "Table2_P1_2024_edge_enrichment_top30_1999_2024.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(OUT / "P1_expanded_1999_2024_results_tables_v1.xlsx", engine="openpyxl") as writer:
        table1.to_excel(writer, sheet_name="node_burden_change", index=False)
        table2.to_excel(writer, sheet_name="top_2024_edges", index=False)
        trends.to_excel(writer, sheet_name="trend_models", index=False)
        terminal_summary.to_excel(writer, sheet_name="terminal_sensitivity", index=False)
        metrics[metrics["year"].isin([START_YEAR, END_YEAR])].to_excel(writer, sheet_name="network_metrics_1999_2024", index=False)

    top_change = trends.sort_values("absolute_change_pct_points", ascending=False).head(8)
    top_edges = table2.head(8)
    lines = [
        "# P1 1999-2024 Expanded Resident-Scope Results",
        "",
        "## Data status",
        "",
        f"- Years analyzed: {START_YEAR}-{END_YEAR}.",
        "- Scope: U.S. resident deaths; foreign residents excluded by resident-status code 4.",
        "- Lung cancer underlying cause: ICD-10 C34; comorbid conditions from record-axis multiple causes.",
        "- Source: NCHS Mortality Multiple Cause public-use files.",
        "",
        "## Main 2024 node burden",
        "",
    ]
    for _, row in table1.head(10).iterrows():
        lines.append(
            f"- {row['label']}: {int(row['node_deaths']):,} deaths "
            f"({row['node_proportion_pct']:.2f}%)."
        )
    lines += [
        "",
        "## Largest 1999-2024 increases",
        "",
    ]
    for _, row in top_change.iterrows():
        lines.append(
            f"- {row['label']}: {row['absolute_change_pct_points']:+.2f} percentage points; "
            f"APC {row['apc_pct_per_year']:+.2f}%/year "
            f"({row['apc_lcl']:+.2f} to {row['apc_ucl']:+.2f}), FDR q={row['apc_fdr_q_value']:.3g}."
        )
    lines += [
        "",
        "## Top 2024 co-mentioned pairs",
        "",
    ]
    for _, row in top_edges.iterrows():
        lines.append(
            f"- {row['edge_label']}: {int(row['co_mentioned_deaths']):,} deaths "
            f"({row['proportion_among_lung_cancer_deaths'] * 100:.2f}%), "
            f"lift {row['lift_vs_independence']:.2f}."
        )
    lines += [
        "",
        "## Figure outputs",
        "",
        "- Figure1_P1_comorbidity_trends_1999_2024",
        "- Figure2_P1_2024_comorbidity_network_1999_2024_resident",
        "- Figure3_P1_change_1999_2024",
        "- Figure9_P1_formal_trend_models_apc_1999_2024",
        "",
        "## Interpretation boundary",
        "",
        "These are death-certificate co-mention patterns among deaths with lung cancer as the underlying cause. They should not be interpreted as clinical prevalence, incidence, or causal effects.",
    ]
    (OUT / "P1_expanded_1999_2024_results_report_v1.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    counts, pairs = load_inputs()
    nodes = build_node_table(counts)
    edges = build_edge_enrichment(pairs, nodes)
    metrics, chronic_edges, terminal_summary = compute_network_metrics(nodes, edges)
    model_input, trends = build_trend_models(counts)
    draw_trend_figure(counts)
    draw_network_figure(counts, pairs)
    draw_change_figure(trends)
    draw_apc_figure(trends)
    write_tables_and_report(nodes, edges, metrics, chronic_edges, terminal_summary, model_input, trends)
    print("Wrote P1 1999-2024 expanded result tables, models, report, and figures.")


if __name__ == "__main__":
    main()
