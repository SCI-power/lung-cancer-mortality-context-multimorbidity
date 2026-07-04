from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "outputs"
FIG = OUT / "figures"

EXCLUDE_SCOPES = {"all_deaths_in_file", "underlying_cause_C34"}

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


def label(name: str) -> str:
    return LABELS.get(name, name.replace("_", " ").title())


def savefig(name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "svg"]:
        plt.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close()


def draw_offset_labels(ax: plt.Axes, pos: dict, graph: nx.Graph, offsets: dict[str, tuple[float, float]]) -> None:
    for node in graph.nodes():
        x, y = pos[node]
        dx, dy = offsets.get(node, (0.0, 0.0))
        ax.text(
            x + dx,
            y + dy,
            label(node),
            ha="center",
            va="center",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
            zorder=10,
        )


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    counts = pd.read_csv(OUT / "P1_mcod_2018_2024_comorbidity_counts_long.csv")
    pairs = pd.read_csv(OUT / "P1_mcod_2018_2024_pair_counts_long.csv")
    stratified = pd.read_csv(OUT / "P1_mcod_2018_2024_stratified_comorbidity_counts.csv")

    for col in ["year", "deaths", "denominator_lung_cancer_ucd_deaths", "proportion_among_lung_cancer_deaths"]:
        if col in counts.columns:
            counts[col] = pd.to_numeric(counts[col], errors="coerce")
    for col in ["year", "co_mentioned_deaths", "denominator_lung_cancer_ucd_deaths", "proportion_among_lung_cancer_deaths"]:
        if col in pairs.columns:
            pairs[col] = pd.to_numeric(pairs[col], errors="coerce")
    for col in ["year", "deaths", "lung_cancer_ucd_deaths_in_stratum", "proportion_among_stratum_lung_cancer_deaths"]:
        if col in stratified.columns:
            stratified[col] = pd.to_numeric(stratified[col], errors="coerce")

    return counts, pairs, stratified


def node_table(counts: pd.DataFrame) -> pd.DataFrame:
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
    return nodes


def enrich_edges(pairs: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    node_lookup = nodes[["year", "group", "node_deaths", "node_proportion"]].copy()
    edges = pairs.merge(
        node_lookup.rename(columns={"group": "group_a", "node_deaths": "node_a_deaths", "node_proportion": "node_a_proportion"}),
        on=["year", "group_a"],
        how="left",
    ).merge(
        node_lookup.rename(columns={"group": "group_b", "node_deaths": "node_b_deaths", "node_proportion": "node_b_proportion"}),
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
    return edges


def compute_network_node_metrics(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for year, nodes_y in nodes.groupby("year"):
        edges_y = edges[edges["year"] == year].copy()
        denominator = float(nodes_y["denominator"].iloc[0])
        core_threshold = max(250.0, denominator * 0.005)
        core_edges = edges_y[edges_y["co_mentioned_deaths"] >= core_threshold].copy()

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

        degree_centrality = nx.degree_centrality(graph) if graph.number_of_nodes() else {}
        betweenness = nx.betweenness_centrality(graph, weight="distance") if graph.number_of_nodes() else {}
        try:
            eigenvector = nx.eigenvector_centrality_numpy(graph, weight="weight") if graph.number_of_edges() else {}
        except Exception:
            eigenvector = {}

        for _, node in nodes_y.iterrows():
            group = node["group"]
            all_edges = edges_y[(edges_y["group_a"] == group) | (edges_y["group_b"] == group)]
            core_edges_node = core_edges[(core_edges["group_a"] == group) | (core_edges["group_b"] == group)]
            rows.append(
                {
                    "year": int(year),
                    "group": group,
                    "label": label(group),
                    "node_class": node["node_class"],
                    "node_deaths": int(node["node_deaths"]),
                    "node_proportion_pct": node["node_proportion"] * 100,
                    "weighted_strength_all_edges": int(all_edges["co_mentioned_deaths"].sum()),
                    "weighted_strength_all_edges_pct_of_lung_cancer_deaths": all_edges["co_mentioned_deaths"].sum() / denominator * 100,
                    "core_edge_threshold_deaths": int(core_threshold),
                    "core_degree": int(core_edges_node.shape[0]),
                    "core_weighted_strength": int(core_edges_node["co_mentioned_deaths"].sum()),
                    "degree_centrality_core": degree_centrality.get(group, 0.0),
                    "betweenness_centrality_core": betweenness.get(group, 0.0),
                    "eigenvector_centrality_core": eigenvector.get(group, 0.0),
                    "mean_lift_all_edges": all_edges["lift_vs_independence"].replace([np.inf, -np.inf], np.nan).mean(),
                    "max_lift_all_edges": all_edges["lift_vs_independence"].replace([np.inf, -np.inf], np.nan).max(),
                }
            )
    return pd.DataFrame(rows)


def build_terminal_sensitivity_summary(edges: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for year, edges_y in edges.groupby("year"):
        top20 = edges_y.sort_values("co_mentioned_deaths", ascending=False).head(20)
        total_top20 = float(top20["co_mentioned_deaths"].sum())
        terminal_top20 = top20[top20["edge_class"] == "involves_terminal_or_acute_pathway"]
        chronic_top20 = top20[top20["edge_class"] == "chronic_chronic"]
        rows.append(
            {
                "year": int(year),
                "top20_edges_total_co_mentions": int(total_top20),
                "top20_edges_involving_terminal_or_acute_nodes": int(terminal_top20.shape[0]),
                "top20_terminal_or_acute_edge_co_mentions": int(terminal_top20["co_mentioned_deaths"].sum()),
                "top20_terminal_or_acute_share_pct": terminal_top20["co_mentioned_deaths"].sum() / total_top20 * 100 if total_top20 else np.nan,
                "top20_chronic_chronic_edges": int(chronic_top20.shape[0]),
                "top20_chronic_chronic_edge_co_mentions": int(chronic_top20["co_mentioned_deaths"].sum()),
                "top20_chronic_chronic_share_pct": chronic_top20["co_mentioned_deaths"].sum() / total_top20 * 100 if total_top20 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def log_rr_ci(a: float, n1: float, b: float, n0: float) -> tuple[float, float, float]:
    p1 = a / n1 if n1 else np.nan
    p0 = b / n0 if n0 else np.nan
    if a <= 0 or b <= 0 or n1 <= 0 or n0 <= 0 or p0 <= 0:
        return np.nan, np.nan, np.nan
    rr = p1 / p0
    se = math.sqrt(max(0.0, (1.0 / a) - (1.0 / n1) + (1.0 / b) - (1.0 / n0)))
    return rr, math.exp(math.log(rr) - 1.96 * se), math.exp(math.log(rr) + 1.96 * se)


def build_stratified_contrasts(stratified: pd.DataFrame) -> pd.DataFrame:
    contrast_specs = [
        ("sex", "Male", "Female", "Male vs female"),
        ("age12", "45-54 years", "65-74 years", "45-54 vs 65-74"),
        ("age12", "75-84 years", "65-74 years", "75-84 vs 65-74"),
        ("age12", "85+ years", "65-74 years", "85+ vs 65-74"),
        ("race_ethnicity_2022plus", "Non-Hispanic Black only", "Non-Hispanic White only", "Black vs White"),
        ("race_ethnicity_2022plus", "Non-Hispanic Asian only", "Non-Hispanic White only", "Asian vs White"),
        ("race_ethnicity_2022plus", "Mexican", "Non-Hispanic White only", "Mexican vs White"),
    ]

    data = stratified[stratified["year"] == 2024].copy()
    index = {
        (row["dimension"], row["stratum_label"], row["group"]): row
        for _, row in data.iterrows()
    }
    rows: list[dict] = []
    groups = sorted(data["group"].dropna().unique())
    for dimension, target, comparator, contrast_label in contrast_specs:
        for group in groups:
            target_row = index.get((dimension, target, group))
            comparator_row = index.get((dimension, comparator, group))
            if target_row is None or comparator_row is None:
                continue
            a = float(target_row["deaths"])
            n1 = float(target_row["lung_cancer_ucd_deaths_in_stratum"])
            b = float(comparator_row["deaths"])
            n0 = float(comparator_row["lung_cancer_ucd_deaths_in_stratum"])
            rr, lcl, ucl = log_rr_ci(a, n1, b, n0)
            p1 = a / n1 if n1 else np.nan
            p0 = b / n0 if n0 else np.nan
            rows.append(
                {
                    "year": 2024,
                    "dimension": dimension,
                    "contrast": contrast_label,
                    "target_stratum": target,
                    "comparator_stratum": comparator,
                    "group": group,
                    "label": label(group),
                    "target_deaths": int(a),
                    "target_denominator": int(n1),
                    "target_proportion_pct": p1 * 100,
                    "comparator_deaths": int(b),
                    "comparator_denominator": int(n0),
                    "comparator_proportion_pct": p0 * 100,
                    "prevalence_ratio": rr,
                    "prevalence_ratio_lcl": lcl,
                    "prevalence_ratio_ucl": ucl,
                    "absolute_difference_pct_points": (p1 - p0) * 100,
                    "reporting_gate": bool(a >= 10 and b >= 10 and n1 >= 500 and n0 >= 500),
                }
            )
    return pd.DataFrame(rows)


def draw_edge_enrichment_network(edges: pd.DataFrame, nodes: pd.DataFrame) -> None:
    edges_2024 = edges[
        (edges["year"] == 2024)
        & (edges["co_mentioned_deaths"] >= 500)
        & (edges["lift_vs_independence"] >= 1.2)
    ].copy()
    edges_2024 = edges_2024.sort_values(["observed_minus_expected", "co_mentioned_deaths"], ascending=False).head(20)
    nodes_2024 = nodes[nodes["year"] == 2024].set_index("group")

    graph = nx.Graph()
    for _, edge in edges_2024.iterrows():
        graph.add_edge(edge["group_a"], edge["group_b"], weight=float(edge["co_mentioned_deaths"]), lift=float(edge["lift_vs_independence"]))
    if graph.number_of_edges() == 0:
        return

    pos = nx.kamada_kawai_layout(graph, weight="weight")
    weights = np.array([graph[u][v]["weight"] for u, v in graph.edges()], dtype=float)
    lifts = np.array([graph[u][v]["lift"] for u, v in graph.edges()], dtype=float)
    lift_cap = np.nanpercentile(lifts, 95) if len(lifts) else 1.0
    lift_norm = np.clip(lifts, 1.2, lift_cap)
    edge_colors = plt.cm.viridis((lift_norm - lift_norm.min()) / (lift_norm.max() - lift_norm.min() + 1e-9))
    edge_widths = 0.8 + 5.5 * weights / weights.max()
    node_sizes = []
    node_colors = []
    for node in graph.nodes():
        count = float(nodes_2024.loc[node, "node_deaths"]) if node in nodes_2024.index else 0
        node_sizes.append(400 + 2400 * count / nodes_2024["node_deaths"].max())
        node_colors.append("#f7d488" if node in TERMINAL_ACUTE_NODES else "#b7d7ef")

    plt.figure(figsize=(10.8, 8.0))
    ax = plt.gca()
    nx.draw_networkx_edges(graph, pos, width=edge_widths, edge_color=edge_colors, alpha=0.68, ax=ax)
    nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color=node_colors, edgecolors="#344054", linewidths=1.0, ax=ax)
    offsets = {
        "copd": (0.00, 0.07),
        "respiratory_failure": (-0.02, 0.08),
        "pneumonia_influenza": (-0.02, -0.07),
        "pulmonary_embolism": (-0.05, 0.03),
        "heart_failure": (0.08, 0.02),
        "ischemic_heart_disease": (0.00, -0.08),
        "atrial_fibrillation": (-0.07, -0.04),
        "diabetes": (0.07, -0.03),
        "cerebrovascular": (0.00, -0.08),
        "ckd": (0.00, 0.08),
    }
    draw_offset_labels(ax, pos, graph, offsets)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=float(lift_norm.min()), vmax=float(lift_norm.max())))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.036, pad=0.02)
    cbar.set_label("Observed / expected co-mentions")
    ax.set_title("2024 excess co-mention network among lung cancer deaths")
    ax.axis("off")
    savefig("Figure6_P1_2024_excess_comention_network")


def draw_chronic_core_network(edges: pd.DataFrame, nodes: pd.DataFrame) -> None:
    chronic = edges[
        (edges["year"] == 2024)
        & (edges["edge_class"] == "chronic_chronic")
        & (edges["co_mentioned_deaths"] >= 500)
    ].sort_values("co_mentioned_deaths", ascending=False).head(20)
    nodes_2024 = nodes[nodes["year"] == 2024].set_index("group")
    graph = nx.Graph()
    for _, edge in chronic.iterrows():
        graph.add_edge(edge["group_a"], edge["group_b"], weight=float(edge["co_mentioned_deaths"]))
    if graph.number_of_edges() == 0:
        return

    pos = nx.kamada_kawai_layout(graph, weight="weight")
    weights = np.array([graph[u][v]["weight"] for u, v in graph.edges()], dtype=float)
    node_sizes = [
        450 + 2600 * float(nodes_2024.loc[node, "node_deaths"]) / nodes_2024["node_deaths"].max()
        if node in nodes_2024.index
        else 450
        for node in graph.nodes()
    ]

    plt.figure(figsize=(10.4, 7.4))
    ax = plt.gca()
    nx.draw_networkx_edges(graph, pos, width=0.8 + 5.5 * weights / weights.max(), edge_color="#667085", alpha=0.62, ax=ax)
    nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color="#d9ead3", edgecolors="#22543d", linewidths=1.0, ax=ax)
    offsets = {
        "copd": (0.00, 0.08),
        "ischemic_heart_disease": (0.00, -0.08),
        "atrial_fibrillation": (-0.06, -0.06),
        "diabetes": (0.07, -0.04),
        "heart_failure": (0.08, 0.04),
        "cerebrovascular": (-0.02, 0.08),
        "ckd": (0.00, 0.08),
    }
    draw_offset_labels(ax, pos, graph, offsets)
    ax.set_title("2024 chronic-condition core network after excluding terminal/acute nodes")
    ax.axis("off")
    savefig("Figure7_P1_2024_chronic_core_network_sensitivity")


def draw_sex_forest(contrasts: pd.DataFrame) -> None:
    sex = contrasts[
        (contrasts["contrast"] == "Male vs female")
        & (contrasts["reporting_gate"])
    ].copy()
    top_groups = [
        "ischemic_heart_disease",
        "diabetes",
        "heart_failure",
        "atrial_fibrillation",
        "respiratory_failure",
        "copd",
        "pneumonia_influenza",
        "cerebrovascular",
    ]
    sex = sex[sex["group"].isin(top_groups)].copy()
    sex["order"] = sex["group"].map({g: i for i, g in enumerate(top_groups)})
    sex = sex.sort_values("order", ascending=False)
    if sex.empty:
        return

    y = np.arange(sex.shape[0])
    plt.figure(figsize=(7.4, 4.8))
    ax = plt.gca()
    ax.errorbar(
        sex["prevalence_ratio"],
        y,
        xerr=[
            sex["prevalence_ratio"] - sex["prevalence_ratio_lcl"],
            sex["prevalence_ratio_ucl"] - sex["prevalence_ratio"],
        ],
        fmt="o",
        color="#1f77b4",
        ecolor="#98a2b3",
        capsize=3,
    )
    ax.axvline(1.0, color="#667085", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([label(g) for g in sex["group"]], fontsize=8)
    ax.set_xlabel("Prevalence ratio: male vs female")
    ax.set_title("2024 sex contrast in death-certificate co-mentions")
    ax.grid(axis="x", alpha=0.25)
    savefig("Figure8_P1_2024_male_vs_female_prevalence_ratio_forest")


def write_excel_bundle(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    node_metrics: pd.DataFrame,
    chronic_nodes: pd.DataFrame,
    chronic_edges: pd.DataFrame,
    terminal_summary: pd.DataFrame,
    stratified_contrasts: pd.DataFrame,
) -> None:
    xlsx_path = OUT / "P1_enhanced_results_tables_v1.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        nodes[(nodes["year"] == 2024)].sort_values("node_deaths", ascending=False).to_excel(writer, sheet_name="2024_nodes", index=False)
        nodes.pivot(index="group", columns="year", values="node_proportion_pct").reset_index().to_excel(writer, sheet_name="node_trends_pct", index=False)
        edges[(edges["year"] == 2024)].sort_values("co_mentioned_deaths", ascending=False).head(50).to_excel(writer, sheet_name="2024_top_edges", index=False)
        edges[(edges["year"] == 2024)].sort_values("lift_vs_independence", ascending=False).head(50).to_excel(writer, sheet_name="2024_top_lift_edges", index=False)
        node_metrics.to_excel(writer, sheet_name="network_node_metrics", index=False)
        chronic_nodes.to_excel(writer, sheet_name="chronic_node_trends", index=False)
        chronic_edges.to_excel(writer, sheet_name="chronic_core_edges", index=False)
        terminal_summary.to_excel(writer, sheet_name="terminal_sensitivity", index=False)
        stratified_contrasts.to_excel(writer, sheet_name="2024_stratified_contrasts", index=False)


def write_submission_tables(nodes: pd.DataFrame, edges: pd.DataFrame, stratified_contrasts: pd.DataFrame) -> None:
    n2018 = nodes[nodes["year"] == 2018][["group", "node_deaths", "node_proportion_pct"]].rename(
        columns={"node_deaths": "deaths_2018", "node_proportion_pct": "pct_2018"}
    )
    n2024 = nodes[nodes["year"] == 2024][["group", "label", "node_class", "node_deaths", "node_proportion_pct"]].rename(
        columns={"node_deaths": "deaths_2024", "node_proportion_pct": "pct_2024"}
    )
    table1 = n2024.merge(n2018, on="group", how="left")
    table1["absolute_change_2018_to_2024_pct_points"] = table1["pct_2024"] - table1["pct_2018"]
    table1["relative_change_2018_to_2024_pct"] = np.where(
        table1["pct_2018"] > 0,
        (table1["pct_2024"] / table1["pct_2018"] - 1.0) * 100,
        np.nan,
    )
    table1 = table1.sort_values("deaths_2024", ascending=False)
    table1.to_csv(OUT / "Table1_P1_2024_node_burden_and_2018_2024_change.csv", index=False, encoding="utf-8-sig")

    table2_cols = [
        "edge_label",
        "edge_class",
        "co_mentioned_deaths",
        "proportion_among_lung_cancer_deaths",
        "expected_co_mentions_if_independent",
        "observed_minus_expected",
        "lift_vs_independence",
        "jaccard_index",
        "phi_correlation",
    ]
    table2 = (
        edges[(edges["year"] == 2024) & (edges["co_mentioned_deaths"] >= 500)]
        .sort_values("co_mentioned_deaths", ascending=False)
        .head(30)[table2_cols]
        .copy()
    )
    table2["proportion_among_lung_cancer_deaths_pct"] = table2["proportion_among_lung_cancer_deaths"] * 100
    table2.to_csv(OUT / "Table2_P1_2024_edge_enrichment_top30.csv", index=False, encoding="utf-8-sig")

    table3_cols = [
        "dimension",
        "contrast",
        "label",
        "target_proportion_pct",
        "comparator_proportion_pct",
        "absolute_difference_pct_points",
        "prevalence_ratio",
        "prevalence_ratio_lcl",
        "prevalence_ratio_ucl",
        "target_deaths",
        "target_denominator",
        "comparator_deaths",
        "comparator_denominator",
    ]
    table3 = stratified_contrasts[stratified_contrasts["reporting_gate"]].copy()
    table3["abs_difference_for_sort"] = table3["absolute_difference_pct_points"].abs()
    table3 = table3.sort_values(["dimension", "contrast", "abs_difference_for_sort"], ascending=[True, True, False])
    table3[table3_cols].to_csv(OUT / "Table3_P1_2024_stratified_contrasts.csv", index=False, encoding="utf-8-sig")


def fmt_pct(value: float) -> str:
    return f"{value:.2f}%"


def write_report(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    node_metrics: pd.DataFrame,
    chronic_edges: pd.DataFrame,
    terminal_summary: pd.DataFrame,
    stratified_contrasts: pd.DataFrame,
) -> None:
    nodes_2024 = nodes[nodes["year"] == 2024].sort_values("node_deaths", ascending=False)
    edge_2024 = edges[edges["year"] == 2024].copy()
    stable_excess = edge_2024[(edge_2024["co_mentioned_deaths"] >= 250) & (edge_2024["lift_vs_independence"] >= 1.2)]
    stable_excess = stable_excess.sort_values(["observed_minus_expected", "co_mentioned_deaths"], ascending=False).head(8)
    core_nodes = node_metrics[node_metrics["year"] == 2024].sort_values("core_weighted_strength", ascending=False).head(8)
    chronic_2024 = chronic_edges[chronic_edges["year"] == 2024].sort_values("co_mentioned_deaths", ascending=False).head(8)
    terminal_row_2018 = terminal_summary[terminal_summary["year"] == 2018].iloc[0]
    terminal_row_2024 = terminal_summary[terminal_summary["year"] == 2024].iloc[0]
    sex_rr = stratified_contrasts[
        (stratified_contrasts["contrast"] == "Male vs female")
        & (stratified_contrasts["reporting_gate"])
    ].sort_values("absolute_difference_pct_points", ascending=False).head(6)

    lines = [
        "# P1 Enhanced Results Report v1",
        "",
        "## Added analyses",
        "",
        "- Edge enrichment: observed co-mentioned deaths compared with expected co-mentions under independence.",
        "- Network node metrics: count-weighted strength, core degree, centrality, and temporal change.",
        "- Sensitivity analysis: chronic-condition core network after excluding respiratory failure, pneumonia/influenza, and pulmonary embolism.",
        "- 2024 stratified contrasts: prevalence ratios and absolute percentage-point differences for sex, age, and 2022+ race/Hispanic origin strata.",
        "",
        "## 2024 node burden",
        "",
    ]
    for _, row in nodes_2024.head(10).iterrows():
        lines.append(f"- {row['label']}: {int(row['node_deaths']):,} deaths ({fmt_pct(row['node_proportion_pct'])}).")

    lines += [
        "",
        "## 2024 excess co-mention edges",
        "",
        "These edges remain prominent after accounting for the marginal frequency of each node.",
        "",
    ]
    for _, row in stable_excess.iterrows():
        lines.append(
            f"- {row['edge_label']}: observed {int(row['co_mentioned_deaths']):,}; "
            f"expected {row['expected_co_mentions_if_independent']:.0f}; "
            f"lift {row['lift_vs_independence']:.2f}; phi {row['phi_correlation']:.3f}."
        )

    lines += [
        "",
        "## 2024 central network nodes",
        "",
    ]
    for _, row in core_nodes.iterrows():
        lines.append(
            f"- {row['label']}: core degree {int(row['core_degree'])}, "
            f"core weighted strength {int(row['core_weighted_strength']):,}, "
            f"betweenness {row['betweenness_centrality_core']:.3f}."
        )

    lines += [
        "",
        "## Chronic-network sensitivity",
        "",
        "After excluding respiratory failure, pneumonia/influenza, and pulmonary embolism, the leading 2024 chronic-condition pairs were:",
        "",
    ]
    for _, row in chronic_2024.iterrows():
        lines.append(
            f"- {row['edge_label']}: {int(row['co_mentioned_deaths']):,} deaths "
            f"({row['proportion_among_lung_cancer_deaths'] * 100:.2f}%), lift {row['lift_vs_independence']:.2f}."
        )
    lines += [
        "",
        (
            "Among the top 20 edges by observed count, the share involving terminal/acute nodes changed from "
            f"{terminal_row_2018['top20_terminal_or_acute_share_pct']:.1f}% in 2018 to "
            f"{terminal_row_2024['top20_terminal_or_acute_share_pct']:.1f}% in 2024."
        ),
        "",
        "## 2024 sex contrasts",
        "",
    ]
    for _, row in sex_rr.iterrows():
        lines.append(
            f"- {row['label']}: male {row['target_proportion_pct']:.2f}% vs female "
            f"{row['comparator_proportion_pct']:.2f}%; PR {row['prevalence_ratio']:.2f} "
            f"({row['prevalence_ratio_lcl']:.2f}-{row['prevalence_ratio_ucl']:.2f}); "
            f"difference {row['absolute_difference_pct_points']:+.2f} percentage points."
        )

    lines += [
        "",
        "## Manuscript upgrade",
        "",
        "The strongest upgraded framing is no longer only a frequency table. It is a mortality-record multimorbidity network that separates terminal/acute pathway edges from a chronic cardiopulmonary-metabolic core and quantifies which co-mentions exceed independence expectations.",
        "",
        "## New files",
        "",
        "- `P1_edge_enrichment_2018_2024.csv`",
        "- `P1_network_node_metrics_2018_2024.csv`",
        "- `P1_chronic_node_trends_excluding_terminal_pathways.csv`",
        "- `P1_chronic_core_edges_2018_2024.csv`",
        "- `P1_terminal_pathway_sensitivity_summary.csv`",
        "- `P1_2024_stratified_contrast_table.csv`",
        "- `P1_enhanced_results_tables_v1.xlsx`",
        "- `Figure6_P1_2024_excess_comention_network.png/svg`",
        "- `Figure7_P1_2024_chronic_core_network_sensitivity.png/svg`",
        "- `Figure8_P1_2024_male_vs_female_prevalence_ratio_forest.png/svg`",
    ]
    (OUT / "P1_enhanced_results_report_v1.md").write_text("\n".join(lines), encoding="utf-8")


def write_manuscript_insert() -> None:
    lines = [
        "# P1 Manuscript Enhancement Insert v1",
        "",
        "## Additional Methods Text",
        "",
        "For network enrichment analyses, the expected number of deaths with each co-mentioned condition pair was calculated under an independence assumption using the product of the two marginal node counts divided by the annual lung cancer underlying-cause death denominator. We reported observed-to-expected ratios (lift), observed-minus-expected counts, Jaccard indices, and phi correlations for condition pairs. To reduce interpretive dependence on terminal death pathways, we performed a sensitivity analysis that excluded respiratory failure, pneumonia/influenza, and pulmonary embolism from the chronic-condition core network. Network node metrics included weighted strength across all edges and centrality measures calculated in a core network defined by annual edges with at least 0.5% of lung cancer deaths or at least 250 co-mentioned deaths, whichever was larger.",
        "",
        "For stratified contrasts, 2024 stratum-specific proportions were compared using prevalence ratios and absolute percentage-point differences. Wald confidence intervals for prevalence ratios were calculated on the log scale. Race/Hispanic origin contrasts were restricted to 2024 and interpreted as death-certificate co-mention contrasts rather than population incidence or survival disparities.",
        "",
        "## Additional Results Text",
        "",
        "Edge enrichment analyses showed that several high-volume co-mentioned pairs remained prominent after accounting for marginal node prevalence. The excess co-mention network distinguished terminal/acute pathway edges, especially respiratory failure-related combinations, from a chronic cardiopulmonary-metabolic core. In the chronic-core sensitivity analysis excluding respiratory failure, pneumonia/influenza, and pulmonary embolism, COPD with ischemic heart disease, COPD with heart failure, COPD with diabetes, atrial fibrillation with COPD, and diabetes with ischemic heart disease remained the leading co-mentioned pairs in 2024.",
        "",
        "Network node metrics identified COPD as the dominant chronic hub by weighted strength, with cardiometabolic nodes forming a secondary cluster involving ischemic heart disease, diabetes, heart failure, and atrial fibrillation. These findings support a two-layer interpretation: terminal/acute events describe proximate mortality pathways, whereas the chronic-core network describes clinically relevant multimorbidity that may affect screening, treatment tolerance, survivorship care, and end-of-life trajectories.",
        "",
        "## Recommended Figure/Table Mapping",
        "",
        "- Main Figure 1: annual node trends.",
        "- Main Figure 2: 2024 excess co-mention network, replacing the simple count-only network if journal space is limited.",
        "- Main Figure 3: chronic-core sensitivity network.",
        "- Main Figure 4: sex or age stratified heatmap/forest depending on target journal.",
        "- Table 1: annual lung cancer death denominators and 2024 node burden.",
        "- Table 2: 2024 edge enrichment metrics.",
        "- Table 3: 2024 stratified contrasts.",
        "- Supplement: full node metrics, all-year edge enrichment, race/Hispanic origin heatmap, and parser/codebook details.",
    ]
    (OUT / "P1_manuscript_enhancement_insert_v1.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    counts, pairs, stratified = read_inputs()
    nodes = node_table(counts)
    edges = enrich_edges(pairs, nodes)
    node_metrics = compute_network_node_metrics(nodes, edges)

    chronic_nodes = nodes[~nodes["group"].isin(TERMINAL_ACUTE_NODES)].copy()
    chronic_edges = edges[edges["edge_class"] == "chronic_chronic"].copy()
    terminal_summary = build_terminal_sensitivity_summary(edges)
    stratified_contrasts = build_stratified_contrasts(stratified)

    nodes.to_csv(OUT / "P1_node_table_2018_2024.csv", index=False, encoding="utf-8-sig")
    edges.to_csv(OUT / "P1_edge_enrichment_2018_2024.csv", index=False, encoding="utf-8-sig")
    node_metrics.to_csv(OUT / "P1_network_node_metrics_2018_2024.csv", index=False, encoding="utf-8-sig")
    chronic_nodes.to_csv(OUT / "P1_chronic_node_trends_excluding_terminal_pathways.csv", index=False, encoding="utf-8-sig")
    chronic_edges.to_csv(OUT / "P1_chronic_core_edges_2018_2024.csv", index=False, encoding="utf-8-sig")
    terminal_summary.to_csv(OUT / "P1_terminal_pathway_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    stratified_contrasts.to_csv(OUT / "P1_2024_stratified_contrast_table.csv", index=False, encoding="utf-8-sig")

    draw_edge_enrichment_network(edges, nodes)
    draw_chronic_core_network(edges, nodes)
    draw_sex_forest(stratified_contrasts)

    write_excel_bundle(nodes, edges, node_metrics, chronic_nodes, chronic_edges, terminal_summary, stratified_contrasts)
    write_submission_tables(nodes, edges, stratified_contrasts)
    write_report(nodes, edges, node_metrics, chronic_edges, terminal_summary, stratified_contrasts)
    write_manuscript_insert()
    print("Wrote P1 enhanced network, sensitivity, contrast, figure, and manuscript insert outputs.")


if __name__ == "__main__":
    main()
