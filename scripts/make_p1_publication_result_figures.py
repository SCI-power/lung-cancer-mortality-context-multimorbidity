from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "outputs"
FIG = OUT / "figures"
PUBFIG = OUT / "publication_figures"

EXCLUDE_SCOPES = {"all_deaths_in_file", "underlying_cause_C34"}
TERMINAL_ACUTE_NODES = {"respiratory_failure", "pneumonia_influenza", "pulmonary_embolism"}

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

COLORS = {
    "copd": "#1f77b4",
    "respiratory_failure": "#d95f02",
    "ischemic_heart_disease": "#7570b3",
    "pneumonia_influenza": "#e7298a",
    "diabetes": "#66a61e",
    "heart_failure": "#e6ab02",
    "atrial_fibrillation": "#1b9e77",
    "cerebrovascular": "#666666",
    "terminal": "#d95f02",
    "chronic": "#1f77b4",
}


def label(group: str) -> str:
    return LABELS.get(group, group.replace("_", " ").title())


def save_current(name: str) -> None:
    PUBFIG.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "svg", "pdf"]:
        plt.savefig(PUBFIG / f"{name}.{ext}", dpi=400, bbox_inches="tight")
    plt.close()


def panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.12,
        1.08,
        letter,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


def draw_trends(ax: plt.Axes, counts: pd.DataFrame) -> None:
    top_groups = [
        "copd",
        "respiratory_failure",
        "ischemic_heart_disease",
        "pneumonia_influenza",
        "diabetes",
        "heart_failure",
        "atrial_fibrillation",
    ]
    label_y = {
        "copd": 19.35,
        "respiratory_failure": 16.85,
        "ischemic_heart_disease": 6.70,
        "pneumonia_influenza": 6.12,
        "diabetes": 5.55,
        "heart_failure": 4.98,
        "atrial_fibrillation": 4.40,
    }
    label_x_offset = {
        "copd": 0.10,
        "respiratory_failure": 0.10,
        "ischemic_heart_disease": 0.10,
        "pneumonia_influenza": 0.58,
        "diabetes": 0.10,
        "heart_failure": 0.58,
        "atrial_fibrillation": 0.10,
    }
    trend = counts[~counts["scope"].isin(EXCLUDE_SCOPES)].copy()
    for group in top_groups:
        part = trend[trend["scope"] == group].sort_values("year")
        ax.plot(
            part["year"],
            part["proportion_among_lung_cancer_deaths"] * 100,
            marker="o",
            linewidth=1.8,
            markersize=3.5,
            color=COLORS.get(group),
            label=label(group),
        )
        last = part[part["year"] == part["year"].max()]
        if not last.empty:
            ax.text(
                float(last["year"].iloc[0]) + label_x_offset.get(group, 0.10),
                label_y.get(group, float(last["proportion_among_lung_cancer_deaths"].iloc[0]) * 100),
                label(group),
                fontsize=6.8,
                va="center",
                color=COLORS.get(group),
            )
    ax.set_title("A. Temporal shift in co-mentioned conditions", loc="left", fontsize=10, fontweight="bold")
    ax.set_xlabel("Year", fontsize=8)
    ax.set_ylabel("Co-mentioned deaths (%)", fontsize=8)
    ax.set_xlim(2017.8, 2026.25)
    ax.set_ylim(0, 22)
    ax.grid(axis="y", alpha=0.22)
    ax.tick_params(labelsize=8)


def draw_edge_enrichment(ax: plt.Axes, edges: pd.DataFrame) -> None:
    data = edges[
        (edges["year"] == 2024)
        & (edges["co_mentioned_deaths"] >= 1000)
        & (edges["lift_vs_independence"] >= 1.2)
    ].copy()
    data = data.sort_values("observed_minus_expected", ascending=False).head(9)
    data = data.sort_values("observed_minus_expected", ascending=True)
    y = np.arange(data.shape[0])
    colors = np.where(data["edge_class"] == "chronic_chronic", COLORS["chronic"], COLORS["terminal"])
    ax.barh(y, data["observed_minus_expected"], color=colors, alpha=0.82)
    for idx, (_, row) in enumerate(data.iterrows()):
        ax.text(
            row["observed_minus_expected"] + 70,
            idx,
            f"lift {row['lift_vs_independence']:.2f}",
            va="center",
            fontsize=7,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(data["edge_label"], fontsize=7)
    ax.set_xlabel("Observed minus expected co-mentions", fontsize=8)
    ax.set_title("B. Condition pairs exceeding independence expectation", loc="left", fontsize=10, fontweight="bold")
    ax.grid(axis="x", alpha=0.22)
    ax.tick_params(axis="x", labelsize=8)
    ax.text(
        0.0,
        -0.20,
        "Blue: chronic-chronic; orange: terminal/acute involved",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
    )


def draw_chronic_network(ax: plt.Axes, nodes: pd.DataFrame, edges: pd.DataFrame) -> None:
    chronic_edges = edges[
        (edges["year"] == 2024)
        & (edges["edge_class"] == "chronic_chronic")
        & (edges["co_mentioned_deaths"] >= 500)
    ].copy()
    chronic_edges = chronic_edges.sort_values("co_mentioned_deaths", ascending=False).head(14)
    node_counts = nodes[nodes["year"] == 2024].set_index("group")["node_deaths"].to_dict()
    graph = nx.Graph()
    for _, row in chronic_edges.iterrows():
        graph.add_edge(row["group_a"], row["group_b"], weight=float(row["co_mentioned_deaths"]))

    pos = {
        "copd": (0.0, 0.0),
        "ischemic_heart_disease": (-0.92, -0.18),
        "heart_failure": (-0.68, 0.68),
        "diabetes": (-0.25, -0.92),
        "atrial_fibrillation": (0.78, -0.55),
        "cerebrovascular": (0.96, 0.36),
        "ckd": (1.52, 0.03),
        "ild": (0.28, 0.86),
        "depression_anxiety": (0.28, 1.18),
    }
    pos = {node: pos.get(node, (0.0, 0.0)) for node in graph.nodes()}
    weights = np.array([graph[u][v]["weight"] for u, v in graph.edges()], dtype=float)
    max_weight = weights.max() if len(weights) else 1.0
    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        width=0.8 + 5.2 * weights / max_weight,
        edge_color="#667085",
        alpha=0.58,
    )
    max_node = max(node_counts.values())
    node_sizes = [360 + 2200 * node_counts.get(node, 1) / max_node for node in graph.nodes()]
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_size=node_sizes,
        node_color="#d9ead3",
        edgecolors="#22543d",
        linewidths=1.0,
    )
    for node, (x, y_pos) in pos.items():
        ax.text(
            x,
            y_pos,
            label(node),
            ha="center",
            va="center",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0},
        )
    ax.set_title("C. Chronic-condition core after excluding terminal/acute nodes", loc="left", fontsize=10, fontweight="bold")
    ax.set_axis_off()
    ax.set_xlim(-1.25, 1.75)
    ax.set_ylim(-1.18, 1.35)


def draw_sex_forest(ax: plt.Axes, contrasts: pd.DataFrame) -> None:
    wanted = [
        "ischemic_heart_disease",
        "diabetes",
        "atrial_fibrillation",
        "pneumonia_influenza",
        "heart_failure",
        "respiratory_failure",
        "copd",
    ]
    data = contrasts[
        (contrasts["contrast"] == "Male vs female")
        & (contrasts["reporting_gate"])
        & (contrasts["group"].isin(wanted))
    ].copy()
    data["order"] = data["group"].map({g: i for i, g in enumerate(wanted)})
    data = data.sort_values("order", ascending=False)
    y = np.arange(data.shape[0])
    ax.errorbar(
        data["prevalence_ratio"],
        y,
        xerr=[
            data["prevalence_ratio"] - data["prevalence_ratio_lcl"],
            data["prevalence_ratio_ucl"] - data["prevalence_ratio"],
        ],
        fmt="o",
        color="#1f77b4",
        ecolor="#98a2b3",
        elinewidth=1.3,
        capsize=2.5,
        markersize=4,
    )
    ax.axvline(1.0, color="#667085", linestyle="--", linewidth=1)
    for idx, (_, row) in enumerate(data.iterrows()):
        ax.text(
            row["prevalence_ratio_ucl"] + 0.035,
            idx,
            f"{row['target_proportion_pct']:.1f}% vs {row['comparator_proportion_pct']:.1f}%",
            va="center",
            fontsize=7,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([label(g) for g in data["group"]], fontsize=7)
    ax.set_xlabel("Prevalence ratio, male vs female", fontsize=8)
    ax.set_title("D. Sex contrasts in 2024 co-mentions", loc="left", fontsize=10, fontweight="bold")
    ax.grid(axis="x", alpha=0.22)
    ax.set_xlim(0.75, 2.15)
    ax.tick_params(axis="x", labelsize=8)


def make_main_figure(counts: pd.DataFrame, nodes: pd.DataFrame, edges: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(13.6, 9.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], height_ratios=[1.0, 1.02], wspace=0.43, hspace=0.38)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    draw_trends(ax_a, counts)
    draw_edge_enrichment(ax_b, edges)
    draw_chronic_network(ax_c, nodes, edges)
    draw_sex_forest(ax_d, contrasts)
    fig.suptitle(
        "Multimorbidity network among US lung cancer deaths, 2018-2024",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.01,
        0.01,
        "Source: NCHS Mortality Multiple Cause public-use files. Cohort: deaths with underlying cause ICD-10 C34.",
        fontsize=7,
    )
    save_current("P1_MainFigure_multimorbidity_network_results_v1")


def make_edge_bar_figure(edges: pd.DataFrame) -> None:
    plt.figure(figsize=(8.2, 5.6))
    ax = plt.gca()
    draw_edge_enrichment(ax, edges)
    save_current("P1_ResultFigure_edge_enrichment_top_pairs_v1")


def write_figure_legend_file() -> None:
    lines = [
        "# P1 Result Figure Package v1",
        "",
        "## Main Figure",
        "",
        "`P1_MainFigure_multimorbidity_network_results_v1.png/svg/pdf`",
        "",
        "Panel A shows annual proportions of selected record-axis co-mentioned conditions among deaths with lung cancer as the underlying cause. Panel B shows the leading 2024 condition pairs exceeding expected co-mentions under independence, with labels showing observed-to-expected lift. Panel C shows the chronic-condition core network after excluding respiratory failure, pneumonia/influenza, and pulmonary embolism. Panel D shows 2024 male-versus-female prevalence ratios for selected co-mentions, with percentages shown as male vs female.",
        "",
        "## Single Figures Already Available",
        "",
        "- `Figure1_P1_comorbidity_trends_2018_2024.png/svg`",
        "- `Figure2_P1_2024_comorbidity_network.png/svg`",
        "- `Figure3_P1_change_2018_2024.png/svg`",
        "- `Figure4_P1_2024_sex_stratified_heatmap.png/svg`",
        "- `Figure5_P1_2024_age_stratified_heatmap.png/svg`",
        "- `Figure6_P1_2024_excess_comention_network.png/svg`",
        "- `Figure7_P1_2024_chronic_core_network_sensitivity.png/svg`",
        "- `Figure8_P1_2024_male_vs_female_prevalence_ratio_forest.png/svg`",
        "- `SuppFigS1_P1_2024_race_ethnicity_stratified_heatmap.png/svg`",
        "",
        "## Suggested Main-Text Use",
        "",
        "Use the four-panel main figure as the primary results figure. Keep Figure 1, Figure 6, Figure 7, and Figure 8 as editable source figures. Move the heatmaps and race/Hispanic origin figure to supplementary materials unless the target journal asks for subgroup emphasis.",
    ]
    (PUBFIG / "P1_result_figure_legends_v1.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    PUBFIG.mkdir(parents=True, exist_ok=True)
    counts = pd.read_csv(OUT / "P1_mcod_2018_2024_comorbidity_counts_long.csv")
    nodes = pd.read_csv(OUT / "P1_node_table_2018_2024.csv")
    edges = pd.read_csv(OUT / "P1_edge_enrichment_2018_2024.csv")
    contrasts = pd.read_csv(OUT / "P1_2024_stratified_contrast_table.csv")
    make_main_figure(counts, nodes, edges, contrasts)
    make_edge_bar_figure(edges)
    write_figure_legend_file()
    print(f"Wrote publication result figures to {PUBFIG}")


if __name__ == "__main__":
    main()
