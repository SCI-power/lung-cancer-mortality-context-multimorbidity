from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "outputs"
WONDER = OUT / "wonder_exports"
FIG = OUT / "figures"

NODE_ORDER = [
    "copd",
    "respiratory_failure",
    "ischemic_heart_disease",
    "heart_failure",
    "diabetes",
    "atrial_fibrillation",
]

NODE_LABELS = {
    "lung_cancer_total": "Lung cancer total",
    "copd": "COPD",
    "respiratory_failure": "Respiratory failure",
    "ischemic_heart_disease": "Ischemic heart disease",
    "heart_failure": "Heart failure",
    "diabetes": "Diabetes",
    "atrial_fibrillation": "Atrial fibrillation",
}

REGION_ORDER = [
    "Northeast",
    "Midwest",
    "South",
    "West",
]

URBAN_ORDER = [
    "Large Central Metro",
    "Large Fringe Metro",
    "Medium Metro",
    "Small Metro",
    "Micropolitan (Nonmetro)",
    "NonCore (Nonmetro)",
]

REGION_COLORS = {
    "Northeast": "#4e79a7",
    "Midwest": "#59a14f",
    "South": "#d95f02",
    "West": "#7570b3",
}

URBAN_COLORS = {
    "Large Central Metro": "#4e79a7",
    "Large Fringe Metro": "#76b7b2",
    "Medium Metro": "#59a14f",
    "Small Metro": "#edc948",
    "Micropolitan (Nonmetro)": "#f28e2b",
    "NonCore (Nonmetro)": "#e15759",
}


def read_wonder_table(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    header = rows[0]
    data_rows = []
    for row in rows[1:]:
        if len(row) != len(header):
            continue
        row = [cell.strip() for cell in row]
        if len(row) < 3 or not row[1].isdigit():
            continue
        data_rows.append(row)
    df = pd.DataFrame(data_rows, columns=header)
    for col in df.columns:
        if col in {"Notes", "Census Region", "Census Region Code", "2013 Urbanization", "2013 Urbanization Code"}:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_region(value: str) -> str:
    if not isinstance(value, str):
        return value
    return value.replace("Census Region 1: ", "").replace("Census Region 2: ", "").replace(
        "Census Region 3: ", ""
    ).replace("Census Region 4: ", "")


def collect_family(prefix: str, stratum_col: str, stratum_code_col: str) -> pd.DataFrame:
    frames = []
    for node in ["lung_cancer_total", *NODE_ORDER]:
        path = WONDER / f"{prefix}_{node}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = read_wonder_table(path)
        df["node"] = node
        df["node_label"] = NODE_LABELS[node]
        df["stratum"] = df[stratum_col].astype("string")
        df["stratum_code"] = df[stratum_code_col].astype("string")
        if stratum_col == "Census Region":
            df["stratum"] = df["stratum"].map(clean_region)
        df["stratum_type"] = "census_region" if stratum_col == "Census Region" else "urbanization_2013"
        df["is_total_row"] = df["Notes"].eq("Total")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out.rename(
        columns={
            "Year": "year",
            "Deaths": "deaths",
            "Population": "population",
            "Crude Rate": "crude_rate_per_100k",
            "Crude Rate Lower 95% Confidence Interval": "crude_rate_lcl",
            "Crude Rate Upper 95% Confidence Interval": "crude_rate_ucl",
            "Crude Rate Standard Error": "crude_rate_se",
            "Age Adjusted Rate": "age_adjusted_rate_per_100k",
            "Age Adjusted Rate Lower 95% Confidence Interval": "age_adjusted_rate_lcl",
            "Age Adjusted Rate Upper 95% Confidence Interval": "age_adjusted_rate_ucl",
            "Age Adjusted Rate Standard Error": "age_adjusted_rate_se",
        }
    )
    out["year"] = out["year"].astype(int)
    keep_cols = [
        "year",
        "node",
        "node_label",
        "stratum_type",
        "stratum",
        "stratum_code",
        "is_total_row",
        "deaths",
        "population",
        "crude_rate_per_100k",
        "crude_rate_lcl",
        "crude_rate_ucl",
        "crude_rate_se",
        "age_adjusted_rate_per_100k",
        "age_adjusted_rate_lcl",
        "age_adjusted_rate_ucl",
        "age_adjusted_rate_se",
    ]
    return out[[col for col in keep_cols if col in out.columns]]


def add_lung_denominators(df: pd.DataFrame) -> pd.DataFrame:
    denom = df[df["node"].eq("lung_cancer_total")][
        ["year", "stratum_type", "stratum", "is_total_row", "deaths"]
    ].rename(columns={"deaths": "lung_cancer_deaths_wonder"})
    merged = df.merge(denom, on=["year", "stratum_type", "stratum", "is_total_row"], how="left")
    merged["comention_pct_among_lung_cancer_deaths_wonder"] = np.where(
        merged["node"].ne("lung_cancer_total"),
        merged["deaths"] / merged["lung_cancer_deaths_wonder"] * 100.0,
        np.nan,
    )
    return merged


def order_strata(df: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    out = df.copy()
    out["stratum"] = pd.Categorical(out["stratum"], categories=order, ordered=True)
    out["node_label"] = pd.Categorical(
        out["node_label"], categories=[NODE_LABELS[node] for node in NODE_ORDER], ordered=True
    )
    return out


def annotate_heatmap(ax: plt.Axes, matrix: pd.DataFrame, fmt: str = ".1f") -> None:
    values = matrix.to_numpy(dtype=float)
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            val = values[y, x]
            if np.isfinite(val):
                ax.text(x, y, format(val, fmt), ha="center", va="center", fontsize=8, color="#111111")


def save_figure(name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "svg", "pdf"]:
        plt.savefig(FIG / f"{name}.{ext}", dpi=400, bbox_inches="tight")
    plt.close()


def panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.10, 1.08, letter, transform=ax.transAxes, fontsize=13, fontweight="bold")


def plot_region(region: pd.DataFrame) -> None:
    region_nt = order_strata(region[~region["is_total_row"]].copy(), REGION_ORDER)
    fig = plt.figure(figsize=(13.5, 8.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 1.0], wspace=0.34, hspace=0.48)

    ax1 = fig.add_subplot(gs[0, 0])
    lc = region_nt[region_nt["node"].eq("lung_cancer_total")]
    for region_name in REGION_ORDER:
        part = lc[lc["stratum"].astype(str).eq(region_name)].sort_values("year")
        ax1.plot(
            part["year"],
            part["age_adjusted_rate_per_100k"],
            marker="o",
            linewidth=2.0,
            markersize=4.0,
            color=REGION_COLORS[region_name],
            label=region_name,
        )
    ax1.set_title("Age-adjusted lung cancer mortality rate by Census Region", fontsize=11, loc="left")
    ax1.set_ylabel("Deaths per 100,000 population")
    ax1.set_xlabel("Year")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(frameon=False, ncol=2, fontsize=8)
    panel_letter(ax1, "A")

    ax2 = fig.add_subplot(gs[0, 1])
    data_2024 = region_nt[(region_nt["year"].eq(2024)) & (region_nt["node"].isin(NODE_ORDER))]
    heat = data_2024.pivot_table(
        index="node_label",
        columns="stratum",
        values="comention_pct_among_lung_cancer_deaths_wonder",
        observed=False,
    )
    heat = heat.reindex([NODE_LABELS[node] for node in NODE_ORDER], columns=REGION_ORDER)
    im = ax2.imshow(heat, cmap="YlGnBu", aspect="auto")
    annotate_heatmap(ax2, heat)
    ax2.set_xticks(range(len(REGION_ORDER)))
    ax2.set_xticklabels(REGION_ORDER, rotation=30, ha="right")
    ax2.set_yticks(range(len(heat.index)))
    ax2.set_yticklabels(heat.index)
    ax2.set_title("2024 co-mention proportion among lung cancer deaths", fontsize=11, loc="left")
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("%")
    panel_letter(ax2, "B")

    ax3 = fig.add_subplot(gs[1, 0])
    change = region_nt[region_nt["node"].isin(NODE_ORDER)].pivot_table(
        index=["node_label", "stratum"],
        columns="year",
        values="comention_pct_among_lung_cancer_deaths_wonder",
        observed=False,
    )
    change.columns = [int(col) for col in change.columns]
    change["delta_2018_2024_pp"] = change[2024] - change[2018]
    delta = change["delta_2018_2024_pp"].reset_index()
    delta["node_label"] = pd.Categorical(delta["node_label"], [NODE_LABELS[node] for node in NODE_ORDER], ordered=True)
    delta["stratum"] = pd.Categorical(delta["stratum"], REGION_ORDER, ordered=True)
    delta = delta.sort_values(["node_label", "stratum"])
    x = np.arange(len(NODE_ORDER))
    width = 0.18
    for i, region_name in enumerate(REGION_ORDER):
        part = delta[delta["stratum"].astype(str).eq(region_name)]
        ax3.bar(
            x + (i - 1.5) * width,
            part["delta_2018_2024_pp"],
            width=width,
            color=REGION_COLORS[region_name],
            label=region_name,
        )
    ax3.axhline(0, color="#333333", linewidth=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels([NODE_LABELS[node] for node in NODE_ORDER], rotation=30, ha="right")
    ax3.set_ylabel("Percentage-point change")
    ax3.set_title("Change in co-mention proportion, 2018 to 2024", fontsize=11, loc="left")
    ax3.grid(axis="y", alpha=0.25)
    panel_letter(ax3, "C")

    ax4 = fig.add_subplot(gs[1, 1])
    ranges = (
        data_2024.groupby("node_label", observed=False)["comention_pct_among_lung_cancer_deaths_wonder"]
        .agg(["min", "max"])
        .reset_index()
    )
    ranges["range_pp"] = ranges["max"] - ranges["min"]
    ranges = ranges.sort_values("range_pp", ascending=True)
    ax4.barh(ranges["node_label"].astype(str), ranges["range_pp"], color="#6b7280")
    ax4.set_xlabel("Max-min regional difference, percentage points")
    ax4.set_title("2024 regional heterogeneity", fontsize=11, loc="left")
    ax4.grid(axis="x", alpha=0.25)
    panel_letter(ax4, "D")

    fig.suptitle("CDC WONDER validation: regional lung cancer mortality and comorbidity co-mentions", fontsize=14)
    save_figure("Figure11_P1_wonder_region_validation")


def plot_urban(urban: pd.DataFrame) -> None:
    urban_nt = order_strata(urban[~urban["is_total_row"]].copy(), URBAN_ORDER)
    fig = plt.figure(figsize=(13.5, 8.1))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.12, 1.0], height_ratios=[1.0, 1.0], wspace=0.34, hspace=0.46)

    ax1 = fig.add_subplot(gs[0, 0])
    lc = urban_nt[urban_nt["node"].eq("lung_cancer_total")].dropna(subset=["crude_rate_per_100k"])
    rate_years = sorted(int(year) for year in lc["year"].dropna().unique())
    for urban_name in URBAN_ORDER:
        part = lc[lc["stratum"].astype(str).eq(urban_name)].sort_values("year")
        ax1.plot(
            part["year"],
            part["crude_rate_per_100k"],
            marker="o",
            linewidth=1.8,
            markersize=3.8,
            color=URBAN_COLORS[urban_name],
            label=urban_name,
        )
    ax1.set_title(
        f"Crude lung cancer mortality rate by urbanization category ({rate_years[0]}-{rate_years[-1]})",
        fontsize=11,
        loc="left",
    )
    ax1.set_ylabel("Deaths per 100,000 population")
    ax1.set_xlabel("Year")
    ax1.set_xticks(rate_years)
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(frameon=False, fontsize=7.5, ncol=2)
    panel_letter(ax1, "A")

    ax2 = fig.add_subplot(gs[0, 1])
    data_2024 = urban_nt[(urban_nt["year"].eq(2024)) & (urban_nt["node"].isin(NODE_ORDER))]
    heat = data_2024.pivot_table(
        index="node_label",
        columns="stratum",
        values="comention_pct_among_lung_cancer_deaths_wonder",
        observed=False,
    )
    heat = heat.reindex([NODE_LABELS[node] for node in NODE_ORDER], columns=URBAN_ORDER)
    im = ax2.imshow(heat, cmap="YlOrRd", aspect="auto")
    annotate_heatmap(ax2, heat)
    ax2.set_xticks(range(len(URBAN_ORDER)))
    ax2.set_xticklabels(URBAN_ORDER, rotation=35, ha="right")
    ax2.set_yticks(range(len(heat.index)))
    ax2.set_yticklabels(heat.index)
    ax2.set_title("2024 co-mention proportion among lung cancer deaths", fontsize=11, loc="left")
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("%")
    panel_letter(ax2, "B")

    ax3 = fig.add_subplot(gs[1, 0])
    ratio_rows = []
    for node in NODE_ORDER:
        part = data_2024[data_2024["node"].eq(node)].set_index("stratum")
        central = float(part.loc["Large Central Metro", "comention_pct_among_lung_cancer_deaths_wonder"])
        noncore = float(part.loc["NonCore (Nonmetro)", "comention_pct_among_lung_cancer_deaths_wonder"])
        ratio_rows.append({"node_label": NODE_LABELS[node], "noncore_vs_large_central_ratio": noncore / central})
    ratio = pd.DataFrame(ratio_rows).sort_values("noncore_vs_large_central_ratio")
    colors = ["#e15759" if value > 1 else "#4e79a7" for value in ratio["noncore_vs_large_central_ratio"]]
    ax3.barh(ratio["node_label"], ratio["noncore_vs_large_central_ratio"], color=colors)
    ax3.axvline(1.0, color="#333333", linewidth=0.9)
    ax3.set_xlabel("Ratio of co-mention proportion")
    ax3.set_title("2024 NonCore vs Large Central Metro gradient", fontsize=11, loc="left")
    ax3.grid(axis="x", alpha=0.25)
    panel_letter(ax3, "C")

    ax4 = fig.add_subplot(gs[1, 1])
    change = urban_nt[urban_nt["node"].isin(NODE_ORDER)].pivot_table(
        index=["node_label", "stratum"],
        columns="year",
        values="comention_pct_among_lung_cancer_deaths_wonder",
        observed=False,
    )
    change.columns = [int(col) for col in change.columns]
    change["delta_2018_2024_pp"] = change[2024] - change[2018]
    delta_heat = (
        change["delta_2018_2024_pp"]
        .reset_index()
        .pivot_table(index="node_label", columns="stratum", values="delta_2018_2024_pp", observed=False)
    )
    delta_heat = delta_heat.reindex([NODE_LABELS[node] for node in NODE_ORDER], columns=URBAN_ORDER)
    lim = float(np.nanmax(np.abs(delta_heat.to_numpy(dtype=float))))
    im2 = ax4.imshow(delta_heat, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    annotate_heatmap(ax4, delta_heat)
    ax4.set_xticks(range(len(URBAN_ORDER)))
    ax4.set_xticklabels(URBAN_ORDER, rotation=35, ha="right")
    ax4.set_yticks(range(len(delta_heat.index)))
    ax4.set_yticklabels(delta_heat.index)
    ax4.set_title("Change in co-mention proportion, 2018 to 2024", fontsize=11, loc="left")
    cbar2 = fig.colorbar(im2, ax=ax4, fraction=0.046, pad=0.04)
    cbar2.set_label("Percentage points")
    panel_letter(ax4, "D")

    fig.suptitle("CDC WONDER validation: urban-rural gradient in lung cancer mortality and comorbidity co-mentions", fontsize=14)
    save_figure("Figure12_P1_wonder_urbanization_validation")


def format_pct(value: float) -> str:
    return f"{value:.1f}%"


def write_report(region: pd.DataFrame, urban: pd.DataFrame) -> None:
    region_2024 = region[(region["year"].eq(2024)) & (~region["is_total_row"])]
    urban_2024 = urban[(urban["year"].eq(2024)) & (~urban["is_total_row"])]
    total = region[(region["node"].eq("lung_cancer_total")) & (region["is_total_row"])].sort_values("year")
    total_2018 = int(total.loc[total["year"].eq(2018), "deaths"].iloc[0])
    total_2024 = int(total.loc[total["year"].eq(2024), "deaths"].iloc[0])

    lc_region = region_2024[region_2024["node"].eq("lung_cancer_total")].sort_values(
        "age_adjusted_rate_per_100k", ascending=False
    )
    urban_rate_year = int(
        urban[
            urban["node"].eq("lung_cancer_total")
            & (~urban["is_total_row"])
            & urban["crude_rate_per_100k"].notna()
        ]["year"].max()
    )
    lc_urban_rates = urban[
        urban["year"].eq(urban_rate_year)
        & urban["node"].eq("lung_cancer_total")
        & (~urban["is_total_row"])
    ].sort_values(
        "crude_rate_per_100k", ascending=False
    )
    lc_urban_2024_counts = urban_2024[urban_2024["node"].eq("lung_cancer_total")].sort_values(
        "deaths", ascending=False
    )

    node_region_lines = []
    for node in NODE_ORDER:
        part = region_2024[region_2024["node"].eq(node)].copy()
        hi = part.loc[part["comention_pct_among_lung_cancer_deaths_wonder"].idxmax()]
        lo = part.loc[part["comention_pct_among_lung_cancer_deaths_wonder"].idxmin()]
        node_region_lines.append(
            f"- {NODE_LABELS[node]}: highest {hi['stratum']} {format_pct(hi['comention_pct_among_lung_cancer_deaths_wonder'])}; "
            f"lowest {lo['stratum']} {format_pct(lo['comention_pct_among_lung_cancer_deaths_wonder'])}; "
            f"range {hi['comention_pct_among_lung_cancer_deaths_wonder'] - lo['comention_pct_among_lung_cancer_deaths_wonder']:.1f} pp."
        )

    urban_ratio_lines = []
    for node in NODE_ORDER:
        part = urban_2024[urban_2024["node"].eq(node)].set_index("stratum")
        central = float(part.loc["Large Central Metro", "comention_pct_among_lung_cancer_deaths_wonder"])
        noncore = float(part.loc["NonCore (Nonmetro)", "comention_pct_among_lung_cancer_deaths_wonder"])
        urban_ratio_lines.append(
            f"- {NODE_LABELS[node]}: NonCore {format_pct(noncore)} vs Large Central Metro {format_pct(central)} "
            f"(ratio {noncore / central:.2f})."
        )

    lines = [
        "# P1 CDC WONDER verified region/urbanization add-on results",
        "",
        "Generated from CDC WONDER Multiple Cause of Death, 2018-2024, Single Race exports.",
        "",
        "## Scope note",
        "",
        "- WONDER tables use U.S. resident population denominators and WONDER resident death scope.",
        "- Main MCOD raw-file analyses remain the primary denominator for manuscript results.",
        "- These WONDER outputs are best framed as external validation and geographic/urbanization extension.",
        f"- WONDER UCD C34 total deaths: {total_2018:,} in 2018 and {total_2024:,} in 2024.",
        "",
        "## 2024 regional mortality gradient",
        "",
        *[
            f"- {row['stratum']}: lung cancer age-adjusted mortality {row['age_adjusted_rate_per_100k']:.1f} per 100,000."
            for _, row in lc_region.iterrows()
        ],
        "",
        "## 2024 regional co-mention heterogeneity",
        "",
        *node_region_lines,
        "",
        f"## {urban_rate_year} urbanization mortality gradient, latest WONDER rate year available",
        "",
        *[
            f"- {row['stratum']}: lung cancer crude mortality {row['crude_rate_per_100k']:.1f} per 100,000."
            for _, row in lc_urban_rates.iterrows()
        ],
        "",
        "## 2024 urbanization lung cancer death counts",
        "",
        *[
            f"- {row['stratum']}: {int(row['deaths']):,} lung cancer deaths."
            for _, row in lc_urban_2024_counts.iterrows()
        ],
        "",
        "## 2024 NonCore versus Large Central Metro co-mention gradient",
        "",
        *urban_ratio_lines,
        "",
        "## Manuscript-ready interpretation",
        "",
        "The WONDER add-on supports a geographic extension of the main MCOD network findings: the South and Midwest carried higher lung cancer mortality rates than the West/Northeast, while COPD and terminal respiratory co-mentions showed meaningful regional and urbanization gradients. The urban-rural gradient was strongest for COPD, supporting a clinically interpretable hypothesis that smoking-related and access-sensitive chronic respiratory burden remains embedded in lung cancer mortality trajectories.",
        "",
        "## Generated files",
        "",
        "- `P1_wonder_region_rates_long_2018_2024.csv`",
        "- `P1_wonder_region_rates_merged_2018_2024.csv`",
        "- `P1_wonder_urbanization_long_2018_2024.csv`",
        "- `P1_wonder_urbanization_merged_2018_2024.csv`",
        "- `figures/Figure11_P1_wonder_region_validation.png`",
        "- `figures/Figure12_P1_wonder_urbanization_validation.png`",
    ]
    (OUT / "P1_wonder_verified_region_urbanization_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    region = collect_family("verified_region_rates", "Census Region", "Census Region Code")
    urban = collect_family("verified_urbanization_counts", "2013 Urbanization", "2013 Urbanization Code")
    region_merged = add_lung_denominators(region)
    urban_merged = add_lung_denominators(urban)

    region.to_csv(OUT / "P1_wonder_region_rates_long_2018_2024.csv", index=False, encoding="utf-8-sig")
    region_merged.to_csv(OUT / "P1_wonder_region_rates_merged_2018_2024.csv", index=False, encoding="utf-8-sig")
    urban.to_csv(OUT / "P1_wonder_urbanization_long_2018_2024.csv", index=False, encoding="utf-8-sig")
    urban_merged.to_csv(OUT / "P1_wonder_urbanization_merged_2018_2024.csv", index=False, encoding="utf-8-sig")

    plot_region(region_merged)
    plot_urban(urban_merged)
    write_report(region_merged, urban_merged)
    print(f"Wrote verified WONDER outputs to {OUT}")


if __name__ == "__main__":
    main()
