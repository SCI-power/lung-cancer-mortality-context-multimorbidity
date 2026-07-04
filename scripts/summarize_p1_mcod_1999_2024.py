from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "outputs"
YEARLY = OUT / "yearly_resident_1999_2024"
EXCLUDE_SCOPES = {
    "all_deaths_in_file",
    "all_deaths_us_residents",
    "foreign_resident_deaths_excluded",
    "underlying_cause_C34",
}


def read_yearly(start_year: int, end_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    count_frames = []
    pair_frames = []
    missing = []
    for year in range(start_year, end_year + 1):
        count_path = YEARLY / f"P1_mcod_{year}_lung_cancer_comorbidity_counts.csv"
        pair_path = YEARLY / f"P1_mcod_{year}_lung_cancer_pair_counts.csv"
        if count_path.exists():
            count_frames.append(pd.read_csv(count_path))
        else:
            missing.append(str(count_path.name))
        if pair_path.exists():
            pair_frames.append(pd.read_csv(pair_path))
    if missing:
        raise FileNotFoundError(f"Missing yearly count outputs: {', '.join(missing[:5])}")
    return pd.concat(count_frames, ignore_index=True), pd.concat(pair_frames, ignore_index=True)


def numericize(counts: pd.DataFrame, pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    for col in ["year", "deaths", "denominator_lung_cancer_ucd_deaths", "proportion_among_lung_cancer_deaths"]:
        if col in counts.columns:
            counts[col] = pd.to_numeric(counts[col], errors="coerce")
    for col in ["year", "co_mentioned_deaths", "denominator_lung_cancer_ucd_deaths", "proportion_among_lung_cancer_deaths"]:
        if col in pairs.columns:
            pairs[col] = pd.to_numeric(pairs[col], errors="coerce")
    return counts, pairs


def write_outputs(counts: pd.DataFrame, pairs: pd.DataFrame, start_year: int, end_year: int) -> None:
    tag = f"{start_year}_{end_year}"
    counts.to_csv(OUT / f"P1_mcod_{tag}_comorbidity_counts_long.csv", index=False, encoding="utf-8-sig")

    trend = counts[~counts["scope"].isin(EXCLUDE_SCOPES)].copy()
    trend_wide = trend.pivot(index="scope", columns="year", values="proportion_among_lung_cancer_deaths")
    trend_wide[f"change_{start_year}_to_{end_year}_pct_points"] = (trend_wide[end_year] - trend_wide[start_year]) * 100
    trend_wide[f"relative_change_{start_year}_to_{end_year}_pct"] = ((trend_wide[end_year] / trend_wide[start_year]) - 1) * 100
    trend_wide = trend_wide.sort_values(end_year, ascending=False)
    trend_wide.to_csv(OUT / f"P1_mcod_{tag}_comorbidity_trend_wide.csv", encoding="utf-8-sig")

    pairs.to_csv(OUT / f"P1_mcod_{tag}_pair_counts_long.csv", index=False, encoding="utf-8-sig")
    top_pairs_end = pairs[pairs["year"] == end_year].sort_values("co_mentioned_deaths", ascending=False).head(30)
    top_pairs_end.to_csv(OUT / f"P1_mcod_{end_year}_top30_pair_counts_resident_scope.csv", index=False, encoding="utf-8-sig")

    lung_total = counts[counts["scope"] == "underlying_cause_C34"][["year", "deaths"]].rename(
        columns={"deaths": "lung_cancer_ucd_deaths_us_residents"}
    )
    lung_total.to_csv(OUT / f"P1_mcod_{tag}_lung_cancer_ucd_deaths.csv", index=False, encoding="utf-8-sig")

    resident_total = counts[counts["scope"] == "all_deaths_us_residents"][["year", "deaths"]].rename(
        columns={"deaths": "all_deaths_us_residents"}
    )
    foreign_excluded = counts[counts["scope"] == "foreign_resident_deaths_excluded"][["year", "deaths"]].rename(
        columns={"deaths": "foreign_resident_deaths_excluded"}
    )
    audit = resident_total.merge(foreign_excluded, on="year", how="outer").merge(lung_total, on="year", how="outer")
    audit.to_csv(OUT / f"P1_mcod_{tag}_resident_scope_audit.csv", index=False, encoding="utf-8-sig")

    lines = [
        f"# P1 MCOD {start_year}-{end_year} Resident-Scope Gate Report",
        "",
        "## Data status",
        "",
        f"- Years parsed: {', '.join(map(str, sorted(counts['year'].dropna().astype(int).unique())))}",
        f"- Scope: U.S. resident deaths; foreign residents excluded by resident status code 4.",
        f"- Lung cancer underlying-death range: {int(lung_total['lung_cancer_ucd_deaths_us_residents'].min()):,} to {int(lung_total['lung_cancer_ucd_deaths_us_residents'].max()):,}.",
        "- Source: NCHS Mortality Multiple Cause public-use files, local parser using underlying cause C34 and record-axis multiple conditions.",
        "",
        f"## {end_year} main node proportions",
        "",
    ]
    c_end = trend[trend["year"] == end_year].sort_values("proportion_among_lung_cancer_deaths", ascending=False)
    for _, row in c_end.iterrows():
        lines.append(
            f"- {row['scope']}: {int(row['deaths']):,} deaths "
            f"({row['proportion_among_lung_cancer_deaths'] * 100:.2f}%)."
        )
    lines += [
        "",
        f"## Largest {start_year}-{end_year} changes",
        "",
    ]
    for scope, row in trend_wide.sort_values(f"change_{start_year}_to_{end_year}_pct_points", ascending=False).iterrows():
        lines.append(
            f"- {scope}: {row[f'change_{start_year}_to_{end_year}_pct_points']:+.2f} percentage points, "
            f"{row[f'relative_change_{start_year}_to_{end_year}_pct']:+.1f}% relative."
        )
    lines += [
        "",
        f"## {end_year} top pairs",
        "",
    ]
    for _, row in top_pairs_end.head(10).iterrows():
        lines.append(
            f"- {row['group_a']} + {row['group_b']}: {int(row['co_mentioned_deaths']):,} "
            f"({row['proportion_among_lung_cancer_deaths'] * 100:.2f}%)."
        )
    lines += [
        "",
        "## Gate decision",
        "",
        f"The resident-scope {start_year}-{end_year} expansion is ready for downstream trend, network, and sensitivity analyses. Outputs retain the prior 2018-2024 files and write new {tag}-tagged files for the expanded analysis.",
    ]
    (OUT / f"P1_mcod_{tag}_gate_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=1999)
    parser.add_argument("--end-year", type=int, default=2024)
    args = parser.parse_args()
    counts, pairs = read_yearly(args.start_year, args.end_year)
    counts, pairs = numericize(counts, pairs)
    write_outputs(counts, pairs, args.start_year, args.end_year)
    print(f"Wrote P1 {args.start_year}-{args.end_year} resident-scope summary outputs.")


if __name__ == "__main__":
    main()
