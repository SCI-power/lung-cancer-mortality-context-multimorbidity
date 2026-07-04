from __future__ import annotations

import itertools
from collections import Counter
from pathlib import Path

import pandas as pd

from parse_nchs_mcod_year import any_match, iter_records, load_groups, parse_record


PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "raw"
OUT = PROJECT / "outputs"

COVID_TERMS = ["U071"]
EXCLUDE_SCOPES = {"all_deaths_in_file", "underlying_cause_C34"}


def record_has_group(codes: list[str], terms: list[str]) -> bool:
    return any(any_match(code, terms) for code in codes)


def process_year(year: int) -> tuple[list[dict], list[dict]]:
    zip_path = RAW / f"Mort{year}us.zip"
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    groups = load_groups(priority=None)
    all_lung = 0
    covid_lung = 0
    noncovid_lung = 0
    covid_group_counts = Counter()
    noncovid_group_counts = Counter()
    covid_pair_counts = Counter()
    noncovid_pair_counts = Counter()

    for idx, line in enumerate(iter_records(zip_path), start=1):
        if idx % 500000 == 0:
            print(f"{year}: processed {idx:,} records; lung deaths {all_lung:,}; COVID-coded lung deaths {covid_lung:,}")
        rec = parse_record(line)
        if not rec["ucd"].startswith("C34"):
            continue
        all_lung += 1
        codes = rec["record_axis"]
        has_covid = record_has_group(codes, COVID_TERMS)
        if has_covid:
            covid_lung += 1
        else:
            noncovid_lung += 1

        present: list[str] = []
        for group in groups:
            if record_has_group(codes, group["terms"]):
                present.append(group["group"])
        present = sorted(set(present))

        if has_covid:
            for group in present:
                covid_group_counts[group] += 1
            for a, b in itertools.combinations(present, 2):
                covid_pair_counts[(a, b)] += 1
        else:
            for group in present:
                noncovid_group_counts[group] += 1
            for a, b in itertools.combinations(present, 2):
                noncovid_pair_counts[(a, b)] += 1

    count_rows: list[dict] = [
        {
            "year": year,
            "analysis_set": "all_lung_cancer_ucd_deaths",
            "scope": "underlying_cause_C34",
            "deaths": all_lung,
            "denominator_lung_cancer_ucd_deaths": all_lung,
            "proportion_among_lung_cancer_deaths": 1.0,
        },
        {
            "year": year,
            "analysis_set": "covid_coded_lung_cancer_ucd_deaths",
            "scope": "covid_u071",
            "deaths": covid_lung,
            "denominator_lung_cancer_ucd_deaths": all_lung,
            "proportion_among_lung_cancer_deaths": covid_lung / all_lung if all_lung else "",
        },
        {
            "year": year,
            "analysis_set": "non_covid_lung_cancer_ucd_deaths",
            "scope": "underlying_cause_C34_no_record_axis_U071",
            "deaths": noncovid_lung,
            "denominator_lung_cancer_ucd_deaths": noncovid_lung,
            "proportion_among_lung_cancer_deaths": 1.0,
        },
    ]
    for group in groups:
        g = group["group"]
        count_rows.append(
            {
                "year": year,
                "analysis_set": "covid_coded_lung_cancer_ucd_deaths",
                "scope": g,
                "deaths": covid_group_counts[g],
                "denominator_lung_cancer_ucd_deaths": covid_lung,
                "proportion_among_lung_cancer_deaths": covid_group_counts[g] / covid_lung if covid_lung else "",
            }
        )
        count_rows.append(
            {
                "year": year,
                "analysis_set": "non_covid_lung_cancer_ucd_deaths",
                "scope": g,
                "deaths": noncovid_group_counts[g],
                "denominator_lung_cancer_ucd_deaths": noncovid_lung,
                "proportion_among_lung_cancer_deaths": noncovid_group_counts[g] / noncovid_lung if noncovid_lung else "",
            }
        )

    pair_rows: list[dict] = []
    for analysis_set, pair_counts, denom in [
        ("covid_coded_lung_cancer_ucd_deaths", covid_pair_counts, covid_lung),
        ("non_covid_lung_cancer_ucd_deaths", noncovid_pair_counts, noncovid_lung),
    ]:
        for (a, b), count in pair_counts.most_common():
            pair_rows.append(
                {
                    "year": year,
                    "analysis_set": analysis_set,
                    "group_a": a,
                    "group_b": b,
                    "co_mentioned_deaths": count,
                    "denominator_lung_cancer_ucd_deaths": denom,
                    "proportion_among_lung_cancer_deaths": count / denom if denom else "",
                }
            )

    print(f"Finished {year}: C34 deaths={all_lung:,}; with U07.1={covid_lung:,}; without U07.1={noncovid_lung:,}")
    return count_rows, pair_rows


def build_comparison(counts: pd.DataFrame) -> pd.DataFrame:
    main = pd.read_csv(OUT / "P1_mcod_2018_2024_comorbidity_counts_long.csv")
    main = main[~main["scope"].isin(EXCLUDE_SCOPES)].copy()
    main = main.rename(
        columns={
            "deaths": "main_deaths",
            "denominator_lung_cancer_ucd_deaths": "main_denominator",
            "proportion_among_lung_cancer_deaths": "main_proportion",
        }
    )
    noncovid = counts[counts["analysis_set"] == "non_covid_lung_cancer_ucd_deaths"].copy()
    noncovid = noncovid[~noncovid["scope"].isin(["underlying_cause_C34_no_record_axis_U071"])].copy()
    noncovid = noncovid.rename(
        columns={
            "deaths": "noncovid_deaths",
            "denominator_lung_cancer_ucd_deaths": "noncovid_denominator",
            "proportion_among_lung_cancer_deaths": "noncovid_proportion",
        }
    )
    merged = main[["year", "scope", "main_deaths", "main_denominator", "main_proportion"]].merge(
        noncovid[["year", "scope", "noncovid_deaths", "noncovid_denominator", "noncovid_proportion"]],
        on=["year", "scope"],
        how="left",
    )
    merged["main_proportion_pct"] = pd.to_numeric(merged["main_proportion"], errors="coerce") * 100
    merged["noncovid_proportion_pct"] = pd.to_numeric(merged["noncovid_proportion"], errors="coerce") * 100
    merged["absolute_difference_noncovid_minus_main_pct_points"] = (
        merged["noncovid_proportion_pct"] - merged["main_proportion_pct"]
    )
    merged["relative_difference_noncovid_vs_main_pct"] = (
        (merged["noncovid_proportion_pct"] / merged["main_proportion_pct"] - 1.0) * 100
    )
    return merged


def write_report(counts: pd.DataFrame, comparison: pd.DataFrame) -> None:
    covid_rows = counts[counts["analysis_set"] == "covid_coded_lung_cancer_ucd_deaths"]
    covid_marker = covid_rows[covid_rows["scope"] == "covid_u071"].copy()
    covid_marker["proportion_pct"] = pd.to_numeric(covid_marker["proportion_among_lung_cancer_deaths"], errors="coerce") * 100

    comp2024 = comparison[comparison["year"] == 2024].copy()
    comp2024 = comp2024.sort_values("absolute_difference_noncovid_minus_main_pct_points")
    lines = [
        "# P1 COVID Sensitivity Report",
        "",
        "## COVID co-mention burden among lung cancer underlying-cause deaths",
        "",
    ]
    for _, row in covid_marker.sort_values("year").iterrows():
        lines.append(
            f"- {int(row['year'])}: {int(row['deaths']):,} of {int(row['denominator_lung_cancer_ucd_deaths']):,} "
            f"lung cancer deaths included record-axis U07.1 ({row['proportion_pct']:.2f}%)."
        )

    lines += [
        "",
        "## 2024 effect of excluding record-axis U07.1",
        "",
        "Negative values mean the condition proportion was lower after excluding COVID-coded deaths.",
        "",
    ]
    for _, row in comp2024.head(10).iterrows():
        lines.append(
            f"- {row['scope']}: main {row['main_proportion_pct']:.2f}%, non-COVID {row['noncovid_proportion_pct']:.2f}%, "
            f"difference {row['absolute_difference_noncovid_minus_main_pct_points']:+.2f} percentage points."
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "This sensitivity analysis tests whether respiratory and infectious co-mention patterns are dominated by deaths that also listed COVID-19. The primary network remains interpretable only if chronic-core nodes and main chronic pairs persist after excluding U07.1-coded deaths.",
        "",
        "Use the non-COVID tables as sensitivity results, not as the primary analysis, because COVID-19 is itself part of the observed 2020-2024 mortality context.",
    ]
    (OUT / "P1_covid_sensitivity_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    all_count_rows: list[dict] = []
    all_pair_rows: list[dict] = []
    for year in range(2018, 2025):
        counts, pairs = process_year(year)
        all_count_rows.extend(counts)
        all_pair_rows.extend(pairs)

    counts_df = pd.DataFrame(all_count_rows)
    pairs_df = pd.DataFrame(all_pair_rows)
    counts_df.to_csv(OUT / "P1_covid_sensitivity_counts_2018_2024.csv", index=False, encoding="utf-8-sig")
    pairs_df.to_csv(OUT / "P1_covid_sensitivity_pair_counts_2018_2024.csv", index=False, encoding="utf-8-sig")
    comparison = build_comparison(counts_df)
    comparison.to_csv(OUT / "P1_covid_sensitivity_main_vs_noncovid_comparison.csv", index=False, encoding="utf-8-sig")
    write_report(counts_df, comparison)
    print("Wrote COVID sensitivity outputs.")


if __name__ == "__main__":
    main()
