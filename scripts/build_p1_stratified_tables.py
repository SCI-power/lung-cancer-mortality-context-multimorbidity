from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[0]
RAW = PROJECT / "raw"
OUT = PROJECT / "outputs"
P1_PARSER = PROJECT / "scripts" / "parse_nchs_mcod_year.py"


spec = importlib.util.spec_from_file_location("p1_parser", P1_PARSER)
p1_parser = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(p1_parser)


SEX_LABELS = {
    "M": "Male",
    "F": "Female",
    "": "Unknown",
}

AGE12_LABELS = {
    "01": "Under 1 year",
    "02": "1-4 years",
    "03": "5-14 years",
    "04": "15-24 years",
    "05": "25-34 years",
    "06": "35-44 years",
    "07": "45-54 years",
    "08": "55-64 years",
    "09": "65-74 years",
    "10": "75-84 years",
    "11": "85+ years",
    "12": "Age not stated",
}

HISP_RACE_2022_LABELS = {
    "01": "Mexican",
    "02": "Puerto Rican",
    "03": "Cuban",
    "04": "Dominican",
    "05": "Central American",
    "06": "South American",
    "07": "Other/unknown Hispanic",
    "08": "Non-Hispanic White only",
    "09": "Non-Hispanic Black only",
    "10": "Non-Hispanic AIAN only",
    "11": "Non-Hispanic Asian only",
    "12": "Non-Hispanic NHOPI only",
    "13": "Non-Hispanic more than one race",
    "14": "Hispanic origin unknown/not stated",
}


def race_ethnicity_2022plus(line: str, year: int) -> str:
    if year < 2022:
        return ""
    code = line[486:488].strip()
    return code


def make_stratum_labels(year: int, rec: dict, line: str) -> list[tuple[str, str, str]]:
    rows = [
        ("sex", rec["sex"] or "", SEX_LABELS.get(rec["sex"] or "", "Unknown")),
        ("age12", rec["age12"] or "", AGE12_LABELS.get(rec["age12"] or "", f"Age12 {rec['age12']}")),
    ]
    race_code = race_ethnicity_2022plus(line, year)
    if race_code:
        rows.append(("race_ethnicity_2022plus", race_code, HISP_RACE_2022_LABELS.get(race_code, f"Code {race_code}")))
    return rows


def process_year(year: int, groups: list[dict]) -> tuple[list[dict], list[dict]]:
    zip_path = RAW / f"Mort{year}us.zip"
    denom = Counter()
    counts = Counter()

    for idx, line in enumerate(p1_parser.iter_records(zip_path), start=1):
        if idx % 500000 == 0:
            print(f"{year}: processed {idx:,}")
        rec = p1_parser.parse_record(line)
        if not rec["ucd"].startswith("C34"):
            continue
        strata = make_stratum_labels(year, rec, line)
        for dimension, stratum_code, _ in strata:
            denom[(year, dimension, stratum_code)] += 1

        present = []
        codes = rec["record_axis"]
        for group in groups:
            if any(p1_parser.any_match(code, group["terms"]) for code in codes):
                present.append(group["group"])

        for dimension, stratum_code, _ in strata:
            for group_name in set(present):
                counts[(year, dimension, stratum_code, group_name)] += 1

    denom_rows = []
    for (yr, dimension, stratum_code), deaths in sorted(denom.items()):
        label = {
            "sex": SEX_LABELS,
            "age12": AGE12_LABELS,
            "race_ethnicity_2022plus": HISP_RACE_2022_LABELS,
        }.get(dimension, {}).get(stratum_code, stratum_code)
        denom_rows.append(
            {
                "year": yr,
                "dimension": dimension,
                "stratum_code": stratum_code,
                "stratum_label": label,
                "lung_cancer_ucd_deaths": deaths,
            }
        )

    count_rows = []
    for (yr, dimension, stratum_code, group_name), deaths in sorted(counts.items()):
        denom_deaths = denom[(yr, dimension, stratum_code)]
        label = {
            "sex": SEX_LABELS,
            "age12": AGE12_LABELS,
            "race_ethnicity_2022plus": HISP_RACE_2022_LABELS,
        }.get(dimension, {}).get(stratum_code, stratum_code)
        count_rows.append(
            {
                "year": yr,
                "dimension": dimension,
                "stratum_code": stratum_code,
                "stratum_label": label,
                "group": group_name,
                "deaths": deaths,
                "lung_cancer_ucd_deaths_in_stratum": denom_deaths,
                "proportion_among_stratum_lung_cancer_deaths": deaths / denom_deaths if denom_deaths else 0,
            }
        )
    print(f"{year}: strata={len(denom_rows)}, count_rows={len(count_rows)}")
    return denom_rows, count_rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    groups = p1_parser.load_groups(priority=None)

    denom_all = []
    counts_all = []
    for year in range(2018, 2025):
        denom_rows, count_rows = process_year(year, groups)
        denom_all.extend(denom_rows)
        counts_all.extend(count_rows)

    denom_df = pd.DataFrame(denom_all)
    counts_df = pd.DataFrame(counts_all)
    denom_path = OUT / "P1_mcod_2018_2024_stratified_denominators.csv"
    counts_path = OUT / "P1_mcod_2018_2024_stratified_comorbidity_counts.csv"
    denom_df.to_csv(denom_path, index=False, encoding="utf-8-sig")
    counts_df.to_csv(counts_path, index=False, encoding="utf-8-sig")

    top2024 = (
        counts_df[(counts_df["year"] == 2024) & (counts_df["dimension"].isin(["sex", "age12", "race_ethnicity_2022plus"]))]
        .sort_values(["dimension", "group", "proportion_among_stratum_lung_cancer_deaths"], ascending=[True, True, False])
    )
    top2024.to_csv(OUT / "P1_mcod_2024_stratified_comorbidity_counts.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# P1 Stratified MCOD Gate Report",
        "",
        "## Data source and scope",
        "",
        "- Source: NCHS MCOD public-use raw files, 2018-2024.",
        "- Cohort: underlying cause C34 lung cancer deaths.",
        "- Stratification dimensions generated from raw file: sex, age12, and Hispanic origin/race recode for 2022-2024.",
        "- Urban-rural is not available in the raw public-use MCOD files used here; it requires CDC WONDER aggregated export or another geographic crosswalk source.",
        "- Race/Hispanic origin uses the expanded 2022+ single-race recode and should not be treated as a continuous 2018-2024 race trend.",
        "",
        "## Outputs",
        "",
        f"- `{denom_path.name}`",
        f"- `{counts_path.name}`",
        "- `P1_mcod_2024_stratified_comorbidity_counts.csv`",
        "",
        "## 2024 denominator coverage",
        "",
    ]
    d2024 = denom_df[denom_df["year"] == 2024].sort_values(["dimension", "lung_cancer_ucd_deaths"], ascending=[True, False])
    for _, row in d2024.iterrows():
        lines.append(f"- {row['dimension']} / {row['stratum_label']}: {int(row['lung_cancer_ucd_deaths']):,}")
    lines += [
        "",
        "## Gate decision",
        "",
        "Sex and age stratified analyses are ready for 2018-2024. Race/Hispanic origin stratified analyses are ready for 2022-2024 only. Urban-rural analysis should be treated as a separate CDC WONDER export task, not a raw-file parser task.",
    ]
    (OUT / "P1_mcod_2018_2024_stratified_gate_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {denom_path}")
    print(f"Wrote {counts_path}")


if __name__ == "__main__":
    main()
