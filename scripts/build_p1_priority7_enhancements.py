from __future__ import annotations

import itertools
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "raw"
OUT = PROJECT / "outputs"
YEARLY = OUT / "yearly_resident_1999_2024"
FIG = OUT / "figures"

START_YEAR = 1999
END_YEAR = 2024
RACE_START_YEAR = 2018
REFERENCE_YEAR = 2024
REFERENCE_RACE = "Non-Hispanic White"

TERMINAL_ACUTE_NODES = {"respiratory_failure", "pneumonia_influenza", "pulmonary_embolism"}
CHRONIC_NODES = {
    "copd",
    "ild",
    "ischemic_heart_disease",
    "heart_failure",
    "atrial_fibrillation",
    "cerebrovascular",
    "diabetes",
    "ckd",
    "autoimmune_broad",
    "depression_anxiety",
    "serious_mental_illness",
    "non_tobacco_substance_opioid",
}
CORE_CHRONIC_NODES = [
    "copd",
    "ischemic_heart_disease",
    "diabetes",
    "heart_failure",
    "atrial_fibrillation",
    "cerebrovascular",
    "ckd",
    "ild",
]

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

NETWORK_LABELS = {
    "copd": "COPD",
    "ischemic_heart_disease": "IHD",
    "diabetes": "Diabetes",
    "heart_failure": "HF",
    "atrial_fibrillation": "AF",
    "cerebrovascular": "CVD",
    "ckd": "CKD",
    "ild": "ILD",
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
AGE_CODES = ["04", "05", "06", "07", "08", "09", "10", "11"]
RACES_MAIN = [
    "Non-Hispanic White",
    "Non-Hispanic Black",
    "Hispanic",
    "Non-Hispanic Asian",
    "Non-Hispanic AIAN",
    "Non-Hispanic Multiracial",
]
RACES_STABLE_FIG = [
    "Non-Hispanic White",
    "Non-Hispanic Black",
    "Hispanic",
    "Non-Hispanic Asian",
    "Non-Hispanic AIAN",
]
SEX_ORDER = ["Male", "Female"]


def label(group: str) -> str:
    return LABELS.get(group, group.replace("_", " ").title())


def import_parser_helpers():
    scripts_dir = PROJECT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from parse_nchs_mcod_year import any_match, iter_records, load_groups, parse_record

    return any_match, iter_records, load_groups, parse_record


def parse_int(raw: str) -> int | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def race_ethnicity_from_line(line: str) -> tuple[str, str, str]:
    hispanic_origin = line[483:486].strip() if len(line) >= 486 else ""
    race40 = line[488:490].strip() if len(line) >= 490 else ""
    hisp = parse_int(hispanic_origin)
    if hisp is not None and 200 <= hisp <= 299:
        return "Hispanic", hispanic_origin, race40
    if hisp is None or hisp >= 996:
        return "Unknown/Not stated", hispanic_origin, race40
    race = parse_int(race40)
    if race == 1:
        return "Non-Hispanic White", hispanic_origin, race40
    if race == 2:
        return "Non-Hispanic Black", hispanic_origin, race40
    if race == 3:
        return "Non-Hispanic AIAN", hispanic_origin, race40
    if race is not None and 4 <= race <= 10:
        return "Non-Hispanic Asian", hispanic_origin, race40
    if race is not None and 11 <= race <= 14:
        return "Non-Hispanic NHOPI", hispanic_origin, race40
    if race is not None and 15 <= race <= 40:
        return "Non-Hispanic Multiracial", hispanic_origin, race40
    return "Unknown/Not stated", hispanic_origin, race40


def sex_label(raw: str) -> str:
    if raw == "M":
        return "Male"
    if raw == "F":
        return "Female"
    return "Unknown"


def burden_category(count: int) -> str:
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count == 2:
        return "2"
    return "3+"


def simple_markdown_table(df: pd.DataFrame, floatfmt: str = ".2f") -> str:
    if df.empty:
        return "_No rows._"
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else format(float(x), floatfmt))
        else:
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else str(x))
    headers = [str(c) for c in out.columns]
    rows = out.astype(str).values.tolist()
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    lines = [
        "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |",
    ]
    lines.extend("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows)
    return "\n".join(lines)


def savefig(name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "svg"]:
        plt.savefig(FIG / f"{name}.{ext}", dpi=400, bbox_inches="tight")
    plt.close()


def scan_cache_paths() -> dict[str, Path]:
    return {
        "burden": OUT / "P1_priority7_chronic_burden_raw_1999_2024.csv",
        "burden_age": OUT / "P1_priority7_chronic_burden_age_raw_1999_2024.csv",
        "burden_race": OUT / "P1_priority7_chronic_burden_race_ethnicity_raw_2018_2024.csv",
        "burden_race_sex": OUT / "P1_priority7_chronic_burden_race_sex_raw_2018_2024.csv",
        "race_sex_denom": OUT / "P1_priority7_race_sex_denominators_2018_2024.csv",
        "race_sex_age_denom": OUT / "P1_priority7_race_sex_age_denominators_2018_2024.csv",
        "race_sex_nodes": OUT / "P1_priority7_race_sex_node_counts_2018_2024.csv",
        "race_sex_age_nodes": OUT / "P1_priority7_race_sex_node_age_counts_2018_2024.csv",
        "covid": OUT / "P1_priority7_covid_resident_counts_2018_2024.csv",
        "audit": OUT / "P1_priority7_scan_audit_1999_2024.csv",
    }


def scan_or_load() -> dict[str, pd.DataFrame]:
    paths = scan_cache_paths()
    if all(path.exists() and path.stat().st_size > 0 for path in paths.values()):
        print("Using cached priority-7 scan outputs.", flush=True)
        return {
            "burden": pd.read_csv(paths["burden"]),
            "burden_age": pd.read_csv(paths["burden_age"], dtype={"age12": str}),
            "burden_race": pd.read_csv(paths["burden_race"]),
            "burden_race_sex": pd.read_csv(paths["burden_race_sex"]),
            "race_sex_denom": pd.read_csv(paths["race_sex_denom"]),
            "race_sex_age_denom": pd.read_csv(paths["race_sex_age_denom"], dtype={"age12": str}),
            "race_sex_nodes": pd.read_csv(paths["race_sex_nodes"]),
            "race_sex_age_nodes": pd.read_csv(paths["race_sex_age_nodes"], dtype={"age12": str}),
            "covid": pd.read_csv(paths["covid"]),
            "audit": pd.read_csv(paths["audit"]),
        }

    any_match, iter_records, load_groups, parse_record = import_parser_helpers()
    groups = load_groups(priority=None)
    groups_by_id = {g["group"]: g for g in groups}
    chronic_groups = [groups_by_id[g] for g in CHRONIC_NODES if g in groups_by_id]
    terminal_groups = [groups_by_id[g] for g in TERMINAL_ACUTE_NODES if g in groups_by_id]

    burden_counts = Counter()
    burden_age_counts = Counter()
    burden_race_counts = Counter()
    burden_race_sex_counts = Counter()
    total_chronic_mentions = Counter()
    total_terminal_mentions = Counter()
    race_sex_denom = Counter()
    race_sex_age_denom = Counter()
    race_sex_nodes = Counter()
    race_sex_age_nodes = Counter()
    covid_counts = Counter()
    audit_rows = []

    for year in range(START_YEAR, END_YEAR + 1):
        zip_path = RAW / f"Mort{year}us.zip"
        if not zip_path.exists():
            raise FileNotFoundError(zip_path)
        total = 0
        residents = 0
        foreign = 0
        lung = 0
        print(f"Priority-7 scan {year}...", flush=True)
        for idx, line in enumerate(iter_records(zip_path), start=1):
            total += 1
            if idx % 500000 == 0:
                print(f"  {year}: processed {idx:,}; resident C34 deaths {lung:,}", flush=True)
            rec = parse_record(line)
            if rec["resident_status"] == "4":
                foreign += 1
                continue
            residents += 1
            if not rec["ucd"].startswith("C34"):
                continue
            lung += 1
            age12 = rec["age12"] if rec["age12"] in AGE12_LABELS else "12"
            sex = sex_label(rec["sex"])
            codes = rec["record_axis"]
            chronic_present = sorted(
                g["group"] for g in chronic_groups if any(any_match(code, g["terms"]) for code in codes)
            )
            terminal_present = sorted(
                g["group"] for g in terminal_groups if any(any_match(code, g["terms"]) for code in codes)
            )
            bcat = burden_category(len(chronic_present))
            burden_counts[(year, bcat)] += 1
            burden_age_counts[(year, age12, bcat)] += 1
            total_chronic_mentions[year] += len(chronic_present)
            total_terminal_mentions[year] += len(terminal_present)

            if year >= RACE_START_YEAR:
                race, _, _ = race_ethnicity_from_line(line)
                burden_race_counts[(year, race, bcat)] += 1
                burden_race_sex_counts[(year, race, sex, bcat)] += 1
                race_sex_denom[(year, race, sex)] += 1
                race_sex_age_denom[(year, race, sex, age12)] += 1
                for group in sorted(set(chronic_present + terminal_present)):
                    race_sex_nodes[(year, race, sex, group)] += 1
                    race_sex_age_nodes[(year, race, sex, age12, group)] += 1

                has_covid = any(any_match(code, ["U071"]) for code in codes)
                covid_set = "u071_coded" if has_covid else "non_u071"
                covid_counts[(year, covid_set, "underlying_cause_C34")] += 1
                for group in sorted(set(chronic_present + terminal_present)):
                    covid_counts[(year, covid_set, group)] += 1

        for bcat in ["0", "1", "2", "3+"]:
            burden_counts[(year, bcat)] += 0
        audit_rows.append(
            {
                "year": year,
                "all_deaths_in_file": total,
                "all_deaths_us_residents": residents,
                "foreign_resident_deaths_excluded": foreign,
                "resident_underlying_cause_C34": lung,
                "chronic_mention_sum": total_chronic_mentions[year],
                "terminal_or_acute_mention_sum": total_terminal_mentions[year],
            }
        )
        print(f"Finished {year}: resident C34={lung:,}", flush=True)

    denom_by_year = {row["year"]: row["resident_underlying_cause_C34"] for row in audit_rows}
    burden = pd.DataFrame(
        [
            {
                "year": year,
                "burden_category": cat,
                "deaths": count,
                "denominator_lung_cancer_ucd_deaths": denom_by_year[year],
                "proportion": count / denom_by_year[year] if denom_by_year[year] else np.nan,
                "mean_chronic_mentions_per_death": total_chronic_mentions[year] / denom_by_year[year] if denom_by_year[year] else np.nan,
                "mean_terminal_or_acute_mentions_per_death": total_terminal_mentions[year] / denom_by_year[year] if denom_by_year[year] else np.nan,
            }
            for (year, cat), count in sorted(burden_counts.items())
        ]
    )
    burden_age = pd.DataFrame(
        [
            {"year": year, "age12": age12, "burden_category": cat, "deaths": count}
            for (year, age12, cat), count in sorted(burden_age_counts.items())
        ]
    )
    burden_race = pd.DataFrame(
        [
            {"year": year, "race_ethnicity": race, "burden_category": cat, "deaths": count}
            for (year, race, cat), count in sorted(burden_race_counts.items())
        ]
    )
    burden_race_sex = pd.DataFrame(
        [
            {"year": year, "race_ethnicity": race, "sex": sex, "burden_category": cat, "deaths": count}
            for (year, race, sex, cat), count in sorted(burden_race_sex_counts.items())
        ]
    )
    race_sex_denom_df = pd.DataFrame(
        [
            {"year": year, "race_ethnicity": race, "sex": sex, "lung_cancer_ucd_deaths": count}
            for (year, race, sex), count in sorted(race_sex_denom.items())
        ]
    )
    race_sex_age_denom_df = pd.DataFrame(
        [
            {"year": year, "race_ethnicity": race, "sex": sex, "age12": age12, "lung_cancer_ucd_deaths": count}
            for (year, race, sex, age12), count in sorted(race_sex_age_denom.items())
        ]
    )
    race_sex_nodes_df = pd.DataFrame(
        [
            {"year": year, "race_ethnicity": race, "sex": sex, "group": group, "deaths": count}
            for (year, race, sex, group), count in sorted(race_sex_nodes.items())
        ]
    )
    race_sex_age_nodes_df = pd.DataFrame(
        [
            {"year": year, "race_ethnicity": race, "sex": sex, "age12": age12, "group": group, "deaths": count}
            for (year, race, sex, age12, group), count in sorted(race_sex_age_nodes.items())
        ]
    )
    covid_df = pd.DataFrame(
        [
            {"year": year, "analysis_set": aset, "group": group, "deaths": count}
            for (year, aset, group), count in sorted(covid_counts.items())
        ]
    )
    audit = pd.DataFrame(audit_rows)

    paths["burden"].parent.mkdir(parents=True, exist_ok=True)
    burden.to_csv(paths["burden"], index=False)
    burden_age.to_csv(paths["burden_age"], index=False)
    burden_race.to_csv(paths["burden_race"], index=False)
    burden_race_sex.to_csv(paths["burden_race_sex"], index=False)
    race_sex_denom_df.to_csv(paths["race_sex_denom"], index=False)
    race_sex_age_denom_df.to_csv(paths["race_sex_age_denom"], index=False)
    race_sex_nodes_df.to_csv(paths["race_sex_nodes"], index=False)
    race_sex_age_nodes_df.to_csv(paths["race_sex_age_nodes"], index=False)
    covid_df.to_csv(paths["covid"], index=False)
    audit.to_csv(paths["audit"], index=False)

    return {
        "burden": burden,
        "burden_age": burden_age,
        "burden_race": burden_race,
        "burden_race_sex": burden_race_sex,
        "race_sex_denom": race_sex_denom_df,
        "race_sex_age_denom": race_sex_age_denom_df,
        "race_sex_nodes": race_sex_nodes_df,
        "race_sex_age_nodes": race_sex_age_nodes_df,
        "covid": covid_df,
        "audit": audit,
    }


def load_age_denominators() -> pd.DataFrame:
    denom = pd.read_csv(OUT / "P1_mcod_1999_2024_age12_denominators.csv", dtype={"age12": str})
    denom["age12"] = denom["age12"].map(lambda x: f"{int(x):02d}" if str(x).isdigit() else str(x))
    return denom


def load_age_group_counts() -> pd.DataFrame:
    rows = []
    for year in range(START_YEAR, END_YEAR + 1):
        path = YEARLY / f"P1_mcod_{year}_lung_cancer_group_by_age12.csv"
        part = pd.read_csv(path, dtype={"age12": str})
        part["age12"] = part["age12"].map(lambda x: f"{int(x):02d}" if str(x).isdigit() else str(x))
        rows.append(part)
    out = pd.concat(rows, ignore_index=True)
    out["year"] = out["year"].astype(int)
    out["deaths"] = pd.to_numeric(out["deaths"], errors="coerce").fillna(0)
    return out


def build_age_decomposition() -> pd.DataFrame:
    age_denom = load_age_denominators()
    age_counts = load_age_group_counts()
    groups = sorted(age_counts["group"].unique())
    rows = []
    d0 = age_denom[(age_denom["year"] == START_YEAR) & (age_denom["age12"].isin(AGE_CODES))].copy()
    d1 = age_denom[(age_denom["year"] == END_YEAR) & (age_denom["age12"].isin(AGE_CODES))].copy()
    d0 = d0.set_index("age12")["lung_cancer_ucd_deaths"].reindex(AGE_CODES).fillna(0).astype(float)
    d1 = d1.set_index("age12")["lung_cancer_ucd_deaths"].reindex(AGE_CODES).fillna(0).astype(float)
    w0 = d0 / d0.sum()
    w1 = d1 / d1.sum()
    for group in groups:
        if group in {"all_deaths_in_file", "all_deaths_us_residents", "foreign_resident_deaths_excluded", "underlying_cause_C34"}:
            continue
        c0 = (
            age_counts[(age_counts["year"] == START_YEAR) & (age_counts["group"] == group)]
            .set_index("age12")["deaths"]
            .reindex(AGE_CODES)
            .fillna(0)
            .astype(float)
        )
        c1 = (
            age_counts[(age_counts["year"] == END_YEAR) & (age_counts["group"] == group)]
            .set_index("age12")["deaths"]
            .reindex(AGE_CODES)
            .fillna(0)
            .astype(float)
        )
        r0 = c0 / d0.replace(0, np.nan)
        r1 = c1 / d1.replace(0, np.nan)
        observed_start = float((w0 * r0).sum())
        observed_end = float((w1 * r1).sum())
        composition_component = float((0.5 * (w1 - w0) * (r0 + r1)).sum())
        within_age_component = float((0.5 * (r1 - r0) * (w0 + w1)).sum())
        rows.append(
            {
                "group": group,
                "label": label(group),
                "start_year": START_YEAR,
                "end_year": END_YEAR,
                "start_age_standardized_like_pct": observed_start * 100,
                "end_age_standardized_like_pct": observed_end * 100,
                "total_change_pct_points": (observed_end - observed_start) * 100,
                "age_structure_component_pct_points": composition_component * 100,
                "within_age_component_pct_points": within_age_component * 100,
                "within_age_share_of_change_pct": within_age_component / (observed_end - observed_start) * 100
                if observed_end != observed_start
                else np.nan,
            }
        )
    out = pd.DataFrame(rows).sort_values("total_change_pct_points", ascending=False)
    out.to_csv(OUT / "P1_priority7_age_decomposition_1999_2024.csv", index=False)
    return out


def build_chronic_burden_outputs(scan: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    burden = scan["burden"].copy()
    burden["proportion_pct"] = burden["proportion"] * 100
    burden["burden_category"] = pd.Categorical(burden["burden_category"], categories=["0", "1", "2", "3+"], ordered=True)

    burden_age = scan["burden_age"].copy()
    age_denom = load_age_denominators()
    ref = age_denom[(age_denom["year"] == REFERENCE_YEAR) & (age_denom["age12"].isin(AGE_CODES))].copy()
    ref["reference_weight"] = ref["lung_cancer_ucd_deaths"] / ref["lung_cancer_ucd_deaths"].sum()
    scaffold = pd.MultiIndex.from_product(
        [range(START_YEAR, END_YEAR + 1), ["0", "1", "2", "3+"], AGE_CODES],
        names=["year", "burden_category", "age12"],
    ).to_frame(index=False)
    age = scaffold.merge(burden_age, on=["year", "age12", "burden_category"], how="left")
    age["deaths"] = age["deaths"].fillna(0)
    age = age.merge(age_denom[["year", "age12", "lung_cancer_ucd_deaths"]], on=["year", "age12"], how="left")
    age = age.merge(ref[["age12", "reference_weight"]], on="age12", how="left")
    age["age_specific_proportion"] = np.where(age["lung_cancer_ucd_deaths"] > 0, age["deaths"] / age["lung_cancer_ucd_deaths"], np.nan)
    age["weighted_proportion"] = age["age_specific_proportion"] * age["reference_weight"]
    std = (
        age.groupby(["year", "burden_category"], as_index=False)
        .agg(age_standardized_proportion=("weighted_proportion", "sum"))
    )
    std["age_standardized_pct"] = std["age_standardized_proportion"] * 100

    race = scan["burden_race"].copy()
    race_denom = pd.read_csv(OUT / "P1_race_ethnicity_mcod_2018_2024_denominators.csv")
    race = race.merge(race_denom, on=["year", "race_ethnicity"], how="left")
    race["proportion_pct"] = race["deaths"] / race["lung_cancer_ucd_deaths"] * 100

    burden.to_csv(OUT / "P1_priority7_chronic_burden_index_1999_2024.csv", index=False)
    std.to_csv(OUT / "P1_priority7_age_standardized_chronic_burden_index_1999_2024.csv", index=False)
    race.to_csv(OUT / "P1_priority7_race_ethnicity_chronic_burden_index_2018_2024.csv", index=False)
    return burden, std, race


def build_race_sex_outputs(scan: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = scan["race_sex_nodes"].copy()
    denom = scan["race_sex_denom"].copy()
    age_nodes = scan["race_sex_age_nodes"].copy()
    age_denom = scan["race_sex_age_denom"].copy()
    groups = sorted(nodes["group"].unique())
    scaffold = pd.MultiIndex.from_product(
        [range(RACE_START_YEAR, END_YEAR + 1), RACES_MAIN, SEX_ORDER, groups],
        names=["year", "race_ethnicity", "sex", "group"],
    ).to_frame(index=False)
    node = scaffold.merge(nodes, on=["year", "race_ethnicity", "sex", "group"], how="left")
    node["deaths"] = node["deaths"].fillna(0).astype(int)
    node = node.merge(denom, on=["year", "race_ethnicity", "sex"], how="left")
    node["proportion_pct"] = node["deaths"] / node["lung_cancer_ucd_deaths"] * 100
    node["label"] = node["group"].map(label)
    node.to_csv(OUT / "P1_priority7_race_sex_node_burden_2018_2024.csv", index=False)

    ref_base = pd.read_csv(OUT / "P1_race_ethnicity_mcod_2018_2024_age_denominators.csv", dtype={"age12": str})
    ref = ref_base[(ref_base["year"] == REFERENCE_YEAR) & (ref_base["age12"].isin(AGE_CODES))].copy()
    ref = ref.groupby("age12", as_index=False)["lung_cancer_ucd_deaths"].sum()
    ref["reference_weight"] = ref["lung_cancer_ucd_deaths"] / ref["lung_cancer_ucd_deaths"].sum()
    scaffold_age = pd.MultiIndex.from_product(
        [range(RACE_START_YEAR, END_YEAR + 1), RACES_MAIN, SEX_ORDER, groups, AGE_CODES],
        names=["year", "race_ethnicity", "sex", "group", "age12"],
    ).to_frame(index=False)
    std_age = scaffold_age.merge(age_nodes, on=["year", "race_ethnicity", "sex", "group", "age12"], how="left")
    std_age["deaths"] = std_age["deaths"].fillna(0)
    std_age = std_age.merge(age_denom, on=["year", "race_ethnicity", "sex", "age12"], how="left")
    std_age = std_age.merge(ref[["age12", "reference_weight"]], on="age12", how="left")
    std_age["age_specific_proportion"] = np.where(
        std_age["lung_cancer_ucd_deaths"] > 0,
        std_age["deaths"] / std_age["lung_cancer_ucd_deaths"],
        np.nan,
    )
    std_age["weighted_proportion"] = std_age["age_specific_proportion"] * std_age["reference_weight"]
    std = (
        std_age.groupby(["year", "race_ethnicity", "sex", "group"], as_index=False)
        .agg(age_standardized_proportion=("weighted_proportion", "sum"))
    )
    std["age_standardized_pct"] = std["age_standardized_proportion"] * 100
    std["label"] = std["group"].map(label)
    std.to_csv(OUT / "P1_priority7_race_sex_age_standardized_node_burden_2018_2024.csv", index=False)

    interaction_rows = []
    model_age = std_age[(std_age["year"] == REFERENCE_YEAR) & (std_age["race_ethnicity"].isin(RACES_STABLE_FIG)) & (std_age["sex"].isin(SEX_ORDER))].copy()
    model_age = model_age[model_age["lung_cancer_ucd_deaths"] > 0].copy()
    model_age["race_ethnicity"] = pd.Categorical(model_age["race_ethnicity"], categories=RACES_STABLE_FIG)
    model_age["sex"] = pd.Categorical(model_age["sex"], categories=["Female", "Male"])
    model_age["age12"] = pd.Categorical(model_age["age12"], categories=AGE_CODES)
    for group, part in model_age.groupby("group", observed=True):
        if part["deaths"].sum() < 50:
            continue
        try:
            fit = smf.glm(
                "deaths ~ C(age12) + C(race_ethnicity, Treatment(reference='Non-Hispanic White')) * C(sex, Treatment(reference='Female'))",
                data=part,
                family=sm.families.Poisson(),
                offset=np.log(part["lung_cancer_ucd_deaths"].astype(float)),
            ).fit(cov_type="HC0")
        except Exception as exc:
            interaction_rows.append({"group": group, "label": label(group), "model_status": f"failed: {exc}"})
            continue
        for term, p_value in fit.pvalues.items():
            if ":C(sex" in term:
                interaction_rows.append(
                    {
                        "group": group,
                        "label": label(group),
                        "interaction_term": term,
                        "interaction_rate_ratio": math.exp(float(fit.params[term])),
                        "interaction_p_value": float(p_value),
                        "model_status": "ok",
                    }
                )
    interactions = pd.DataFrame(interaction_rows)
    if not interactions.empty and "interaction_p_value" in interactions.columns:
        mask = interactions["interaction_p_value"].notna()
        interactions.loc[mask, "interaction_fdr_q_value"] = multipletests(
            interactions.loc[mask, "interaction_p_value"], method="fdr_bh"
        )[1]
    interactions.to_csv(OUT / "P1_priority7_race_sex_interaction_models_2024.csv", index=False)
    return std, interactions


def build_terminal_chronic_sensitivity() -> pd.DataFrame:
    nodes = pd.read_csv(OUT / "P1_node_table_1999_2024.csv")
    edges = pd.read_csv(OUT / "P1_edge_enrichment_1999_2024.csv")
    nodes["node_family"] = np.where(nodes["group"].isin(TERMINAL_ACUTE_NODES), "terminal_or_acute", "chronic_or_other")
    family = (
        nodes.groupby(["year", "node_family"], as_index=False)
        .agg(total_node_mentions=("node_deaths", "sum"), denominator=("denominator", "first"))
    )
    family["mention_intensity_per_100_lung_cancer_deaths"] = family["total_node_mentions"] / family["denominator"] * 100
    edge_rows = []
    for year, part in edges.groupby("year"):
        top = part.sort_values("co_mentioned_deaths", ascending=False).head(20).copy()
        terminal = top[
            top["group_a"].isin(TERMINAL_ACUTE_NODES) | top["group_b"].isin(TERMINAL_ACUTE_NODES)
        ]
        chronic = top[
            ~top["group_a"].isin(TERMINAL_ACUTE_NODES) & ~top["group_b"].isin(TERMINAL_ACUTE_NODES)
        ]
        edge_rows.append(
            {
                "year": int(year),
                "top20_edge_mentions": int(top["co_mentioned_deaths"].sum()),
                "top20_terminal_or_acute_edge_mentions": int(terminal["co_mentioned_deaths"].sum()),
                "top20_chronic_chronic_edge_mentions": int(chronic["co_mentioned_deaths"].sum()),
                "terminal_or_acute_share_of_top20_edge_mentions_pct": terminal["co_mentioned_deaths"].sum()
                / top["co_mentioned_deaths"].sum()
                * 100
                if top["co_mentioned_deaths"].sum()
                else np.nan,
            }
        )
    edge_summary = pd.DataFrame(edge_rows)
    merged = family.merge(edge_summary, on="year", how="left")
    merged.to_csv(OUT / "P1_priority7_terminal_vs_chronic_sensitivity_1999_2024.csv", index=False)
    return merged


def build_geographic_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    region = pd.read_csv(OUT / "P1_wonder_region_rates_merged_2018_2024.csv")
    urban = pd.read_csv(OUT / "P1_wonder_urbanization_merged_2018_2024.csv")
    region_nt = region[~region["is_total_row"] & region["node"].ne("lung_cancer_total")].copy()
    urban_nt = urban[~urban["is_total_row"] & urban["node"].ne("lung_cancer_total")].copy()
    region_rows = []
    for node, part in region_nt[region_nt["year"].eq(REFERENCE_YEAR)].groupby("node"):
        part = part.dropna(subset=["comention_pct_among_lung_cancer_deaths_wonder"])
        hi = part.loc[part["comention_pct_among_lung_cancer_deaths_wonder"].idxmax()]
        lo = part.loc[part["comention_pct_among_lung_cancer_deaths_wonder"].idxmin()]
        region_rows.append(
            {
                "node": node,
                "node_label": label(node),
                "highest_region": hi["stratum"],
                "highest_region_pct": hi["comention_pct_among_lung_cancer_deaths_wonder"],
                "lowest_region": lo["stratum"],
                "lowest_region_pct": lo["comention_pct_among_lung_cancer_deaths_wonder"],
                "region_range_pct_points": hi["comention_pct_among_lung_cancer_deaths_wonder"]
                - lo["comention_pct_among_lung_cancer_deaths_wonder"],
            }
        )
    urban_rows = []
    for node, part in urban_nt[urban_nt["year"].eq(REFERENCE_YEAR)].groupby("node"):
        vals = part.set_index("stratum")["comention_pct_among_lung_cancer_deaths_wonder"]
        large = vals.get("Large Central Metro", np.nan)
        noncore = vals.get("NonCore (Nonmetro)", np.nan)
        micro = vals.get("Micropolitan (Nonmetro)", np.nan)
        hi = part.loc[part["comention_pct_among_lung_cancer_deaths_wonder"].idxmax()]
        urban_rows.append(
            {
                "node": node,
                "node_label": label(node),
                "large_central_metro_pct": large,
                "micropolitan_pct": micro,
                "noncore_pct": noncore,
                "noncore_minus_large_central_pct_points": noncore - large if pd.notna(noncore) and pd.notna(large) else np.nan,
                "highest_urbanization_stratum": hi["stratum"],
                "highest_urbanization_pct": hi["comention_pct_among_lung_cancer_deaths_wonder"],
            }
        )
    region_summary = pd.DataFrame(region_rows).sort_values("region_range_pct_points", ascending=False)
    urban_summary = pd.DataFrame(urban_rows).sort_values("noncore_minus_large_central_pct_points", ascending=False)
    region_summary.to_csv(OUT / "P1_priority7_geographic_region_summary_2024.csv", index=False)
    urban_summary.to_csv(OUT / "P1_priority7_geographic_urbanization_summary_2024.csv", index=False)
    return region_summary, urban_summary


def build_covid_period_sensitivity(scan: dict[str, pd.DataFrame]) -> pd.DataFrame:
    covid = scan["covid"].copy()
    den = covid[covid["group"].eq("underlying_cause_C34")].rename(columns={"deaths": "denominator"})
    node = covid[~covid["group"].eq("underlying_cause_C34")].copy()
    node = node.merge(den[["year", "analysis_set", "denominator"]], on=["year", "analysis_set"], how="left")
    node["proportion_pct"] = node["deaths"] / node["denominator"] * 100
    node["period"] = pd.cut(
        node["year"],
        bins=[2017, 2019, 2021, 2024],
        labels=["2018-2019 pre-COVID", "2020-2021 COVID-era", "2022-2024 post-acute"],
    )
    period = (
        node.groupby(["period", "analysis_set", "group"], observed=True, as_index=False)
        .agg(deaths=("deaths", "sum"), denominator=("denominator", "sum"))
    )
    period["proportion_pct"] = period["deaths"] / period["denominator"] * 100
    wide = period.pivot_table(index=["period", "group"], columns="analysis_set", values="proportion_pct").reset_index()
    if {"u071_coded", "non_u071"}.issubset(wide.columns):
        wide["u071_minus_non_u071_pct_points"] = wide["u071_coded"] - wide["non_u071"]
    period.to_csv(OUT / "P1_priority7_covid_period_sensitivity_long_2018_2024.csv", index=False)
    wide.to_csv(OUT / "P1_priority7_covid_period_sensitivity_wide_2018_2024.csv", index=False)
    return wide


def plot_age_decomposition(decomp: pd.DataFrame) -> None:
    plot_df = decomp.reindex(decomp["total_change_pct_points"].abs().sort_values(ascending=False).index).head(10).copy()
    plot_df = plot_df.sort_values("total_change_pct_points")
    y = np.arange(plot_df.shape[0])
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    ax.barh(y, plot_df["age_structure_component_pct_points"], label="Age-structure component", color="#8da0cb")
    ax.barh(
        y,
        plot_df["within_age_component_pct_points"],
        left=plot_df["age_structure_component_pct_points"],
        label="Within-age component",
        color="#66c2a5",
    )
    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"])
    ax.set_xlabel("Change, percentage points")
    ax.set_title("Decomposition of 1999-2024 co-mention change")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    savefig("Figure21_P1_priority7_age_decomposition_1999_2024")


def plot_chronic_burden(burden: pd.DataFrame, std: pd.DataFrame, race: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
    cat_order = ["0", "1", "2", "3+"]
    pivot = burden.pivot_table(index="year", columns="burden_category", values="proportion_pct", observed=False).reindex(columns=cat_order)
    axes[0].stackplot(pivot.index, [pivot[c] for c in cat_order], labels=cat_order, colors=["#f0f0f0", "#bdd7e7", "#6baed6", "#2171b5"])
    axes[0].set_title("Crude chronic multimorbidity burden")
    axes[0].set_ylabel("% of lung cancer deaths")
    axes[0].legend(title="Chronic nodes", frameon=False, loc="upper left")

    mean = burden.drop_duplicates("year").sort_values("year")
    axes[1].plot(mean["year"], mean["mean_chronic_mentions_per_death"], marker="o", label="Chronic", color="#2ca25f")
    axes[1].plot(mean["year"], mean["mean_terminal_or_acute_mentions_per_death"], marker="o", label="Terminal/acute", color="#de2d26")
    axes[1].set_title("Mean co-mentions per death")
    axes[1].set_ylabel("Mean node count")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)

    r2024 = race[(race["year"] == REFERENCE_YEAR) & (race["burden_category"].eq("3+")) & (race["race_ethnicity"].isin(RACES_MAIN))].copy()
    r2024 = r2024.sort_values("proportion_pct")
    axes[2].barh(r2024["race_ethnicity"].str.replace("Non-Hispanic ", "NH "), r2024["proportion_pct"], color="#74a9cf")
    axes[2].set_title("3+ chronic nodes by race/ethnicity, 2024")
    axes[2].set_xlabel("% of lung cancer deaths")
    fig.tight_layout()
    savefig("Figure22_P1_priority7_chronic_burden_index")


def plot_race_sex(std: pd.DataFrame) -> None:
    selected = ["copd", "diabetes", "heart_failure", "ckd", "respiratory_failure", "pneumonia_influenza"]
    data = std[(std["year"] == REFERENCE_YEAR) & (std["race_ethnicity"].isin(RACES_STABLE_FIG)) & (std["group"].isin(selected))].copy()
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 8.0))
    axes = axes.ravel()
    vmax = data["age_standardized_pct"].max()
    last_im = None
    for ax, group in zip(axes, selected):
        part = data[data["group"].eq(group)]
        matrix = part.assign(stratum=part["race_ethnicity"].str.replace("Non-Hispanic ", "NH ") + "\n" + part["sex"]).pivot_table(
            index="label", columns="stratum", values="age_standardized_pct", observed=False
        )
        columns = []
        for race in RACES_STABLE_FIG:
            for sex in SEX_ORDER:
                columns.append(race.replace("Non-Hispanic ", "NH ") + "\n" + sex)
        matrix = matrix.reindex(columns=columns)
        last_im = ax.imshow(matrix.to_numpy(dtype=float), cmap="YlGnBu", aspect="auto", vmin=0, vmax=vmax)
        ax.set_title(label(group), fontsize=10)
        ax.set_xticks(range(len(columns)))
        ax.set_xticklabels(columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticks([])
        vals = matrix.to_numpy(dtype=float)
        for j, val in enumerate(vals[0]):
            if np.isfinite(val):
                ax.text(j, 0, f"{val:.1f}", ha="center", va="center", fontsize=7)
    fig.subplots_adjust(left=0.05, right=0.90, bottom=0.18, top=0.88, wspace=0.16, hspace=0.45)
    if last_im is not None:
        cax = fig.add_axes([0.92, 0.28, 0.018, 0.46])
        fig.colorbar(last_im, cax=cax, label="% of lung cancer deaths")
    fig.suptitle("Age-standardized race x sex co-mention burden, 2024", y=0.97, fontsize=13)
    savefig("Figure23_P1_priority7_race_sex_heatmap_2024")


def plot_terminal_chronic(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    fam = summary.drop_duplicates(["year", "node_family"])
    for family, part in fam.groupby("node_family"):
        axes[0].plot(part["year"], part["mention_intensity_per_100_lung_cancer_deaths"], marker="o", label=family.replace("_", " "))
    axes[0].set_title("Node co-mention intensity by node family")
    axes[0].set_ylabel("Mentions per 100 lung cancer deaths")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)
    edge = summary.drop_duplicates("year").sort_values("year")
    axes[1].plot(edge["year"], edge["terminal_or_acute_share_of_top20_edge_mentions_pct"], marker="o", color="#de2d26")
    axes[1].set_title("Terminal/acute share of top-20 edge burden")
    axes[1].set_ylabel("% of top-20 edge co-mentions")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    savefig("Figure24_P1_priority7_terminal_chronic_sensitivity")


def plot_geographic(region_summary: pd.DataFrame, urban_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    reg = region_summary.sort_values("region_range_pct_points")
    axes[0].barh(reg["node_label"], reg["region_range_pct_points"], color="#9ecae1")
    axes[0].set_title("Census Region range, 2024")
    axes[0].set_xlabel("Max-min percentage points")
    urb = urban_summary.sort_values("noncore_minus_large_central_pct_points")
    axes[1].barh(urb["node_label"], urb["noncore_minus_large_central_pct_points"], color="#fdae6b")
    axes[1].axvline(0, color="#333333", linewidth=0.8)
    axes[1].set_title("NonCore minus large central metro, 2024")
    axes[1].set_xlabel("Percentage points")
    fig.tight_layout()
    savefig("Figure25_P1_priority7_geographic_gradients_2024")


def plot_covid(covid_wide: pd.DataFrame) -> None:
    selected = ["respiratory_failure", "pneumonia_influenza", "heart_failure", "ckd", "copd", "diabetes"]
    data = covid_wide[covid_wide["group"].isin(selected)].copy()
    if "u071_minus_non_u071_pct_points" not in data.columns:
        return
    order = ["2018-2019 pre-COVID", "2020-2021 COVID-era", "2022-2024 post-acute"]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for group in selected:
        part = data[data["group"].eq(group)].copy()
        part["period"] = pd.Categorical(part["period"], categories=order, ordered=True)
        part = part.sort_values("period")
        ax.plot(part["period"].astype(str), part["u071_minus_non_u071_pct_points"], marker="o", label=label(group))
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("Difference in co-mention burden: U07.1-coded vs non-U07.1 lung cancer deaths")
    ax.set_ylabel("Percentage-point difference")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    savefig("Figure26_P1_priority7_covid_period_sensitivity")


def write_report(
    decomp: pd.DataFrame,
    burden: pd.DataFrame,
    burden_std: pd.DataFrame,
    burden_race: pd.DataFrame,
    race_sex_std: pd.DataFrame,
    race_sex_interactions: pd.DataFrame,
    terminal_summary: pd.DataFrame,
    region_summary: pd.DataFrame,
    urban_summary: pd.DataFrame,
    covid_wide: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    b1999 = burden[(burden["year"] == START_YEAR)].copy()
    b2024 = burden[(burden["year"] == END_YEAR)].copy()
    bcomp = b1999[["burden_category", "proportion_pct"]].merge(
        b2024[["burden_category", "proportion_pct"]],
        on="burden_category",
        suffixes=("_1999", "_2024"),
    )
    bcomp["change_pct_points"] = bcomp["proportion_pct_2024"] - bcomp["proportion_pct_1999"]
    race_3plus = burden_race[
        (burden_race["year"] == END_YEAR) & (burden_race["burden_category"].eq("3+")) & (burden_race["race_ethnicity"].isin(RACES_MAIN))
    ].sort_values("proportion_pct", ascending=False)
    sex_top = race_sex_std[
        (race_sex_std["year"] == END_YEAR)
        & (race_sex_std["race_ethnicity"].isin(RACES_STABLE_FIG))
        & (race_sex_std["group"].isin(["copd", "diabetes", "heart_failure", "ckd", "respiratory_failure"]))
    ].sort_values("age_standardized_pct", ascending=False).head(18)
    terminal2024 = terminal_summary[terminal_summary["year"] == END_YEAR].drop_duplicates("node_family")
    edge2024 = terminal_summary[terminal_summary["year"] == END_YEAR].drop_duplicates("year")
    covid_top = covid_wide[covid_wide["period"].astype(str).eq("2020-2021 COVID-era")].copy()
    if "u071_minus_non_u071_pct_points" in covid_top.columns:
        covid_top = covid_top.reindex(covid_top["u071_minus_non_u071_pct_points"].abs().sort_values(ascending=False).index).head(10)

    lines = [
        "# P1 priority-7 high-impact enhancement report",
        "",
        "## What was added",
        "",
        "1. Age-structure decomposition for 1999-2024 node changes.",
        "2. Exact chronic multimorbidity burden index per death certificate: 0, 1, 2, or 3+ chronic nodes.",
        "3. Race/ethnicity x sex module with age-standardized 2024 heatmaps and age-adjusted interaction models.",
        "4. Terminal/acute pathway sensitivity separated from chronic-core multimorbidity.",
        "5. CDC WONDER geographic enrichment using Census Region and 2013 urbanization strata.",
        "6. Resident-scope COVID-era/U07.1 sensitivity.",
        "7. Manuscript-level reframing around multimorbidity architecture, aging, equity, geography, and terminal-event bias.",
        "",
        "## Scan audit",
        "",
        simple_markdown_table(audit.tail(8), floatfmt=".0f"),
        "",
        "## 1. Age decomposition: largest 1999-2024 changes",
        "",
        simple_markdown_table(
            decomp.head(12)[
                [
                    "label",
                    "total_change_pct_points",
                    "age_structure_component_pct_points",
                    "within_age_component_pct_points",
                    "within_age_share_of_change_pct",
                ]
            ],
            ".2f",
        ),
        "",
        "## 2. Chronic multimorbidity burden index",
        "",
        simple_markdown_table(bcomp, ".2f"),
        "",
        "2024 race/ethnicity distribution for 3+ chronic nodes:",
        "",
        simple_markdown_table(race_3plus[["race_ethnicity", "deaths", "lung_cancer_ucd_deaths", "proportion_pct"]], ".2f"),
        "",
        "## 3. Race x sex: high-burden 2024 cells",
        "",
        simple_markdown_table(
            sex_top[["race_ethnicity", "sex", "label", "age_standardized_pct"]],
            ".2f",
        ),
        "",
        "Top age-adjusted race x sex interaction signals:",
        "",
        simple_markdown_table(
            race_sex_interactions.sort_values("interaction_p_value").head(12)
            if not race_sex_interactions.empty and "interaction_p_value" in race_sex_interactions.columns
            else race_sex_interactions,
            ".3f",
        ),
        "",
        "## 4. Terminal/acute pathway sensitivity",
        "",
        simple_markdown_table(
            terminal2024[["node_family", "total_node_mentions", "mention_intensity_per_100_lung_cancer_deaths"]],
            ".2f",
        ),
        "",
        simple_markdown_table(
            edge2024[
                [
                    "top20_edge_mentions",
                    "top20_terminal_or_acute_edge_mentions",
                    "top20_chronic_chronic_edge_mentions",
                    "terminal_or_acute_share_of_top20_edge_mentions_pct",
                ]
            ],
            ".2f",
        ),
        "",
        "## 5. Geographic enrichment",
        "",
        "Census Region contrasts:",
        "",
        simple_markdown_table(region_summary.head(8), ".2f"),
        "",
        "Urbanization contrasts:",
        "",
        simple_markdown_table(urban_summary.head(8), ".2f"),
        "",
        "## 6. COVID-era sensitivity",
        "",
        simple_markdown_table(
            covid_top[["period", "group", "u071_coded", "non_u071", "u071_minus_non_u071_pct_points"]]
            if not covid_top.empty and "u071_minus_non_u071_pct_points" in covid_top.columns
            else covid_top,
            ".2f",
        ),
        "",
        "## 7. Manuscript-level interpretation",
        "",
        "After these additions, the study is no longer a simple MCOD descriptive trend paper. It now has four defensible high-impact layers: decomposition of temporal change, exact chronic multimorbidity burden, race/ethnicity-sex heterogeneity, and sensitivity analyses separating chronic disease architecture from terminal respiratory pathways and COVID-era death-certification effects.",
        "",
        "The main residual limitation remains intrinsic to the data source: death-certificate co-mention is not equivalent to clinically ascertained comorbidity prevalence, and it cannot adjust for smoking, lung cancer stage, treatment, histology, survival time, or individual socioeconomic status.",
        "",
        "Generated figures: Figure21-Figure26.",
        "",
    ]
    (OUT / "P1_priority7_high_impact_enhancement_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_excel_bundle(tables: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(OUT / "P1_priority7_high_impact_enhancement_tables_v1.xlsx", engine="openpyxl") as writer:
        for name, df in tables.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    scan = scan_or_load()
    decomp = build_age_decomposition()
    burden, burden_std, burden_race = build_chronic_burden_outputs(scan)
    race_sex_std, race_sex_interactions = build_race_sex_outputs(scan)
    terminal_summary = build_terminal_chronic_sensitivity()
    region_summary, urban_summary = build_geographic_summary()
    covid_wide = build_covid_period_sensitivity(scan)

    plot_age_decomposition(decomp)
    plot_chronic_burden(burden, burden_std, burden_race)
    plot_race_sex(race_sex_std)
    plot_terminal_chronic(terminal_summary)
    plot_geographic(region_summary, urban_summary)
    plot_covid(covid_wide)

    write_report(
        decomp,
        burden,
        burden_std,
        burden_race,
        race_sex_std,
        race_sex_interactions,
        terminal_summary,
        region_summary,
        urban_summary,
        covid_wide,
        scan["audit"],
    )
    write_excel_bundle(
        {
            "age_decomposition": decomp,
            "chronic_burden": burden,
            "age_std_chronic_burden": burden_std,
            "race_chronic_burden": burden_race,
            "race_sex_age_std": race_sex_std,
            "race_sex_interactions": race_sex_interactions,
            "terminal_sensitivity": terminal_summary,
            "region_summary": region_summary,
            "urban_summary": urban_summary,
            "covid_period_sensitivity": covid_wide,
            "scan_audit": scan["audit"],
        }
    )
    print("Priority-7 high-impact enhancements complete.", flush=True)


if __name__ == "__main__":
    main()
