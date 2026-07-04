from __future__ import annotations

import itertools
import math
import sys
from collections import Counter
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
FIG = OUT / "figures"

YEARS = list(range(2018, 2025))
REFERENCE_YEAR = 2024
REFERENCE_RACE = "Non-Hispanic White"

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

NETWORK_LABELS = {
    "copd": "COPD",
    "ischemic_heart_disease": "IHD",
    "diabetes": "Diabetes",
    "heart_failure": "HF",
    "atrial_fibrillation": "AF",
    "cerebrovascular": "CVD",
    "ckd": "CKD",
    "ild": "ILD",
    "autoimmune_broad": "Autoimmune",
    "depression_anxiety": "Dep/anx",
    "serious_mental_illness": "SMI",
    "non_tobacco_substance_opioid": "Substance",
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
AGE_CODES_FOR_STANDARDIZATION = ["04", "05", "06", "07", "08", "09", "10", "11"]

RACE_ORDER = [
    "Non-Hispanic White",
    "Non-Hispanic Black",
    "Hispanic",
    "Non-Hispanic Asian",
    "Non-Hispanic AIAN",
    "Non-Hispanic NHOPI",
    "Non-Hispanic Multiracial",
    "Unknown/Not stated",
]

PRIMARY_RACES = [
    "Non-Hispanic White",
    "Non-Hispanic Black",
    "Hispanic",
    "Non-Hispanic Asian",
    "Non-Hispanic AIAN",
    "Non-Hispanic Multiracial",
]


def import_parser_helpers():
    scripts_dir = PROJECT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from parse_nchs_mcod_year import any_match, iter_records, load_groups, parse_record

    return any_match, iter_records, load_groups, parse_record


def label(group: str) -> str:
    return LABELS.get(group, group.replace("_", " ").title())


def parse_int(raw: str) -> int | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def race_ethnicity_from_line(line: str) -> tuple[str, str, str]:
    """Return harmonized race/ethnicity using 2018+ Hispanic origin and Race Recode 40.

    NCHS fixed-width locations are 1-based in documentation:
    Hispanic origin 484-486, Race Recode 40 489-490.
    """
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


def savefig(name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "svg"]:
        plt.savefig(FIG / f"{name}.{ext}", dpi=400, bbox_inches="tight")
    plt.close()


def simple_markdown_table(df: pd.DataFrame, floatfmt: str = ".2f") -> str:
    """Small dependency-free Markdown table writer.

    pandas.to_markdown needs the optional tabulate package, which is not
    installed in this workstation Python environment.
    """
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
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    header_line = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows]
    return "\n".join([header_line, sep_line, *body])


def wilson_ci(k: pd.Series | np.ndarray, n: pd.Series | np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    k_arr = np.asarray(k, dtype=float)
    n_arr = np.asarray(n, dtype=float)
    p = np.divide(k_arr, n_arr, out=np.full_like(k_arr, np.nan), where=n_arr > 0)
    denom = 1 + z**2 / n_arr
    center = (p + z**2 / (2 * n_arr)) / denom
    half = z * np.sqrt((p * (1 - p) + z**2 / (4 * n_arr)) / n_arr) / denom
    return center - half, center + half


def scan_mcod() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    any_match, iter_records, load_groups, parse_record = import_parser_helpers()
    groups = load_groups(priority=None)

    denominator = Counter()
    denominator_age = Counter()
    node_counts = Counter()
    node_age_counts = Counter()
    pair_counts = Counter()
    raw_race_codes = Counter()
    audit_rows = []

    for year in YEARS:
        zip_path = RAW / f"Mort{year}us.zip"
        if not zip_path.exists():
            raise FileNotFoundError(zip_path)

        total_deaths = 0
        us_resident_deaths = 0
        foreign_resident_deaths = 0
        lung_deaths = 0
        print(f"Scanning {year} from {zip_path.name}...", flush=True)

        for idx, line in enumerate(iter_records(zip_path), start=1):
            total_deaths += 1
            if idx % 500000 == 0:
                print(f"  {year}: processed {idx:,}; C34 resident deaths {lung_deaths:,}", flush=True)

            rec = parse_record(line)
            if rec["resident_status"] == "4":
                foreign_resident_deaths += 1
                continue
            us_resident_deaths += 1
            if not rec["ucd"].startswith("C34"):
                continue

            race_ethnicity, hispanic_origin, race40 = race_ethnicity_from_line(line)
            age12 = rec["age12"] if rec["age12"] in AGE12_LABELS else "12"
            lung_deaths += 1
            denominator[(year, race_ethnicity)] += 1
            denominator_age[(year, race_ethnicity, age12)] += 1
            raw_race_codes[(year, race_ethnicity, hispanic_origin, race40)] += 1

            present = []
            codes = rec["record_axis"]
            for group in groups:
                group_id = group["group"]
                if any(any_match(code, group["terms"]) for code in codes):
                    node_counts[(year, race_ethnicity, group_id)] += 1
                    node_age_counts[(year, race_ethnicity, age12, group_id)] += 1
                    present.append(group_id)

            for group_a, group_b in itertools.combinations(sorted(set(present)), 2):
                pair_counts[(year, race_ethnicity, group_a, group_b)] += 1

        audit_rows.append(
            {
                "year": year,
                "all_deaths_in_file": total_deaths,
                "all_deaths_us_residents": us_resident_deaths,
                "foreign_resident_deaths_excluded": foreign_resident_deaths,
                "underlying_cause_C34": lung_deaths,
                "race_ethnicity_denominator_sum": sum(count for (yr, _race), count in denominator.items() if yr == year),
            }
        )
        print(f"Finished {year}: resident C34 deaths {lung_deaths:,}", flush=True)

    denom_df = pd.DataFrame(
        [
            {"year": year, "race_ethnicity": race, "lung_cancer_ucd_deaths": count}
            for (year, race), count in sorted(denominator.items())
        ]
    )
    denom_age_df = pd.DataFrame(
        [
            {"year": year, "race_ethnicity": race, "age12": age12, "lung_cancer_ucd_deaths": count}
            for (year, race, age12), count in sorted(denominator_age.items())
        ]
    )
    nodes_df = pd.DataFrame(
        [
            {"year": year, "race_ethnicity": race, "group": group, "deaths": count}
            for (year, race, group), count in sorted(node_counts.items())
        ]
    )
    node_age_df = pd.DataFrame(
        [
            {"year": year, "race_ethnicity": race, "age12": age12, "group": group, "deaths": count}
            for (year, race, age12, group), count in sorted(node_age_counts.items())
        ]
    )
    pairs_df = pd.DataFrame(
        [
            {"year": year, "race_ethnicity": race, "group_a": group_a, "group_b": group_b, "co_mentioned_deaths": count}
            for (year, race, group_a, group_b), count in sorted(pair_counts.items())
        ]
    )
    race_code_df = pd.DataFrame(
        [
            {
                "year": year,
                "race_ethnicity": race,
                "hispanic_origin_code": hisp,
                "race_recode_40": race40,
                "lung_cancer_ucd_deaths": count,
            }
            for (year, race, hisp, race40), count in sorted(raw_race_codes.items())
        ]
    )
    audit_df = pd.DataFrame(audit_rows)
    return denom_df, denom_age_df, nodes_df, node_age_df, pairs_df, race_code_df, audit_df


def load_or_scan_mcod() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {
        "denominator": OUT / "P1_race_ethnicity_mcod_2018_2024_denominators.csv",
        "denominator_age": OUT / "P1_race_ethnicity_mcod_2018_2024_age_denominators.csv",
        "raw_nodes": OUT / "P1_race_ethnicity_mcod_2018_2024_node_counts_raw.csv",
        "raw_node_age": OUT / "P1_race_ethnicity_mcod_2018_2024_node_age_counts_raw.csv",
        "raw_pairs": OUT / "P1_race_ethnicity_mcod_2018_2024_pair_counts_raw.csv",
        "race_codes": OUT / "P1_race_ethnicity_raw_code_audit_2018_2024.csv",
        "audit": OUT / "P1_race_ethnicity_denominator_audit_2018_2024.csv",
    }
    if all(path.exists() and path.stat().st_size > 0 for path in paths.values()):
        print("Using cached 2018-2024 race/ethnicity scan outputs.", flush=True)
        return (
            pd.read_csv(paths["denominator"]),
            pd.read_csv(paths["denominator_age"], dtype={"age12": str}),
            pd.read_csv(paths["raw_nodes"]),
            pd.read_csv(paths["raw_node_age"], dtype={"age12": str}),
            pd.read_csv(paths["raw_pairs"]),
            pd.read_csv(paths["race_codes"], dtype={"hispanic_origin_code": str, "race_recode_40": str}),
            pd.read_csv(paths["audit"]),
        )
    return scan_mcod()


def add_node_denominators(nodes: pd.DataFrame, denominator: pd.DataFrame) -> pd.DataFrame:
    groups = sorted(nodes["group"].unique())
    scaffold = pd.MultiIndex.from_product(
        [YEARS, RACE_ORDER, groups],
        names=["year", "race_ethnicity", "group"],
    ).to_frame(index=False)
    out = scaffold.merge(nodes, on=["year", "race_ethnicity", "group"], how="left")
    out["deaths"] = out["deaths"].fillna(0).astype(int)
    out = out.merge(denominator, on=["year", "race_ethnicity"], how="left")
    out["lung_cancer_ucd_deaths"] = out["lung_cancer_ucd_deaths"].fillna(0).astype(int)
    out["proportion"] = np.where(out["lung_cancer_ucd_deaths"] > 0, out["deaths"] / out["lung_cancer_ucd_deaths"], np.nan)
    out["proportion_pct"] = out["proportion"] * 100
    out["label"] = out["group"].map(label)
    out["node_class"] = np.where(out["group"].isin(TERMINAL_ACUTE_NODES), "terminal_or_acute_pathway", "chronic_or_preexisting")
    lcl, ucl = wilson_ci(out["deaths"], out["lung_cancer_ucd_deaths"])
    out["proportion_lcl_pct"] = lcl * 100
    out["proportion_ucl_pct"] = ucl * 100
    return out.sort_values(["year", "race_ethnicity", "deaths"], ascending=[True, True, False])


def build_age_standardized(nodes_age: pd.DataFrame, denominator_age: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    ref = denominator_age[
        (denominator_age["year"] == REFERENCE_YEAR)
        & (denominator_age["age12"].isin(AGE_CODES_FOR_STANDARDIZATION))
    ].copy()
    ref = ref.groupby("age12", as_index=False)["lung_cancer_ucd_deaths"].sum()
    ref["reference_weight_2024_overall"] = ref["lung_cancer_ucd_deaths"] / ref["lung_cancer_ucd_deaths"].sum()
    weights = dict(zip(ref["age12"], ref["reference_weight_2024_overall"]))

    groups = sorted(nodes["group"].unique())
    scaffold = pd.MultiIndex.from_product(
        [YEARS, RACE_ORDER, groups, AGE_CODES_FOR_STANDARDIZATION],
        names=["year", "race_ethnicity", "group", "age12"],
    ).to_frame(index=False)
    age = scaffold.merge(nodes_age, on=["year", "race_ethnicity", "group", "age12"], how="left")
    age["deaths"] = age["deaths"].fillna(0).astype(int)
    age = age.merge(denominator_age, on=["year", "race_ethnicity", "age12"], how="left")
    age["lung_cancer_ucd_deaths"] = age["lung_cancer_ucd_deaths"].fillna(0).astype(int)
    age["age_specific_proportion"] = np.where(age["lung_cancer_ucd_deaths"] > 0, age["deaths"] / age["lung_cancer_ucd_deaths"], np.nan)
    age["reference_weight_2024_overall"] = age["age12"].map(weights)
    age["weighted_proportion"] = age["age_specific_proportion"] * age["reference_weight_2024_overall"]
    age["age_specific_variance"] = np.where(
        age["lung_cancer_ucd_deaths"] > 0,
        age["age_specific_proportion"] * (1 - age["age_specific_proportion"]) / age["lung_cancer_ucd_deaths"],
        np.nan,
    )
    age["weighted_variance"] = (age["reference_weight_2024_overall"] ** 2) * age["age_specific_variance"]

    std = (
        age.groupby(["year", "race_ethnicity", "group"], as_index=False)
        .agg(
            age_standardized_proportion=("weighted_proportion", "sum"),
            age_standardized_variance=("weighted_variance", "sum"),
            age_cells_with_denominator=("lung_cancer_ucd_deaths", lambda x: int((x > 0).sum())),
        )
    )
    std["age_standardized_se"] = np.sqrt(std["age_standardized_variance"])
    std["age_standardized_pct"] = std["age_standardized_proportion"] * 100
    std["age_standardized_lcl_pct"] = (std["age_standardized_proportion"] - 1.96 * std["age_standardized_se"]) * 100
    std["age_standardized_ucl_pct"] = (std["age_standardized_proportion"] + 1.96 * std["age_standardized_se"]) * 100
    std["age_standardized_lcl_pct"] = std["age_standardized_lcl_pct"].clip(lower=0)
    std["label"] = std["group"].map(label)
    std = std.merge(
        nodes[
            [
                "year",
                "race_ethnicity",
                "group",
                "deaths",
                "lung_cancer_ucd_deaths",
                "proportion_pct",
                "proportion_lcl_pct",
                "proportion_ucl_pct",
                "node_class",
            ]
        ],
        on=["year", "race_ethnicity", "group"],
        how="left",
    )
    return std.sort_values(["year", "race_ethnicity", "age_standardized_pct"], ascending=[True, True, False])


def build_disparity(std: pd.DataFrame) -> pd.DataFrame:
    y2024 = std[(std["year"] == REFERENCE_YEAR) & (std["race_ethnicity"].isin(PRIMARY_RACES))].copy()
    ref = y2024[y2024["race_ethnicity"] == REFERENCE_RACE][
        ["group", "age_standardized_proportion", "age_standardized_variance", "age_standardized_pct"]
    ].rename(
        columns={
            "age_standardized_proportion": "reference_age_standardized_proportion",
            "age_standardized_variance": "reference_age_standardized_variance",
            "age_standardized_pct": "reference_age_standardized_pct",
        }
    )
    out = y2024.merge(ref, on="group", how="left")
    out = out[out["race_ethnicity"] != REFERENCE_RACE].copy()
    out["absolute_difference_pct_points_vs_nh_white"] = (
        out["age_standardized_proportion"] - out["reference_age_standardized_proportion"]
    ) * 100
    out["absolute_difference_se"] = np.sqrt(
        out["age_standardized_variance"].fillna(0) + out["reference_age_standardized_variance"].fillna(0)
    )
    out["absolute_difference_lcl_pct_points"] = out["absolute_difference_pct_points_vs_nh_white"] - 1.96 * out["absolute_difference_se"] * 100
    out["absolute_difference_ucl_pct_points"] = out["absolute_difference_pct_points_vs_nh_white"] + 1.96 * out["absolute_difference_se"] * 100
    out["age_standardized_ratio_vs_nh_white"] = out["age_standardized_proportion"] / out["reference_age_standardized_proportion"]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio_se = np.sqrt(
            out["age_standardized_variance"] / (out["age_standardized_proportion"] ** 2)
            + out["reference_age_standardized_variance"] / (out["reference_age_standardized_proportion"] ** 2)
        )
    out["ratio_lcl"] = np.exp(np.log(out["age_standardized_ratio_vs_nh_white"]) - 1.96 * log_ratio_se)
    out["ratio_ucl"] = np.exp(np.log(out["age_standardized_ratio_vs_nh_white"]) + 1.96 * log_ratio_se)
    out = out.sort_values("absolute_difference_pct_points_vs_nh_white", ascending=False)
    return out


def fit_poisson_apc(part: pd.DataFrame) -> dict | None:
    part = part.sort_values("year").copy()
    part = part[part["lung_cancer_ucd_deaths"] > 0]
    if part.shape[0] < 5 or part["deaths"].sum() < 20:
        return None
    t = (part["year"] - part["year"].min()).astype(float).to_numpy()
    y = part["deaths"].astype(float).to_numpy()
    n = part["lung_cancer_ucd_deaths"].astype(float).to_numpy()
    x = sm.add_constant(t)
    fit = sm.GLM(y, x, family=sm.families.Poisson(), offset=np.log(n)).fit(cov_type="HC0")
    beta = float(fit.params[1])
    se = float(fit.bse[1])
    first = part.iloc[0]
    last = part.iloc[-1]
    return {
        "start_year": int(first["year"]),
        "end_year": int(last["year"]),
        "start_deaths": int(first["deaths"]),
        "end_deaths": int(last["deaths"]),
        "start_denominator": int(first["lung_cancer_ucd_deaths"]),
        "end_denominator": int(last["lung_cancer_ucd_deaths"]),
        "start_proportion_pct": first["deaths"] / first["lung_cancer_ucd_deaths"] * 100,
        "end_proportion_pct": last["deaths"] / last["lung_cancer_ucd_deaths"] * 100,
        "absolute_change_pct_points": last["deaths"] / last["lung_cancer_ucd_deaths"] * 100
        - first["deaths"] / first["lung_cancer_ucd_deaths"] * 100,
        "apc_pct_per_year": (math.exp(beta) - 1.0) * 100,
        "apc_lcl": (math.exp(beta - 1.96 * se) - 1.0) * 100,
        "apc_ucl": (math.exp(beta + 1.96 * se) - 1.0) * 100,
        "apc_p_value": float(fit.pvalues[1]),
    }


def build_trends(nodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    data = nodes[nodes["race_ethnicity"].isin(PRIMARY_RACES)].copy()
    for (race, group), part in data.groupby(["race_ethnicity", "group"]):
        trend = fit_poisson_apc(part)
        if trend is None:
            continue
        trend.update({"race_ethnicity": race, "group": group, "label": label(group)})
        rows.append(trend)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["apc_fdr_q_value"] = multipletests(out["apc_p_value"].fillna(1), method="fdr_bh")[1]
    return out.sort_values(["group", "race_ethnicity"])


def build_race_year_interactions(nodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    data = nodes[nodes["race_ethnicity"].isin(PRIMARY_RACES)].copy()
    data = data[data["lung_cancer_ucd_deaths"] > 0].copy()
    data["race_ethnicity"] = pd.Categorical(data["race_ethnicity"], categories=PRIMARY_RACES, ordered=False)
    data["year_centered"] = data["year"] - min(YEARS)
    for group, part in data.groupby("group", observed=True):
        if part["deaths"].sum() < 100:
            continue
        try:
            fit = smf.glm(
                "deaths ~ year_centered * C(race_ethnicity, Treatment(reference='Non-Hispanic White'))",
                data=part,
                family=sm.families.Poisson(),
                offset=np.log(part["lung_cancer_ucd_deaths"].astype(float)),
            ).fit(cov_type="HC0")
        except Exception as exc:
            rows.append({"group": group, "label": label(group), "model_status": f"failed: {exc}"})
            continue
        base_beta = float(fit.params.get("year_centered", np.nan))
        for race in PRIMARY_RACES:
            if race == REFERENCE_RACE:
                rows.append(
                    {
                        "group": group,
                        "label": label(group),
                        "race_ethnicity": race,
                        "model_status": "ok",
                        "apc_pct_per_year": (math.exp(base_beta) - 1.0) * 100 if not np.isnan(base_beta) else np.nan,
                        "interaction_beta_vs_nh_white": 0.0,
                        "interaction_rate_ratio_vs_nh_white": 1.0,
                        "interaction_p_value": np.nan,
                    }
                )
                continue
            term = f"year_centered:C(race_ethnicity, Treatment(reference='Non-Hispanic White'))[T.{race}]"
            beta = float(fit.params.get(term, np.nan))
            rows.append(
                {
                    "group": group,
                    "label": label(group),
                    "race_ethnicity": race,
                    "model_status": "ok",
                    "apc_pct_per_year": (math.exp(base_beta + beta) - 1.0) * 100 if not np.isnan(beta) else np.nan,
                    "interaction_beta_vs_nh_white": beta,
                    "interaction_rate_ratio_vs_nh_white": math.exp(beta) if not np.isnan(beta) else np.nan,
                    "interaction_p_value": float(fit.pvalues.get(term, np.nan)),
                }
            )
    out = pd.DataFrame(rows)
    if "interaction_p_value" in out.columns:
        mask = out["interaction_p_value"].notna()
        out.loc[mask, "interaction_fdr_q_value"] = multipletests(out.loc[mask, "interaction_p_value"], method="fdr_bh")[1]
    return out


def build_edge_enrichment(pairs: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    lookup = nodes[["year", "race_ethnicity", "group", "deaths", "proportion"]].rename(
        columns={"deaths": "node_deaths", "proportion": "node_proportion"}
    )
    edges = pairs.merge(
        lookup.rename(columns={"group": "group_a", "node_deaths": "node_a_deaths", "node_proportion": "node_a_proportion"}),
        on=["year", "race_ethnicity", "group_a"],
        how="left",
    ).merge(
        lookup.rename(columns={"group": "group_b", "node_deaths": "node_b_deaths", "node_proportion": "node_b_proportion"}),
        on=["year", "race_ethnicity", "group_b"],
        how="left",
    )
    den = nodes[["year", "race_ethnicity", "lung_cancer_ucd_deaths"]].drop_duplicates()
    edges = edges.merge(den, on=["year", "race_ethnicity"], how="left")
    n = edges["lung_cancer_ucd_deaths"].astype(float)
    a = edges["node_a_deaths"].astype(float)
    b = edges["node_b_deaths"].astype(float)
    c = edges["co_mentioned_deaths"].astype(float)
    expected = a * b / n
    edges["expected_if_independent"] = expected
    edges["observed_minus_expected"] = c - expected
    edges["lift_vs_independence"] = np.where(expected > 0, c / expected, np.nan)
    edges["jaccard_index"] = np.where((a + b - c) > 0, c / (a + b - c), np.nan)
    n10 = a - c
    n01 = b - c
    n00 = n - a - b + c
    denom_phi = np.sqrt(a * (n - a) * b * (n - b))
    edges["phi_correlation"] = np.where(denom_phi > 0, ((c * n00) - (n10 * n01)) / denom_phi, np.nan)
    edges["label_a"] = edges["group_a"].map(label)
    edges["label_b"] = edges["group_b"].map(label)
    edges["edge_label"] = edges["label_a"] + " + " + edges["label_b"]
    edges["edge_class"] = np.where(
        edges["group_a"].isin(TERMINAL_ACUTE_NODES) | edges["group_b"].isin(TERMINAL_ACUTE_NODES),
        "involves_terminal_or_acute_pathway",
        "chronic_chronic",
    )
    return edges.sort_values(["year", "race_ethnicity", "co_mentioned_deaths"], ascending=[True, True, False])


def build_chronic_core_edges(edges: pd.DataFrame) -> pd.DataFrame:
    y2024 = edges[(edges["year"] == REFERENCE_YEAR) & (edges["edge_class"] == "chronic_chronic")].copy()
    den = y2024[["race_ethnicity", "lung_cancer_ucd_deaths"]].drop_duplicates()
    rows = []
    for race, part in y2024.groupby("race_ethnicity"):
        denominator = int(den.loc[den["race_ethnicity"] == race, "lung_cancer_ucd_deaths"].iloc[0])
        threshold = max(20, denominator * 0.0025)
        keep = part[part["co_mentioned_deaths"] >= threshold].sort_values(
            ["co_mentioned_deaths", "lift_vs_independence"], ascending=False
        )
        keep = keep.head(12).copy()
        keep["race_specific_chronic_core_threshold"] = threshold
        rows.append(keep)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_heatmap(std: pd.DataFrame) -> None:
    y2024 = std[(std["year"] == REFERENCE_YEAR) & (std["race_ethnicity"].isin(PRIMARY_RACES))].copy()
    top_groups = (
        y2024.groupby("group", as_index=False)["deaths"].sum().sort_values("deaths", ascending=False).head(10)["group"].tolist()
    )
    races = [race for race in PRIMARY_RACES if y2024.loc[y2024["race_ethnicity"] == race, "lung_cancer_ucd_deaths"].max() >= 300]
    matrix = (
        y2024[y2024["group"].isin(top_groups) & y2024["race_ethnicity"].isin(races)]
        .pivot(index="group", columns="race_ethnicity", values="age_standardized_pct")
        .reindex(top_groups)
        .reindex(columns=races)
    )
    fig, ax = plt.subplots(figsize=(10.8, 7.8))
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(races)))
    ax.set_xticklabels(races, rotation=35, ha="right")
    ax.set_yticks(range(len(top_groups)))
    ax.set_yticklabels([label(g) for g in top_groups])
    ax.set_title("Age-standardized comorbidity co-mentions by race/ethnicity, 2024")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("% of lung cancer deaths")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8, color="black")
    fig.tight_layout()
    savefig("Figure17_P1_race_ethnicity_node_heatmap_2024")


def plot_disparity(disparity: pd.DataFrame) -> None:
    chronic = disparity[~disparity["group"].isin(TERMINAL_ACUTE_NODES)].copy()
    group_order = (
        chronic.groupby("group")["absolute_difference_pct_points_vs_nh_white"]
        .apply(lambda x: x.abs().max())
        .sort_values(ascending=False)
        .head(9)
        .index.tolist()
    )
    races = [r for r in PRIMARY_RACES if r != REFERENCE_RACE]
    plot_df = chronic[chronic["group"].isin(group_order) & chronic["race_ethnicity"].isin(races)].copy()
    y_positions = {group: i for i, group in enumerate(group_order)}
    colors = plt.cm.Set2(np.linspace(0, 1, len(races)))
    color_map = dict(zip(races, colors))

    fig, ax = plt.subplots(figsize=(10.8, 7.5))
    for race in races:
        part = plot_df[plot_df["race_ethnicity"] == race]
        ax.scatter(
            part["absolute_difference_pct_points_vs_nh_white"],
            [y_positions[g] for g in part["group"]],
            label=race.replace("Non-Hispanic ", "NH "),
            s=42,
            color=color_map[race],
            alpha=0.9,
        )
        for _, row in part.iterrows():
            ax.plot(
                [row["absolute_difference_lcl_pct_points"], row["absolute_difference_ucl_pct_points"]],
                [y_positions[row["group"]], y_positions[row["group"]]],
                color=color_map[race],
                alpha=0.35,
                linewidth=1,
            )
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(range(len(group_order)))
    ax.set_yticklabels([label(g) for g in group_order])
    ax.set_xlabel("Age-standardized difference vs non-Hispanic White, percentage points")
    ax.set_title("Race/ethnicity disparities in chronic comorbidity co-mentions, 2024")
    ax.legend(ncol=2, fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    savefig("Figure18_P1_race_ethnicity_disparity_forest_2024")


def plot_trends(std: pd.DataFrame) -> None:
    y2024 = std[(std["year"] == REFERENCE_YEAR) & (std["race_ethnicity"] == REFERENCE_RACE)].copy()
    top_groups = y2024.sort_values("age_standardized_pct", ascending=False).head(6)["group"].tolist()
    races = ["Non-Hispanic White", "Non-Hispanic Black", "Hispanic", "Non-Hispanic Asian", "Non-Hispanic AIAN"]
    colors = {
        "Non-Hispanic White": "#4c78a8",
        "Non-Hispanic Black": "#f58518",
        "Hispanic": "#54a24b",
        "Non-Hispanic Asian": "#b279a2",
        "Non-Hispanic AIAN": "#e45756",
    }
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.8), sharex=True)
    axes = axes.ravel()
    for ax, group in zip(axes, top_groups):
        part = std[(std["group"] == group) & (std["race_ethnicity"].isin(races))].copy()
        for race in races:
            series = part[part["race_ethnicity"] == race].sort_values("year")
            if series.empty:
                continue
            ax.plot(series["year"], series["age_standardized_pct"], marker="o", markersize=3.5, linewidth=1.7, label=race, color=colors[race])
        ax.set_title(label(group), fontsize=10)
        ax.set_ylabel("%")
        ax.grid(axis="y", alpha=0.25)
    for ax in axes[len(top_groups) :]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        [x.replace("Non-Hispanic ", "NH ") for x in labels],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=5,
        frameon=False,
    )
    fig.suptitle("Race/ethnicity-specific age-standardized trends, 2018-2024", y=0.98, fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    savefig("Figure19_P1_race_ethnicity_trends_2018_2024")


def plot_networks(chronic_edges: pd.DataFrame, nodes: pd.DataFrame) -> None:
    races = ["Non-Hispanic White", "Non-Hispanic Black", "Hispanic", "Non-Hispanic Asian", "Non-Hispanic AIAN"]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.5))
    axes = axes.ravel()
    for ax, race in zip(axes, races):
        part = chronic_edges[(chronic_edges["race_ethnicity"] == race) & (chronic_edges["year"] == REFERENCE_YEAR)].copy()
        part = part.sort_values(["co_mentioned_deaths", "lift_vs_independence"], ascending=False).head(10)
        if part.empty:
            ax.set_title(race)
            ax.text(0.5, 0.5, "Sparse edges\nbelow threshold", ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.axis("off")
            continue
        graph = nx.Graph()
        for _, row in part.iterrows():
            graph.add_edge(row["group_a"], row["group_b"], weight=float(row["co_mentioned_deaths"]), lift=float(row["lift_vs_independence"]))
        node_lookup = nodes[(nodes["year"] == REFERENCE_YEAR) & (nodes["race_ethnicity"] == race)].set_index("group")
        pos = nx.spring_layout(graph, seed=11, weight="weight", k=0.9)
        widths = [1.0 + 3.0 * min(graph[u][v]["lift"] / 5.0, 1.5) for u, v in graph.edges()]
        node_sizes = []
        for node in graph.nodes():
            pct = float(node_lookup.loc[node, "proportion_pct"]) if node in node_lookup.index else 1.0
            node_sizes.append(220 + pct * 32)
        nx.draw_networkx_edges(graph, pos, ax=ax, width=widths, edge_color="#666666", alpha=0.45)
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=node_sizes, node_color="#79b4a9", edgecolors="#1f3d3a", linewidths=0.8)
        nx.draw_networkx_labels(graph, pos, ax=ax, labels={n: NETWORK_LABELS.get(n, label(n)) for n in graph.nodes()}, font_size=8)
        ax.set_title(race.replace("Non-Hispanic ", "NH "))
        xs = np.array([xy[0] for xy in pos.values()])
        ys = np.array([xy[1] for xy in pos.values()])
        xpad = max((xs.max() - xs.min()) * 0.22, 0.18)
        ypad = max((ys.max() - ys.min()) * 0.22, 0.18)
        ax.set_xlim(xs.min() - xpad, xs.max() + xpad)
        ax.set_ylim(ys.min() - ypad, ys.max() + ypad)
        ax.axis("off")
    for ax in axes[len(races) :]:
        ax.axis("off")
    fig.suptitle("Race/ethnicity-specific chronic-core comorbidity networks, 2024", y=0.99, fontsize=13)
    fig.tight_layout()
    savefig("Figure20_P1_race_ethnicity_chronic_core_networks_2024")


def build_summary_report(
    denominator: pd.DataFrame,
    nodes: pd.DataFrame,
    std: pd.DataFrame,
    disparity: pd.DataFrame,
    trends: pd.DataFrame,
    chronic_edges: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    den2024 = denominator[denominator["year"] == REFERENCE_YEAR].sort_values("lung_cancer_ucd_deaths", ascending=False)
    top2024 = std[(std["year"] == REFERENCE_YEAR) & (std["race_ethnicity"].isin(PRIMARY_RACES))].copy()
    top_nodes = (
        top2024.groupby("group", as_index=False)["deaths"].sum().sort_values("deaths", ascending=False).head(8)["group"].tolist()
    )
    disparity_top = disparity[~disparity["group"].isin(TERMINAL_ACUTE_NODES)].copy()
    disparity_top = disparity_top.reindex(disparity_top["absolute_difference_pct_points_vs_nh_white"].abs().sort_values(ascending=False).index).head(12)
    trend_top = trends.reindex(trends["absolute_change_pct_points"].abs().sort_values(ascending=False).index).head(12) if not trends.empty else pd.DataFrame()
    edge_top = chronic_edges[chronic_edges["year"] == REFERENCE_YEAR].sort_values(
        ["co_mentioned_deaths", "lift_vs_independence"], ascending=False
    ).head(12)

    lines = [
        "# P1 race/ethnicity enrichment module, 2018-2024",
        "",
        "## Scope",
        "",
        "Population: U.S.-resident deaths with underlying cause C34 from NCHS Multiple Cause of Death public-use files, 2018-2024.",
        "Race/ethnicity harmonization: Hispanic origin 484-486 plus Race Recode 40 at 489-490; Hispanic origin takes precedence, and non-Hispanic race is grouped as White, Black, AIAN, Asian, NHOPI, or multiracial.",
        "Primary comparative models include non-Hispanic White, non-Hispanic Black, Hispanic, non-Hispanic Asian, non-Hispanic AIAN, and non-Hispanic multiracial groups. NHOPI and unknown/not stated strata are retained in raw denominator tables but not used as primary comparative strata because 2024 cell sizes are small.",
        "Age standardization: direct standardization to the 2024 overall lung-cancer-death age distribution using Age Recode 12 cells 15-24 through 85+.",
        "",
        "## Denominator audit",
        "",
        simple_markdown_table(audit, floatfmt=".0f"),
        "",
        "## 2024 lung cancer death denominator by race/ethnicity",
        "",
        simple_markdown_table(den2024, floatfmt=".0f"),
        "",
        "## Top 2024 age-standardized node burdens",
        "",
    ]
    burden_table = top2024[top2024["group"].isin(top_nodes)][
        ["race_ethnicity", "label", "deaths", "lung_cancer_ucd_deaths", "age_standardized_pct", "age_standardized_lcl_pct", "age_standardized_ucl_pct"]
    ].sort_values(["label", "age_standardized_pct"], ascending=[True, False])
    lines += [simple_markdown_table(burden_table, floatfmt=".2f"), ""]
    lines += ["## Largest 2024 chronic-node disparities vs non-Hispanic White", ""]
    lines += [
        simple_markdown_table(
            disparity_top[
            [
                "race_ethnicity",
                "label",
                "age_standardized_pct",
                "reference_age_standardized_pct",
                "absolute_difference_pct_points_vs_nh_white",
                "age_standardized_ratio_vs_nh_white",
            ]
            ],
            floatfmt=".2f",
        ),
        "",
    ]
    lines += ["## Largest race-specific 2018-2024 changes", ""]
    if trend_top.empty:
        lines += ["No trend rows available.", ""]
    else:
        lines += [
            simple_markdown_table(
                trend_top[
                [
                    "race_ethnicity",
                    "label",
                    "start_proportion_pct",
                    "end_proportion_pct",
                    "absolute_change_pct_points",
                    "apc_pct_per_year",
                    "apc_fdr_q_value",
                ]
                ],
                floatfmt=".3f",
            ),
            "",
        ]
    lines += ["## Top 2024 race-specific chronic-core edges", ""]
    if edge_top.empty:
        lines += ["No chronic-core edges met the race-specific thresholds.", ""]
    else:
        lines += [
            simple_markdown_table(
                edge_top[
                [
                    "race_ethnicity",
                    "edge_label",
                    "co_mentioned_deaths",
                    "lift_vs_independence",
                    "phi_correlation",
                    "race_specific_chronic_core_threshold",
                ]
                ],
                floatfmt=".3f",
            ),
            "",
        ]
    lines += [
        "## Manuscript interpretation",
        "",
        "This module turns the original death-certificate comorbidity network into a disparity-aware analysis. The key editorial value is not simply showing that comorbidities differ by race/ethnicity, but showing which differences persist after age standardization and which chronic-core edges define distinct multimorbidity architectures.",
        "",
        "Main limitations: death-certificate co-mention is not clinical disease prevalence; race/ethnicity is death-certificate reported and may be misclassified; smaller categories, especially AIAN and multiracial strata, require cautious interpretation; NHOPI is retained descriptively only; 2018-2024 is selected because all reporting areas had completed the 2003 death certificate transition and Race Recode 40 is available.",
        "",
        "Generated figures: Figure17 heatmap, Figure18 disparity forest, Figure19 race-specific trends, Figure20 chronic-core race-specific networks.",
        "",
    ]
    (OUT / "P1_race_ethnicity_module_2018_2024_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_excel_bundle(paths: dict[str, pd.DataFrame]) -> None:
    xlsx_path = OUT / "P1_race_ethnicity_module_2018_2024_tables_v1.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for sheet, df in paths.items():
            safe_sheet = sheet[:31]
            df.to_excel(writer, sheet_name=safe_sheet, index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    denominator, denominator_age, raw_nodes, raw_node_age, raw_pairs, race_codes, audit = load_or_scan_mcod()
    denominator.to_csv(OUT / "P1_race_ethnicity_mcod_2018_2024_denominators.csv", index=False)
    denominator_age.to_csv(OUT / "P1_race_ethnicity_mcod_2018_2024_age_denominators.csv", index=False)
    raw_nodes.to_csv(OUT / "P1_race_ethnicity_mcod_2018_2024_node_counts_raw.csv", index=False)
    raw_node_age.to_csv(OUT / "P1_race_ethnicity_mcod_2018_2024_node_age_counts_raw.csv", index=False)
    raw_pairs.to_csv(OUT / "P1_race_ethnicity_mcod_2018_2024_pair_counts_raw.csv", index=False)
    race_codes.to_csv(OUT / "P1_race_ethnicity_raw_code_audit_2018_2024.csv", index=False)
    audit.to_csv(OUT / "P1_race_ethnicity_denominator_audit_2018_2024.csv", index=False)

    nodes = add_node_denominators(raw_nodes, denominator)
    nodes.to_csv(OUT / "P1_race_ethnicity_node_burden_2018_2024.csv", index=False)
    std = build_age_standardized(raw_node_age, denominator_age, nodes)
    std.to_csv(OUT / "P1_race_ethnicity_age_standardized_node_burden_2018_2024.csv", index=False)
    disparity = build_disparity(std)
    disparity.to_csv(OUT / "P1_race_ethnicity_2024_disparity_vs_nh_white.csv", index=False)
    trends = build_trends(nodes)
    trends.to_csv(OUT / "P1_race_ethnicity_trend_models_2018_2024.csv", index=False)
    interactions = build_race_year_interactions(nodes)
    interactions.to_csv(OUT / "P1_race_ethnicity_year_interaction_models_2018_2024.csv", index=False)
    edges = build_edge_enrichment(raw_pairs, nodes)
    edges.to_csv(OUT / "P1_race_ethnicity_edge_enrichment_2018_2024.csv", index=False)
    chronic_edges = build_chronic_core_edges(edges)
    chronic_edges.to_csv(OUT / "P1_race_ethnicity_2024_chronic_core_edges.csv", index=False)

    plot_heatmap(std)
    plot_disparity(disparity)
    plot_trends(std)
    plot_networks(chronic_edges, nodes)

    build_summary_report(denominator, nodes, std, disparity, trends, chronic_edges, audit)
    write_excel_bundle(
        {
            "denominators": denominator,
            "node_burden": nodes,
            "age_standardized": std,
            "disparity_vs_white_2024": disparity,
            "trend_models": trends,
            "race_year_interactions": interactions,
            "edge_enrichment": edges,
            "chronic_core_2024": chronic_edges,
            "audit": audit,
        }
    )
    print("Race/ethnicity module complete.", flush=True)


if __name__ == "__main__":
    main()
