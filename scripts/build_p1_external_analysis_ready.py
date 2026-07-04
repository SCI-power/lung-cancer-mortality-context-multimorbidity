from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "external_public_data" / "raw"
PROCESSED = PROJECT / "external_public_data" / "processed"
OUT = PROJECT / "outputs"


def existing_columns(path: Path, reader: str) -> list[str]:
    if reader == "csv":
        return list(pd.read_csv(path, nrows=0).columns)
    if reader == "stata":
        return list(pd.read_stata(path, preserve_dtypes=False, convert_categoricals=False).head(0).columns)
    if reader == "xpt":
        return list(pd.read_sas(path, format="xport", encoding="utf-8").head(0).columns)
    raise ValueError(reader)


def yes_no(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NA, index=series.index, dtype="Float64")
    out[values == 1] = 1
    out[values == 2] = 0
    return out


def conditional_site_indicator(any_cancer: pd.Series, site_flag: pd.Series, extra_flag: pd.Series | None = None) -> pd.Series:
    cancer = pd.to_numeric(any_cancer, errors="coerce")
    site = pd.to_numeric(site_flag, errors="coerce")
    out = pd.Series(pd.NA, index=any_cancer.index, dtype="Float64")
    out[cancer == 2] = 0
    out[(cancer == 1) & (site == 2)] = 0
    out[(cancer == 1) & (site == 1)] = 1
    if extra_flag is not None:
        extra = pd.to_numeric(extra_flag, errors="coerce").fillna(0)
        out[extra == 1] = 1
        out[(out.isna()) & (extra == 0) & (cancer == 2)] = 0
    return out


def current_smoking_from_skip(ever_smoked: pd.Series, current_status: pd.Series) -> pd.Series:
    ever = pd.to_numeric(ever_smoked, errors="coerce")
    status = pd.to_numeric(current_status, errors="coerce")
    out = pd.Series(pd.NA, index=ever_smoked.index, dtype="Float64")
    out[ever == 2] = 0
    out[(ever == 1) & (status.isin([1, 2]))] = 1
    out[(ever == 1) & (status == 3)] = 0
    return out


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def read_csv_selected(path: Path, columns: list[str]) -> pd.DataFrame:
    available = [col for col in columns if col in existing_columns(path, "csv")]
    df = pd.read_csv(path, usecols=available)
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df[columns]


def build_nhis() -> pd.DataFrame:
    specs = [
        (2023, RAW / "nhis" / "2023" / "adult23csv" / "adult23.csv"),
        (2024, RAW / "nhis" / "2024" / "adult24csv" / "adult24.csv"),
    ]
    columns = [
        "AGEP_A", "SEX_A", "HISPALLP_A", "RACEALLP_A", "REGION", "URBRRL23",
        "PPSU", "PSTRAT", "WTFA_A", "LUNGCAN_A", "CANEV_A", "NUMCAN_A",
        "COPDEV_A", "DIBEV_A", "DIBPILL_A", "HYPEV_A", "CHDEV_A", "DEPEV_A", "DEPLEVEL_A",
        "SMKEV_A", "SMKNOW_A", "SMK30D_A", "BMICAT_A", "BMICATD_A",
    ]
    frames = []
    for year, path in specs:
        raw = read_csv_selected(path, columns)
        lung_cancer_history = conditional_site_indicator(raw["CANEV_A"], raw["LUNGCAN_A"])
        current_smoking = current_smoking_from_skip(raw["SMKEV_A"], raw["SMKNOW_A"])
        out = pd.DataFrame(
            {
                "source_dataset": "NHIS",
                "year": year,
                "person_id": pd.RangeIndex(1, len(raw) + 1).astype(str).map(lambda x: f"NHIS{year}_{x}"),
                "age": numeric(raw["AGEP_A"]),
                "sex": raw["SEX_A"],
                "race": raw["RACEALLP_A"],
                "hispanic_origin": raw["HISPALLP_A"],
                "region": raw["REGION"],
                "urban_rural": raw["URBRRL23"],
                "weight": numeric(raw["WTFA_A"]),
                "stratum": raw["PSTRAT"],
                "psu": raw["PPSU"],
                "lung_cancer_history": lung_cancer_history,
                "any_cancer_history": yes_no(raw["CANEV_A"]),
                "cancer_count": numeric(raw["NUMCAN_A"]),
                "copd": yes_no(raw["COPDEV_A"]),
                "diabetes": yes_no(raw["DIBEV_A"]),
                "diabetes_medication": yes_no(raw["DIBPILL_A"]),
                "hypertension": yes_no(raw["HYPEV_A"]),
                "coronary_heart_disease": yes_no(raw["CHDEV_A"]),
                "depression": yes_no(raw["DEPEV_A"]),
                "depression_level": numeric(raw["DEPLEVEL_A"]),
                "ever_smoked": yes_no(raw["SMKEV_A"]),
                "current_smoking": current_smoking,
                "current_smoking_proxy": yes_no(raw["SMKNOW_A"]),
                "smoked_past_30d": yes_no(raw["SMK30D_A"]),
                "bmi_category": raw["BMICAT_A"],
                "bmi_category_detailed": raw["BMICATD_A"],
            }
        )
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def icd_starts(series: pd.Series, prefixes: tuple[str, ...]) -> pd.Series:
    cleaned = series.fillna("").astype(str).str.upper().str.replace(".", "", regex=False).str.strip()
    return cleaned.map(lambda value: int(any(value.startswith(prefix) for prefix in prefixes)))


def build_meps() -> pd.DataFrame:
    h251_path = RAW / "meps" / "2023" / "h251dta" / "h251.dta"
    h249_path = RAW / "meps" / "2023" / "h249dta" / "h249.dta"
    h251_cols = [
        "DUPERSID", "AGE23X", "SEX", "RACETHX", "HISPANX", "REGION23", "PERWT23F",
        "VARSTR", "VARPSU", "CALUNG", "CANCERDX", "ASTHDX", "EMPHDX",
        "DIABDX_M18", "HIBPDX", "CHDDX", "PHQ242", "OFTSMK53", "TOTEXP23",
        "OBTOTV23", "OPTOTV23", "ERTOT23", "IPDIS23", "RXTOT23",
    ]
    available = [col for col in h251_cols if col in existing_columns(h251_path, "stata")]
    raw = pd.read_stata(h251_path, columns=available, preserve_dtypes=False, convert_categoricals=False)
    for col in h251_cols:
        if col not in raw.columns:
            raw[col] = pd.NA

    cond = pd.read_stata(h249_path, columns=["DUPERSID", "ICD10CDX"], preserve_dtypes=False, convert_categoricals=False)
    cond_flags = pd.DataFrame({"DUPERSID": cond["DUPERSID"]})
    cond_flags["condition_copd_icd"] = icd_starts(cond["ICD10CDX"], ("J40", "J41", "J42", "J43", "J44"))
    cond_flags["condition_lung_cancer_icd"] = icd_starts(cond["ICD10CDX"], ("C34",))
    cond_flags["condition_diabetes_icd"] = icd_starts(cond["ICD10CDX"], ("E10", "E11", "E12", "E13", "E14"))
    cond_flags["condition_cvd_icd"] = icd_starts(cond["ICD10CDX"], ("I20", "I21", "I22", "I23", "I24", "I25", "I48", "I50", "I60", "I61", "I62", "I63", "I64", "I65", "I66", "I67", "I68", "I69"))
    cond_flags["condition_ckd_icd"] = icd_starts(cond["ICD10CDX"], ("N18", "N19"))
    cond_flags["condition_mental_icd"] = icd_starts(cond["ICD10CDX"], ("F32", "F33", "F41"))
    cond_flags["condition_count"] = 1
    cond_person = cond_flags.groupby("DUPERSID", as_index=False).max()
    cond_count = cond_flags.groupby("DUPERSID", as_index=False)["condition_count"].sum()
    cond_person = cond_person.drop(columns=["condition_count"]).merge(cond_count, on="DUPERSID", how="left")

    merged = raw.merge(cond_person, on="DUPERSID", how="left")
    for col in ["condition_lung_cancer_icd", "condition_copd_icd", "condition_diabetes_icd", "condition_cvd_icd", "condition_ckd_icd", "condition_mental_icd", "condition_count"]:
        merged[col] = merged[col].fillna(0)
    lung_cancer_history = conditional_site_indicator(merged["CANCERDX"], merged["CALUNG"], merged["condition_lung_cancer_icd"])

    return pd.DataFrame(
        {
            "source_dataset": "MEPS",
            "year": 2023,
            "person_id": merged["DUPERSID"].astype(str),
            "age": numeric(merged["AGE23X"]),
            "sex": merged["SEX"],
            "race_ethnicity": merged["RACETHX"],
            "hispanic_origin": merged["HISPANX"],
            "region": merged["REGION23"],
            "weight": numeric(merged["PERWT23F"]),
            "stratum": merged["VARSTR"],
            "psu": merged["VARPSU"],
            "lung_cancer_history": lung_cancer_history,
            "any_cancer_history": yes_no(merged["CANCERDX"]),
            "asthma": yes_no(merged["ASTHDX"]),
            "emphysema": yes_no(merged["EMPHDX"]),
            "diabetes": yes_no(merged["DIABDX_M18"]),
            "hypertension": yes_no(merged["HIBPDX"]),
            "coronary_heart_disease": yes_no(merged["CHDDX"]),
            "phq2_variable": numeric(merged["PHQ242"]),
            "smoking_variable": numeric(merged["OFTSMK53"]),
            "total_expenditure": numeric(merged["TOTEXP23"]),
            "office_visits": numeric(merged["OBTOTV23"]),
            "outpatient_visits": numeric(merged["OPTOTV23"]),
            "er_visits": numeric(merged["ERTOT23"]),
            "inpatient_discharges": numeric(merged["IPDIS23"]),
            "rx_fills": numeric(merged["RXTOT23"]),
            "condition_lung_cancer_icd": merged["condition_lung_cancer_icd"].astype(int),
            "condition_copd_icd": merged["condition_copd_icd"].astype(int),
            "condition_diabetes_icd": merged["condition_diabetes_icd"].astype(int),
            "condition_cvd_icd": merged["condition_cvd_icd"].astype(int),
            "condition_ckd_icd": merged["condition_ckd_icd"].astype(int),
            "condition_mental_icd": merged["condition_mental_icd"].astype(int),
            "condition_count": numeric(merged["condition_count"]),
        }
    )


def read_xpt_selected(path: Path, columns: list[str]) -> pd.DataFrame:
    df = pd.read_sas(path, format="xport", encoding="utf-8")
    keep = [col for col in columns if col in df.columns]
    out = df[keep].copy()
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out[columns]


def build_nhanes_cycle(cycle: str, suffix: str) -> pd.DataFrame:
    base = RAW / "nhanes" / cycle
    modules = {
        "DEMO": ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3", "WTINT2YR", "WTMEC2YR", "SDMVSTRA", "SDMVPSU"],
        "MCQ": ["SEQN", "MCQ220", "MCQ230A", "MCQ230B", "MCQ230C", "MCQ160P", "MCQ160G", "MCQ010", "MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"],
        "BPQ": ["SEQN", "BPQ020"],
        "DIQ": ["SEQN", "DIQ010"],
        "KIQ_U": ["SEQN", "KIQ022"],
        "DPQ": ["SEQN", "DPQ010", "DPQ020", "DPQ030", "DPQ040", "DPQ050", "DPQ060", "DPQ070", "DPQ080", "DPQ090"],
        "SMQ": ["SEQN", "SMQ020", "SMQ040"],
        "BMX": ["SEQN", "BMXBMI"],
        "GHB": ["SEQN", "LBXGH"],
        "BIOPRO": ["SEQN", "LBXSCR", "LBXSC3SI"],
    }
    merged: pd.DataFrame | None = None
    for module, columns in modules.items():
        path = base / f"{module}_{suffix}.XPT"
        part = read_xpt_selected(path, columns)
        merged = part if merged is None else merged.merge(part, on="SEQN", how="outer")
    assert merged is not None
    dpq_cols = [f"DPQ{i:03d}" for i in range(10, 100, 10) if f"DPQ{i:03d}" in merged.columns]
    dpq_score = merged[dpq_cols].apply(pd.to_numeric, errors="coerce").where(lambda x: x <= 3).sum(axis=1, min_count=1)
    copd_source = merged["MCQ160P"] if "MCQ160P" in merged.columns and merged["MCQ160P"].notna().any() else merged["MCQ160G"]
    return pd.DataFrame(
        {
            "source_dataset": "NHANES",
            "cycle": cycle,
            "person_id": merged["SEQN"].astype("Int64").astype(str),
            "age": numeric(merged["RIDAGEYR"]),
            "sex": merged["RIAGENDR"],
            "race_ethnicity": merged["RIDRETH3"],
            "interview_weight": numeric(merged["WTINT2YR"]),
            "mec_weight": numeric(merged["WTMEC2YR"]),
            "stratum": merged["SDMVSTRA"],
            "psu": merged["SDMVPSU"],
            "any_cancer_history": yes_no(merged["MCQ220"]),
            "cancer_type_1": merged["MCQ230A"],
            "cancer_type_2": merged["MCQ230B"],
            "cancer_type_3": merged["MCQ230C"],
            "asthma": yes_no(merged["MCQ010"]),
            "copd_or_emphysema": yes_no(copd_source),
            "heart_failure": yes_no(merged["MCQ160B"]),
            "coronary_heart_disease": yes_no(merged["MCQ160C"]),
            "angina": yes_no(merged["MCQ160D"]),
            "myocardial_infarction": yes_no(merged["MCQ160E"]),
            "stroke": yes_no(merged["MCQ160F"]),
            "hypertension": yes_no(merged["BPQ020"]),
            "diabetes": yes_no(merged["DIQ010"]),
            "kidney_disease": yes_no(merged["KIQ022"]),
            "ever_smoked": yes_no(merged["SMQ020"]),
            "current_smoking": current_smoking_from_skip(merged["SMQ020"], merged["SMQ040"]),
            "bmi": numeric(merged["BMXBMI"]),
            "hba1c": numeric(merged["LBXGH"]),
            "serum_creatinine": numeric(merged["LBXSCR"]),
            "bicarbonate": numeric(merged["LBXSC3SI"]),
            "phq9_score": dpq_score,
        }
    )


def build_nhanes() -> pd.DataFrame:
    return pd.concat(
        [
            build_nhanes_cycle("2017-2018", "J"),
            build_nhanes_cycle("2021-2023", "L"),
        ],
        ignore_index=True,
    )


def dictionary() -> pd.DataFrame:
    rows = [
        ("lung_cancer_history", "Indicator for self-reported/public-use lung cancer history when available."),
        ("condition_lung_cancer_icd", "MEPS condition-file ICD-10 C34 lung cancer cluster."),
        ("any_cancer_history", "Indicator for self-reported/public-use any cancer history."),
        ("copd", "NHIS self-reported COPD indicator."),
        ("copd_or_emphysema", "NHANES COPD/emphysema/chronic bronchitis or legacy emphysema indicator."),
        ("condition_copd_icd", "MEPS condition-file ICD-10 COPD/chronic bronchitis/emphysema cluster."),
        ("condition_diabetes_icd", "MEPS condition-file ICD-10 diabetes cluster."),
        ("diabetes", "Self-reported diabetes indicator where available."),
        ("condition_cvd_icd", "MEPS condition-file ICD-10 cardiovascular/stroke cluster."),
        ("condition_ckd_icd", "MEPS condition-file ICD-10 chronic kidney disease cluster."),
        ("phq9_score", "NHANES PHQ-9 sum using valid 0-3 item responses."),
        ("weight", "Dataset-specific survey weight for NHIS/MEPS."),
        ("mec_weight", "NHANES MEC weight for exam/lab analyses."),
    ]
    return pd.DataFrame(rows, columns=["standard_variable", "definition"])


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    nhis = build_nhis()
    meps = build_meps()
    nhanes = build_nhanes()
    dict_df = dictionary()

    nhis.to_csv(PROCESSED / "nhis_2023_2024_analysis_ready.csv", index=False, encoding="utf-8-sig")
    meps.to_csv(PROCESSED / "meps_2023_analysis_ready.csv", index=False, encoding="utf-8-sig")
    nhanes.to_csv(PROCESSED / "nhanes_2017_2023_analysis_ready.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUT / "P1_external_analysis_ready_dictionary_v1.xlsx", engine="openpyxl") as writer:
        pd.DataFrame(
            [
                {"dataset": "NHIS", "rows": len(nhis), "columns": len(nhis.columns), "file": str(PROCESSED / "nhis_2023_2024_analysis_ready.csv")},
                {"dataset": "MEPS", "rows": len(meps), "columns": len(meps.columns), "file": str(PROCESSED / "meps_2023_analysis_ready.csv")},
                {"dataset": "NHANES", "rows": len(nhanes), "columns": len(nhanes.columns), "file": str(PROCESSED / "nhanes_2017_2023_analysis_ready.csv")},
            ]
        ).to_excel(writer, sheet_name="files", index=False)
        dict_df.to_excel(writer, sheet_name="dictionary", index=False)
    print(f"NHIS rows: {len(nhis)}")
    print(f"MEPS rows: {len(meps)}")
    print(f"NHANES rows: {len(nhanes)}")


if __name__ == "__main__":
    main()

