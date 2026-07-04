from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "external_public_data" / "raw"
OUT = PROJECT / "outputs"


DATA_FILES = [
    {
        "dataset": "NHIS",
        "cycle": "2024",
        "label": "Sample Adult",
        "path": RAW / "nhis" / "2024" / "adult24csv" / "adult24.csv",
        "reader": "csv",
        "analysis_role": "Main clinical-survey triangulation for lung cancer history and self-reported chronic conditions.",
    },
    {
        "dataset": "NHIS",
        "cycle": "2023",
        "label": "Sample Adult",
        "path": RAW / "nhis" / "2023" / "adult23csv" / "adult23.csv",
        "reader": "csv",
        "analysis_role": "Prior-year replication for public-use lung cancer and chronic condition variables.",
    },
    {
        "dataset": "MEPS",
        "cycle": "2023",
        "label": "HC-251 Full-Year Consolidated",
        "path": RAW / "meps" / "2023" / "h251dta" / "h251.dta",
        "reader": "stata",
        "analysis_role": "Person-level chronic disease, lung cancer indicator, utilization and expenditure triangulation.",
    },
    {
        "dataset": "MEPS",
        "cycle": "2023",
        "label": "HC-249 Medical Conditions",
        "path": RAW / "meps" / "2023" / "h249dta" / "h249.dta",
        "reader": "stata",
        "analysis_role": "Condition-file ICD-10 cluster triangulation; merge to HC-251 by DUPERSID.",
    },
]

for cycle, suffix in [("2021-2023", "L"), ("2017-2018", "J")]:
    for module in ["DEMO", "MCQ", "SMQ", "BPQ", "DIQ", "KIQ_U", "DPQ", "BMX", "GHB", "BIOPRO"]:
        DATA_FILES.append(
            {
                "dataset": "NHANES",
                "cycle": cycle,
                "label": module,
                "path": RAW / "nhanes" / cycle / f"{module}_{suffix}.XPT",
                "reader": "xpt",
                "analysis_role": "Biomarker/exposure context for premortem burden; not lung-cancer-specific primary analysis.",
            }
        )


KEY_VARIABLES = {
    "NHIS": [
        ("identity_design", "AGEP_A", "Age"),
        ("identity_design", "SEX_A", "Sex"),
        ("identity_design", "HISPALLP_A", "Hispanic origin"),
        ("identity_design", "RACEALLP_A", "Race"),
        ("identity_design", "PPSU", "Pseudo-PSU"),
        ("identity_design", "PSTRAT", "Pseudo-stratum"),
        ("identity_design", "WTFA_A", "Sample adult weight"),
        ("lung_cancer", "LUNGCAN_A", "Ever had lung cancer"),
        ("cancer", "CANEV_A", "Ever had cancer"),
        ("cancer", "NUMCAN_A", "Number of cancers"),
        ("respiratory", "COPDEV_A", "Ever had COPD"),
        ("cardiometabolic", "DIBPILL_A", "Diabetes pill use"),
        ("cardiometabolic", "HYPEV_A", "Ever had hypertension"),
        ("cardiometabolic", "CHDEV_A", "Ever had coronary heart disease"),
        ("mental_health", "DEPEV_A", "Ever had depression"),
        ("mental_health", "DEPLEVEL_A", "Depression symptom level"),
        ("smoking", "SMKEV_A", "Ever smoked"),
        ("smoking", "SMKNOW_A", "Current smoking"),
        ("smoking", "SMK30D_A", "Smoking in past 30 days"),
        ("anthropometry", "BMICAT_A", "BMI category"),
        ("anthropometry", "BMICATD_A", "Detailed BMI category"),
    ],
    "MEPS": [
        ("identity_design", "DUPERSID", "Person identifier"),
        ("identity_design", "PERWT23F", "Person weight"),
        ("identity_design", "VARSTR", "Variance stratum"),
        ("identity_design", "VARPSU", "Variance PSU"),
        ("lung_cancer", "CALUNG", "Lung cancer condition flag"),
        ("cancer", "CANCERDX", "Cancer diagnosis flag"),
        ("respiratory", "ASTHDX", "Asthma diagnosis flag"),
        ("cardiometabolic", "DIABDX_M18", "Diabetes diagnosis flag"),
        ("cardiometabolic", "HIBPDX", "Hypertension diagnosis flag"),
        ("cardiometabolic", "CHDDX", "Coronary heart disease diagnosis flag"),
        ("mental_health", "PHQ242", "PHQ-2 item/score variable"),
        ("smoking", "OFTSMK53", "Smoking-related variable"),
        ("condition_file", "ICD10CDX", "ICD-10 condition code"),
        ("condition_file", "CONDIDX", "Condition identifier"),
        ("condition_file", "CONDN", "Condition number"),
    ],
    "NHANES": [
        ("identity_design", "SEQN", "Respondent sequence number"),
        ("identity_design", "RIDAGEYR", "Age"),
        ("identity_design", "RIAGENDR", "Sex"),
        ("identity_design", "RIDRETH3", "Race/ethnicity with NH Asian"),
        ("identity_design", "WTINT2YR", "Interview weight"),
        ("identity_design", "WTMEC2YR", "MEC exam weight"),
        ("identity_design", "SDMVSTRA", "Masked variance stratum"),
        ("identity_design", "SDMVPSU", "Masked variance PSU"),
        ("cancer", "MCQ220", "Ever told had cancer or malignancy"),
        ("cancer", "MCQ230A", "First cancer type"),
        ("cancer", "MCQ230B", "Second cancer type"),
        ("cancer", "MCQ230C", "Third cancer type"),
        ("respiratory", "MCQ010", "Ever had asthma"),
        ("respiratory", "MCQ160P|MCQ160G", "COPD/emphysema/chronic bronchitis in 2021-2023, legacy emphysema variable in older cycles"),
        ("cardiometabolic", "MCQ160B", "Congestive heart failure"),
        ("cardiometabolic", "MCQ160C", "Coronary heart disease"),
        ("cardiometabolic", "MCQ160D", "Angina"),
        ("cardiometabolic", "MCQ160E", "Heart attack"),
        ("cardiometabolic", "MCQ160F", "Stroke"),
        ("cardiometabolic", "BPQ020", "Ever told had hypertension"),
        ("cardiometabolic", "DIQ010", "Ever told had diabetes"),
        ("kidney", "KIQ022", "Ever told weak/failing kidneys"),
        ("mental_health", "DPQ010", "Little interest"),
        ("mental_health", "DPQ020", "Feeling down"),
        ("mental_health", "DPQ090", "Self-harm thoughts"),
        ("smoking", "SMQ020", "Smoked at least 100 cigarettes"),
        ("smoking", "SMQ040", "Now smoke cigarettes"),
        ("anthropometry", "BMXBMI", "BMI"),
        ("biomarker", "LBXGH", "HbA1c"),
        ("biomarker", "LBXSCR", "Serum creatinine"),
        ("biomarker", "LBXSC3SI", "Bicarbonate"),
    ],
}


ANALYSIS_PLAN = [
    {
        "priority": 1,
        "external_dataset": "NHIS 2023-2024",
        "analysis_name": "Public-use lung-cancer-history chronic-condition profile",
        "population": "Sample adults; primary stratum LUNGCAN_A yes, secondary all-cancer history sensitivity.",
        "methods": "Survey-weighted prevalence/risk-difference table by age, sex, race/ethnicity; compare COPD, diabetes, CHD, hypertension, depression and smoking.",
        "triangulation_value": "Tests whether MCOD chronic-core signals are concordant with premortem self-reported disease burden.",
        "main_limitation": "Public-use lung-cancer-history sample can be sparse; pooled 2023-2024 and all-cancer sensitivity are needed.",
    },
    {
        "priority": 2,
        "external_dataset": "MEPS HC-251 + HC-249 2023",
        "analysis_name": "Health-care burden and condition-file validation",
        "population": "MEPS respondents with CALUNG/CANCERDX and condition-file ICD-10 clusters after merge by DUPERSID.",
        "methods": "Weighted utilization/cost and condition-cluster prevalence; ICD-10 condition-file sensitivity for COPD, diabetes, cardiovascular and kidney disease.",
        "triangulation_value": "Adds health-care-use relevance to the MCOD co-mention network and separates clinical burden from death-certification context.",
        "main_limitation": "Cancer site granularity may still be limited; Python should use Stata files because 2023 SSP files are SAS CPORT.",
    },
    {
        "priority": 3,
        "external_dataset": "NHANES 2017-2018 and 2021-2023",
        "analysis_name": "Biomarker/exposure context for chronic disease signals",
        "population": "Adults, preferably cancer-history subgroup where sample size allows; otherwise all adults for contextual burden ranking.",
        "methods": "Survey-weighted biomarker and questionnaire profiles: smoking, HbA1c, creatinine/eGFR proxy, BMI, PHQ, hypertension, diabetes.",
        "triangulation_value": "Provides objective/exposure context for cardiometabolic, kidney, smoking and depression nodes.",
        "main_limitation": "Not designed to estimate lung-cancer-specific multimorbidity; use as contextual validation, not primary lung cancer cohort.",
    },
]


def csv_shape(path: Path) -> tuple[int, int, list[str]]:
    columns = list(pd.read_csv(path, nrows=0).columns)
    rows = 0
    for chunk in pd.read_csv(path, usecols=[columns[0]], chunksize=250_000):
        rows += len(chunk)
    return rows, len(columns), columns


def read_shape(path: Path, reader: str) -> tuple[int | None, int | None, list[str], str]:
    if not path.exists():
        return None, None, [], "missing"
    try:
        if reader == "csv":
            rows, cols, columns = csv_shape(path)
        elif reader == "stata":
            df = pd.read_stata(path, preserve_dtypes=False, convert_categoricals=False)
            rows, cols, columns = len(df), len(df.columns), list(df.columns)
        elif reader == "xpt":
            df = pd.read_sas(path, format="xport", encoding="utf-8")
            rows, cols, columns = len(df), len(df.columns), list(df.columns)
        else:
            return None, None, [], f"unsupported reader: {reader}"
        return rows, cols, columns, "readable"
    except Exception as exc:
        return None, None, [], f"read_error: {type(exc).__name__}: {exc}"


def profile_files() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    rows = []
    column_cache: dict[str, list[str]] = {}
    for spec in DATA_FILES:
        rows_count, cols_count, columns, status = read_shape(spec["path"], spec["reader"])
        key = f"{spec['dataset']}|{spec['cycle']}|{spec['label']}"
        column_cache[key] = columns
        rows.append(
            {
                "dataset": spec["dataset"],
                "cycle": spec["cycle"],
                "label": spec["label"],
                "reader": spec["reader"],
                "status": status,
                "rows": rows_count,
                "columns": cols_count,
                "file_size_mb": round(spec["path"].stat().st_size / 1024 / 1024, 3) if spec["path"].exists() else None,
                "path": str(spec["path"]),
                "analysis_role": spec["analysis_role"],
            }
        )
    return pd.DataFrame(rows), column_cache


def variable_availability(column_cache: dict[str, list[str]]) -> pd.DataFrame:
    grouped_columns: dict[tuple[str, str], dict[str, list[str]]] = {}
    for key, columns in column_cache.items():
        dataset, cycle, label = key.split("|", 2)
        grouped_columns.setdefault((dataset, cycle), {})
        for col in columns:
            grouped_columns[(dataset, cycle)].setdefault(col.upper(), []).append(label)

    rows = []
    for (dataset, cycle), upper_map in grouped_columns.items():
        for group, variable, description in KEY_VARIABLES.get(dataset, []):
            candidate_vars = [item.strip().upper() for item in variable.split("|")]
            matched_labels = []
            matched_columns = []
            for candidate_var in candidate_vars:
                if candidate_var in upper_map:
                    matched_labels.extend(upper_map[candidate_var])
                    matched_columns.append(candidate_var)
            rows.append(
                {
                    "dataset": dataset,
                    "cycle": cycle,
                    "analysis_level": "merged_cycle" if dataset in {"MEPS", "NHANES"} else "single_file",
                    "variable_group": group,
                    "variable": variable,
                    "available": bool(matched_labels),
                    "matched_column": ";".join(sorted(set(matched_columns))),
                    "source_file_or_module": ";".join(sorted(set(matched_labels))),
                    "description": description,
                }
            )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return ""
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df[columns].iterrows():
        values = [str(row[col]).replace("\n", " ") for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(file_profile: pd.DataFrame, var_profile: pd.DataFrame, plan: pd.DataFrame) -> None:
    readable = int((file_profile["status"] == "readable").sum())
    total = len(file_profile)
    missing_critical = var_profile[(var_profile["available"] == False) & (var_profile["variable_group"].isin(["identity_design", "lung_cancer", "cancer"]))]
    gap_columns = ["dataset", "cycle", "analysis_level", "variable_group", "variable", "description"]
    gap_text = (
        markdown_table(missing_critical, gap_columns)
        if not missing_critical.empty
        else "No critical identity/cancer variable gaps detected in the current profiling pass."
    )
    body = f"""# P1 no-application external public data readiness report

## Readiness summary

- Readable files: {readable}/{total}
- Primary no-application datasets prepared: NHIS 2023-2024, MEPS 2023, NHANES 2017-2018 and 2021-2023
- MEPS analysis files: Stata `.dta` files are the Python-readable source; SAS transport `.ssp` files are retained as original archives but are SAS CPORT for 2023.
- Intended use: contextual triangulation, not causal inference and not conversion of MCOD death-certificate co-mentions into clinical prevalence.

## Critical variable gaps to check

{gap_text}

## Execution order

1. NHIS pooled 2023-2024: estimate lung-cancer-history chronic condition profile.
2. MEPS 2023: merge HC-249 condition file to HC-251 person file by `DUPERSID`; estimate clinical and utilization burden.
3. NHANES: create adult biomarker/exposure context table and cancer-history sensitivity if sample size permits.

## Interpretation guardrail

External databases contextualize MCOD signals. They should be reported as concordance, discordance or contextual burden evidence, not as proof that death-certificate co-mentions equal clinical comorbidities.
"""
    (OUT / "P1_external_public_data_readiness_report.md").write_text(body, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    file_profile, column_cache = profile_files()
    var_profile = variable_availability(column_cache)
    plan = pd.DataFrame(ANALYSIS_PLAN)

    file_profile.to_csv(OUT / "P1_external_public_data_file_profile.csv", index=False, encoding="utf-8-sig")
    var_profile.to_csv(OUT / "P1_external_public_data_variable_availability.csv", index=False, encoding="utf-8-sig")
    plan.to_csv(OUT / "P1_external_public_data_analysis_plan.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUT / "P1_external_public_data_readiness_tables_v1.xlsx", engine="openpyxl") as writer:
        file_profile.to_excel(writer, sheet_name="file_profile", index=False)
        var_profile.to_excel(writer, sheet_name="variable_availability", index=False)
        plan.to_excel(writer, sheet_name="analysis_plan", index=False)
    write_report(file_profile, var_profile, plan)
    print(f"Readable files: {(file_profile['status'] == 'readable').sum()}/{len(file_profile)}")
    print(OUT / "P1_external_public_data_readiness_tables_v1.xlsx")


if __name__ == "__main__":
    main()

