from __future__ import annotations

import argparse
import csv
import textwrap
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "outputs"
FIG = OUT / "figures"
EXT = PROJECT / "external_public_data"


DATABASE_MANIFEST = [
    {
        "dataset": "NHIS",
        "analysis_role": "Main no-application clinical/survey triangulation for self-reported cancer history and chronic conditions.",
        "public_access": "No application for public-use files; restricted geography requires RDC and is excluded.",
        "primary_years": "2023-2024",
        "target_files": "Sample Adult public-use CSV files",
        "download_url": "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2024/adult24csv.zip",
        "backup_url": "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2023/adult23csv.zip",
        "key_variable_domains": "Cancer history; COPD/emphysema/chronic bronchitis; diabetes; heart disease; stroke; kidney disease; smoking; race/ethnicity; sex; survey weights.",
        "triangulation_use": "Estimate whether major MCOD chronic signals are concordant with premortem/self-reported chronic disease burden in adults with cancer history.",
        "main_limitation": "Cancer type and lung-cancer-specific sample size may be limited; self-report; noninstitutionalized living population, not decedents.",
    },
    {
        "dataset": "MEPS",
        "analysis_role": "Main no-application health-care-use and condition-file triangulation.",
        "public_access": "No application for Household Component public-use files.",
        "primary_years": "2023",
        "target_files": "HC-251 Full Year Consolidated File and HC-249 Medical Conditions File, SAS transport ZIP.",
        "download_url": "https://meps.ahrq.gov/mepsweb/data_files/pufs/h251/h251ssp.zip",
        "backup_url": "https://meps.ahrq.gov/mepsweb/data_files/pufs/h249/h249ssp.zip",
        "key_variable_domains": "Cancer priority condition; COPD/chronic bronchitis/emphysema; heart disease; stroke; diabetes; hypertension; condition ICD/CCS-style groupings; expenditure/utilization; survey weights.",
        "triangulation_use": "Quantify chronic-condition clusters and health-care burden among respondents with cancer-related priority conditions.",
        "main_limitation": "Cancer site specificity can be limited in public-use files; condition reporting is household/survey based.",
    },
    {
        "dataset": "NHANES",
        "analysis_role": "Biomarker and exposure context validation, not lung-cancer-specific main validation.",
        "public_access": "No application for public-use questionnaire, exam, and lab XPT files.",
        "primary_years": "2017-2018 and 2021-2023",
        "target_files": "DEMO, MCQ, SMQ, BPQ, DIQ, KIQ_U, DPQ, BMX, GHB, BIOPRO where available.",
        "download_url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DEMO_L.xpt",
        "backup_url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.xpt",
        "key_variable_domains": "Cancer history; smoking; COPD/chronic bronchitis/emphysema; diabetes; kidney disease; depression; BMI; HbA1c; creatinine/eGFR; race/ethnicity; sex; survey weights.",
        "triangulation_use": "Contextualize smoking, cardiometabolic, renal, and inflammatory/metabolic risk profiles behind high MCOD signals.",
        "main_limitation": "Very small lung-cancer history sample; use as mechanism/context layer rather than primary lung-cancer comorbidity estimate.",
    },
]


NHANES_FILES = [
    ("NHANES", "2021-2023", "DEMO_L.XPT", "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DEMO_L.xpt"),
    ("NHANES", "2021-2023", "MCQ_L.XPT", "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/MCQ_L.xpt"),
    ("NHANES", "2021-2023", "SMQ_L.XPT", "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/SMQ_L.xpt"),
    ("NHANES", "2021-2023", "BPQ_L.XPT", "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BPQ_L.xpt"),
    ("NHANES", "2021-2023", "DIQ_L.XPT", "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DIQ_L.xpt"),
    ("NHANES", "2021-2023", "KIQ_U_L.XPT", "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/KIQ_U_L.xpt"),
    ("NHANES", "2021-2023", "DPQ_L.XPT", "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DPQ_L.xpt"),
    ("NHANES", "2021-2023", "BMX_L.XPT", "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BMX_L.xpt"),
    ("NHANES", "2021-2023", "GHB_L.XPT", "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/GHB_L.xpt"),
    ("NHANES", "2021-2023", "BIOPRO_L.XPT", "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BIOPRO_L.xpt"),
    ("NHANES", "2017-2018", "DEMO_J.XPT", "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.xpt"),
    ("NHANES", "2017-2018", "MCQ_J.XPT", "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/MCQ_J.xpt"),
    ("NHANES", "2017-2018", "SMQ_J.XPT", "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/SMQ_J.xpt"),
    ("NHANES", "2017-2018", "BPQ_J.XPT", "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BPQ_J.xpt"),
    ("NHANES", "2017-2018", "DIQ_J.XPT", "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DIQ_J.xpt"),
    ("NHANES", "2017-2018", "KIQ_U_J.XPT", "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/KIQ_U_J.xpt"),
    ("NHANES", "2017-2018", "DPQ_J.XPT", "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DPQ_J.xpt"),
    ("NHANES", "2017-2018", "BMX_J.XPT", "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BMX_J.xpt"),
    ("NHANES", "2017-2018", "GHB_J.XPT", "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/GHB_J.xpt"),
    ("NHANES", "2017-2018", "BIOPRO_J.XPT", "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BIOPRO_J.xpt"),
]


DOWNLOAD_FILES = [
    ("NHIS", "2024", "adult24csv.zip", "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2024/adult24csv.zip"),
    ("NHIS", "2023", "adult23csv.zip", "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2023/adult23csv.zip"),
    ("MEPS", "2023", "h251ssp.zip", "https://meps.ahrq.gov/mepsweb/data_files/pufs/h251/h251ssp.zip"),
    ("MEPS", "2023", "h249ssp.zip", "https://meps.ahrq.gov/mepsweb/data_files/pufs/h249/h249ssp.zip"),
    ("MEPS", "2023", "h251dta.zip", "https://meps.ahrq.gov/mepsweb/data_files/pufs/h251/h251dta.zip"),
    ("MEPS", "2023", "h249dta.zip", "https://meps.ahrq.gov/mepsweb/data_files/pufs/h249/h249dta.zip"),
    *NHANES_FILES,
]


SIGNAL_MAPPING = [
    {
        "mcod_signal": "COPD",
        "mcod_nodes": "copd",
        "external_construct": "Clinically reported COPD/emphysema/chronic bronchitis and smoking burden",
        "nhis": "COPD/emphysema/chronic bronchitis + smoking + cancer history",
        "meps": "Priority condition and medical-condition records for emphysema/chronic bronchitis/COPD-like groups + cancer",
        "nhanes": "Medical conditions + spirometry unavailable for recent cycles; smoking, BMI, inflammation/metabolic context",
        "expected_triangulation": "High MCOD + high external chronic lung/smoking burden = concordant chronic respiratory comorbidity signal",
    },
    {
        "mcod_signal": "Diabetes",
        "mcod_nodes": "diabetes",
        "external_construct": "Diagnosed diabetes and glycemic biomarkers",
        "nhis": "Self-reported diabetes + cancer history",
        "meps": "Diabetes priority condition and condition file",
        "nhanes": "Self-reported diabetes + HbA1c/glucose",
        "expected_triangulation": "High external prevalence with MCOD co-mention supports cardiometabolic comorbidity relevance at death",
    },
    {
        "mcod_signal": "CKD",
        "mcod_nodes": "ckd",
        "external_construct": "Kidney disease and renal biomarkers",
        "nhis": "Self-reported kidney disease where available",
        "meps": "Kidney disease condition codes if available in medical conditions file",
        "nhanes": "Kidney questionnaire + creatinine/eGFR and albuminuria where available",
        "expected_triangulation": "External underdiagnosis vs MCOD co-mention can identify late-stage renal burden at death",
    },
    {
        "mcod_signal": "Cardiovascular cluster",
        "mcod_nodes": "ischemic_heart_disease; heart_failure; atrial_fibrillation; cerebrovascular",
        "external_construct": "Heart disease, stroke, hypertension and cardiovascular multimorbidity",
        "nhis": "Heart disease/stroke/hypertension + cancer history",
        "meps": "Heart disease/stroke/hypertension priority conditions + utilization/cost",
        "nhanes": "BPQ, cholesterol/BMI, diabetes, smoking context",
        "expected_triangulation": "Concordance supports chronic cardiometabolic architecture; discordance may reflect treatment-stage or death-certification effects",
    },
    {
        "mcod_signal": "Respiratory failure and pneumonia/influenza",
        "mcod_nodes": "respiratory_failure; pneumonia_influenza",
        "external_construct": "Acute/terminal respiratory pathway rather than premortem chronic comorbidity",
        "nhis": "Limited direct acute terminal event capture",
        "meps": "Acute respiratory condition records and utilization if present",
        "nhanes": "Not suitable for terminal event prevalence",
        "expected_triangulation": "High MCOD + low external chronic prevalence = terminal/death-process signal; interpret separately from chronic core",
    },
    {
        "mcod_signal": "Mental health/substance nodes",
        "mcod_nodes": "depression_anxiety; serious_mental_illness; non_tobacco_substance_opioid",
        "external_construct": "Depression, serious mental illness proxies, substance use and access burden",
        "nhis": "Depression/anxiety and substance-related survey items where available",
        "meps": "Mental health condition records and expenditures",
        "nhanes": "Depression screener, alcohol/tobacco modules",
        "expected_triangulation": "Useful as hypothesis-generating social/behavioral vulnerability layer; not a primary lung-cancer comorbidity endpoint",
    },
]


MATRIX_ROWS = [
    {
        "mcod_signal": "High",
        "external_evidence": "High",
        "interpretation": "High-confidence mortality-context comorbidity signal",
        "action": "Promote to primary chronic-core interpretation; discuss as clinically concordant disease burden at death.",
    },
    {
        "mcod_signal": "High",
        "external_evidence": "Low/uncertain",
        "interpretation": "Death-process, terminal pathway, or certification-amplified signal",
        "action": "Keep as mortality-context signal; separate from chronic-core comorbidity and test sensitivity.",
    },
    {
        "mcod_signal": "Low/uncertain",
        "external_evidence": "High",
        "interpretation": "Premortem comorbidity common but not strongly represented on death certificates",
        "action": "Interpret as clinical burden with lower direct contribution to death certification.",
    },
    {
        "mcod_signal": "Low/uncertain",
        "external_evidence": "Low/uncertain",
        "interpretation": "Low-priority or poorly measured signal",
        "action": "Keep exploratory or exclude from main manuscript unless clinically justified.",
    },
]


def save_tables() -> dict[str, pd.DataFrame]:
    OUT.mkdir(parents=True, exist_ok=True)
    tables = {
        "database_manifest": pd.DataFrame(DATABASE_MANIFEST),
        "download_manifest": pd.DataFrame(
            [
                {"dataset": ds, "cycle_or_year": year, "file_name": name, "url": url}
                for ds, year, name, url in DOWNLOAD_FILES
            ]
        ),
        "signal_mapping": pd.DataFrame(SIGNAL_MAPPING),
        "triangulation_matrix": pd.DataFrame(MATRIX_ROWS),
    }
    tables["database_manifest"].to_csv(OUT / "P1_triangulation_external_public_database_manifest.csv", index=False, encoding="utf-8-sig")
    tables["download_manifest"].to_csv(OUT / "P1_triangulation_external_download_manifest.csv", index=False, encoding="utf-8-sig")
    tables["signal_mapping"].to_csv(OUT / "P1_triangulation_signal_mapping.csv", index=False, encoding="utf-8-sig")
    tables["triangulation_matrix"].to_csv(OUT / "P1_triangulation_validation_matrix.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUT / "P1_triangulation_prep_tables_v1.xlsx", engine="openpyxl") as writer:
        for name, df in tables.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    return tables


def draw_matrix() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14.5, 8.8))
    ax.set_xlim(-0.42, 2.02)
    ax.set_ylim(-0.18, 2.20)
    ax.axis("off")
    colors = {
        (0, 1): "#78c6a3",
        (1, 1): "#f4c78f",
        (0, 0): "#a9c4e8",
        (1, 0): "#d8d8d8",
    }
    cells = {
        (0, 1): ("High-confidence signal", "MCOD high + external high\nPrimary chronic-core interpretation"),
        (1, 1): ("Terminal/certification-amplified", "MCOD high + external low/uncertain\nMortality-context signal, sensitivity required"),
        (0, 0): ("Premortem burden not\nprominent at death", "MCOD low/uncertain + external high\nClinical burden, weaker death-context signal"),
        (1, 0): ("Low-priority/exploratory", "MCOD low/uncertain + external low/uncertain\nKeep exploratory or exclude"),
    }
    for (x, y), (title, body) in cells.items():
        rect = plt.Rectangle((x, y), 1, 1, facecolor=colors[(x, y)], edgecolor="#333333", linewidth=1.3)
        ax.add_patch(rect)
        ax.text(x + 0.5, y + 0.64, title, ha="center", va="center", fontsize=13.5, fontweight="bold")
        ax.text(x + 0.5, y + 0.36, body, ha="center", va="center", fontsize=10.5)
    ax.text(0.5, 2.07, "External clinical / premortem evidence\nHigh", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.text(1.5, 2.07, "External clinical / premortem evidence\nLow or uncertain", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.text(-0.03, 1.5, "MCOD mortality-context\nsignal: High", ha="right", va="center", fontsize=11.5, fontweight="bold")
    ax.text(-0.03, 0.5, "MCOD mortality-context\nsignal: Low or uncertain", ha="right", va="center", fontsize=11.5, fontweight="bold")
    ax.text(
        1.0,
        -0.10,
        "Interpretation rule: external databases contextualize MCOD signals; they do not convert death-certificate co-mentions into clinical prevalence.",
        ha="center",
        va="top",
        fontsize=10,
        color="#333333",
    )
    fig.suptitle("Triangulation Matrix for Lung Cancer Mortality-Context Multimorbidity", y=0.99, fontsize=15)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.90, bottom=0.08)
    for ext in ["png", "svg"]:
        plt.savefig(FIG / f"Figure27_P1_triangulation_validation_matrix.{ext}", dpi=400, bbox_inches="tight")
    plt.close()


def draw_evidence_flow() -> None:
    fig, ax = plt.subplots(figsize=(14.5, 6.1))
    ax.axis("off")
    boxes = [
        (0.035, 0.54, 0.27, 0.31, "MCOD", "Mortality-context co-mention\n1999-2024 trend and network\nrace/sex/geography strata\nCOVID-period sensitivity"),
        (0.395, 0.68, 0.25, 0.22, "MEPS / NHIS", "Survey/condition-file validation\ncancer history + chronic conditions\nhealth-care burden"),
        (0.395, 0.28, 0.25, 0.22, "NHANES", "Biomarker/exposure context\nsmoking, HbA1c, kidney markers\nBMI and depression screener"),
        (0.735, 0.54, 0.25, 0.31, "Triangulated\ninterpretation", "Clinically concordant chronic core\nvs terminal/death-process signals\nvs externally common but death-silent burden"),
    ]
    for x, y, w, h, title, body in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor="#f6f8fb", edgecolor="#333333", linewidth=1.1))
        ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontsize=13, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.35, body, ha="center", va="center", fontsize=9.5)
    arrowprops = dict(arrowstyle="->", linewidth=1.6, color="#333333")
    ax.annotate("", xy=(0.395, 0.76), xytext=(0.305, 0.70), arrowprops=arrowprops)
    ax.annotate("", xy=(0.395, 0.40), xytext=(0.305, 0.62), arrowprops=arrowprops)
    ax.annotate("", xy=(0.735, 0.72), xytext=(0.645, 0.78), arrowprops=arrowprops)
    ax.annotate("", xy=(0.735, 0.62), xytext=(0.645, 0.38), arrowprops=arrowprops)
    ax.text(0.5, 0.08, "No-application external datasets are used for contextual validation, not causal inference.", ha="center", fontsize=10)
    fig.suptitle("No-Application Triangulation Workflow", y=0.96, fontsize=15)
    fig.subplots_adjust(left=0.02, right=0.99, top=0.90, bottom=0.08)
    for ext in ["png", "svg"]:
        plt.savefig(FIG / f"Figure28_P1_triangulation_evidence_flow.{ext}", dpi=400, bbox_inches="tight")
    plt.close()


def write_protocol() -> None:
    body = f"""# P1 no-application external database triangulation protocol

## Purpose

The MCOD analysis identifies mortality-context multimorbidity signals among U.S. deaths with lung cancer as the underlying cause. The external databases are used to contextualize whether those mortality-context signals are concordant with premortem or survey-based chronic disease burden. This is a triangulation design, not a conversion of death-certificate co-mentions into clinical prevalence.

## Interpretation rule

Use the following language in the manuscript:

> MCOD data identify conditions co-mentioned in the disease context of lung cancer death. External survey and condition-file data were used to contextualize whether high MCOD signals were concordant with premortem chronic disease burden. These external data do not make death-certificate co-mentions equivalent to clinical comorbidity prevalence.

## Primary no-application datasets

1. NHIS public-use Sample Adult files, 2023-2024.
   - Role: self-reported chronic conditions and cancer-history context.
   - Main outputs: chronic condition prevalence among adults with cancer history, by race/ethnicity and sex where sample size permits.
   - Key limitation: lung-cancer-specific public-use sample may be sparse.

2. MEPS public-use HC files, 2023.
   - Role: condition-file and health-care-use triangulation among respondents with cancer-related priority conditions.
   - Main outputs: condition clusters, utilization/cost, and chronic disease concordance.
   - Key limitation: public-use cancer site specificity may be limited.

3. NHANES public-use files, 2017-2018 and 2021-2023.
   - Role: exposure/biomarker context for smoking, cardiometabolic disease, kidney disease, BMI, depression.
   - Main outputs: weighted background prevalence and biomarker profiles in adults with cancer history and in high-risk demographic strata.
   - Key limitation: not powered for lung-cancer-specific comorbidity validation.

## Planned outputs

- `P1_triangulation_external_public_database_manifest.csv`
- `P1_triangulation_external_download_manifest.csv`
- `P1_triangulation_signal_mapping.csv`
- `P1_triangulation_validation_matrix.csv`
- `P1_triangulation_prep_tables_v1.xlsx`
- `Figure27_P1_triangulation_validation_matrix.png/svg`
- `Figure28_P1_triangulation_evidence_flow.png/svg`

## Analysis sequence

1. Keep MCOD as the primary mortality-context analysis.
2. Classify MCOD signals into chronic-core versus terminal/acute pathway.
3. Estimate external chronic burden using NHIS and MEPS public-use files.
4. Use NHANES for exposure/biomarker context, especially smoking, diabetes, CKD and BMI.
5. Assign each MCOD node to one triangulation cell:
   - MCOD high / external high: clinically concordant mortality-context chronic signal.
   - MCOD high / external low: terminal/death-process or death-certification-amplified signal.
   - MCOD low / external high: clinically common but less prominent in death certification.
   - MCOD low / external low: exploratory or low-priority signal.

## Statistical plan

- Use complex survey weights for NHIS, MEPS and NHANES.
- Estimate weighted proportions and 95% confidence intervals.
- Use broad chronic-condition groups rather than over-specific rare disease categories.
- Use minimum unweighted cell thresholds before reporting race/ethnicity or sex subgroup estimates.
- Present external analyses as concordance/context tables, not as direct validation of individual MCOD death records.

## Manuscript wording

Recommended phrase:

`mortality-context multimorbidity signals triangulated against public survey and condition-file evidence`

Avoid:

`death-certificate-derived clinical comorbidity prevalence`
"""
    (OUT / "P1_triangulation_no_application_protocol.md").write_text(body, encoding="utf-8")


def is_html_error_file(path: Path) -> bool:
    if path.suffix.lower() != ".xpt" or not path.exists():
        return False
    try:
        head = path.read_bytes()[:512].lower()
    except OSError:
        return False
    return b"<html" in head or b"<!doctype html" in head or b"page not found" in head


def download_file(url: str, dest: Path, timeout: int = 120) -> tuple[str, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        if is_html_error_file(dest):
            dest.unlink()
        else:
            return "cached", ""
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response, tmp.open("wb") as handle:
            handle.write(response.read())
        tmp.replace(dest)
        if is_html_error_file(dest):
            dest.unlink()
            return "failed", "download returned HTML/error page instead of XPT"
        return "downloaded", ""
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        return "failed", str(exc)


def download_public_files() -> None:
    status_rows = []
    for dataset, year, file_name, url in DOWNLOAD_FILES:
        dest = EXT / "raw" / dataset.lower() / str(year) / file_name
        status, error = download_file(url, dest)
        extracted_files = ""
        if status in {"downloaded", "cached"} and file_name.lower().endswith(".zip"):
            try:
                extract_dir = dest.parent / dest.stem
                extract_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(dest) as zf:
                    zip_members = zf.namelist()
                    zf.extractall(extract_dir)
                extracted_files = ";".join(zip_members)
            except Exception as exc:
                status = "downloaded_extract_failed" if status == "downloaded" else "cached_extract_failed"
                error = str(exc)
        status_rows.append(
            {
                "dataset": dataset,
                "cycle_or_year": year,
                "file_name": file_name,
                "url": url,
                "local_path": str(dest),
                "status": status,
                "bytes": dest.stat().st_size if dest.exists() else 0,
                "extracted_files": extracted_files,
                "error": error,
            }
        )
        print(f"{dataset} {year} {file_name}: {status}", flush=True)
    pd.DataFrame(status_rows).to_csv(OUT / "P1_triangulation_external_download_status.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download no-application public-use files into external_public_data/raw.")
    args = parser.parse_args()

    save_tables()
    draw_matrix()
    draw_evidence_flow()
    write_protocol()
    if args.download:
        download_public_files()
    print("Triangulation preparation complete.")


if __name__ == "__main__":
    main()
