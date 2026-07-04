from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "outputs"
WONDER_DIR = OUT / "wonder_exports"


KEY_NODES = [
    ("lung_cancer_total", "", "Denominator: underlying cause C34 only"),
    ("copd", "J40-J44;J47", "COPD/chronic lower respiratory disease co-mention"),
    ("respiratory_failure", "J96", "Terminal respiratory failure co-mention"),
    ("pneumonia_influenza", "J09-J18", "Respiratory infection co-mention"),
    ("ischemic_heart_disease", "I20-I25", "Ischemic heart disease co-mention"),
    ("heart_failure", "I50", "Heart failure co-mention"),
    ("atrial_fibrillation", "I48", "Atrial fibrillation co-mention"),
    ("diabetes", "E10-E14", "Diabetes co-mention"),
    ("pulmonary_embolism", "I26", "Pulmonary embolism co-mention"),
    ("ckd", "N18", "Chronic kidney disease co-mention"),
]


def build_manifest() -> pd.DataFrame:
    rows = []
    query_groups = [
        (
            "region_rates",
            "Year;Census Region",
            "Export deaths, population, crude rate, age-adjusted rate, 95% CI when available.",
            "Main geographic rate upgrade; region is preferred for age-adjusted rates.",
        ),
        (
            "urbanization_counts",
            "Year;2013 Urbanization",
            "Export deaths and crude rates if WONDER returns them; do not require age-adjusted rate.",
            "Urban-rural upgrade; official single-race MCD has rate constraints for county/urbanization groupings.",
        ),
        (
            "sex_region_counts",
            "Year;Sex;Census Region",
            "Export deaths and rates when available.",
            "Optional reviewer-response table for sex-by-region pattern.",
        ),
    ]
    qid = 1
    for query_family, group_by, metrics, purpose in query_groups:
        for node, multiple_codes, note in KEY_NODES:
            rows.append(
                {
                    "wonder_query_id": f"P1_W{qid:03d}",
                    "query_family": query_family,
                    "node": node,
                    "database": "Multiple Cause of Death, 2018-2024, Single Race",
                    "underlying_cause_icd10": "C34",
                    "multiple_cause_icd10": multiple_codes,
                    "group_results_by": group_by,
                    "years": "2018-2024",
                    "age_filter": "All ages",
                    "metrics_to_export": metrics,
                    "output_file": f"P1_W{qid:03d}_{query_family}_{node}.csv",
                    "purpose": purpose,
                    "node_note": note,
                }
            )
            qid += 1
    return pd.DataFrame(rows)


def write_template() -> None:
    WONDER_DIR.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    manifest.to_csv(WONDER_DIR / "P1_wonder_export_manifest_v1.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# P1 CDC WONDER Export Package v1",
        "",
        "## Why this package is needed",
        "",
        "The raw NCHS MCOD public-use files used for the main analysis do not provide the full public CDC WONDER geographic/rate interface. CDC WONDER can add region, urbanization, population denominators, crude rates, and age-adjusted rates where available.",
        "",
        "## CDC WONDER database",
        "",
        "- Use: Multiple Cause of Death, 2018-2024, Single Race.",
        "- Underlying Cause of Death: `C34` for all queries.",
        "- Multiple Cause of Death: blank for denominator queries; node-specific ICD-10 codes for co-mention numerators.",
        "- Years: 2018-2024.",
        "",
        "## Export families",
        "",
        "### 1. Region rates",
        "",
        "- Group Results By: `Year` and `Census Region`.",
        "- Export deaths, population, crude rate, age-adjusted rate, and 95% CI when WONDER provides them.",
        "- This is the strongest add-on for high-impact framing because region-level rates are more suitable for age adjustment than county/urbanization groupings.",
        "",
        "### 2. Urbanization counts",
        "",
        "- Group Results By: `Year` and `2013 Urbanization`.",
        "- Export deaths and crude rates when available.",
        "- Do not force age-adjusted rates for urbanization: CDC documentation for the 2018-2024 single-race MCD database notes constraints for county-level and urbanization-related rate calculations.",
        "",
        "### 3. Sex by region",
        "",
        "- Group Results By: `Year`, `Sex`, and `Census Region`.",
        "- This is optional. Run only after family 1 and 2 are complete.",
        "",
        "## File naming",
        "",
        "Save all exports to:",
        "",
        "`P1_cdc_wonder_multimorbidity_network/outputs/wonder_exports/`",
        "",
        "Use filenames exactly as listed in `P1_wonder_export_manifest_v1.csv`.",
        "",
        "## Priority order",
        "",
        "1. `region_rates` for `lung_cancer_total`, `copd`, `respiratory_failure`, `ischemic_heart_disease`, `heart_failure`, `diabetes`, `atrial_fibrillation`.",
        "2. `urbanization_counts` for the same seven nodes.",
        "3. Add `pneumonia_influenza`, `pulmonary_embolism`, and `ckd` if export burden is acceptable.",
        "4. Run `sex_region_counts` only if reviewer or target journal requires subgroup geography.",
        "",
        "## Intended manuscript use",
        "",
        "- Region age-adjusted rates: main or supplementary table/figure.",
        "- Urban-rural counts/proportions: supplementary figure unless differences are large.",
        "- If WONDER suppresses or omits rates for urbanization, report urbanization as co-mention proportions among lung cancer deaths rather than population mortality rates.",
    ]
    (WONDER_DIR / "P1_cdc_wonder_export_instructions_v1.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote CDC WONDER export package to {WONDER_DIR}")


if __name__ == "__main__":
    write_template()
