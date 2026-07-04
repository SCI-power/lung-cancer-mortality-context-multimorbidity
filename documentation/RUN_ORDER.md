# Recommended Run Order

The original scripts were developed iteratively. For reproducibility, use this logical order.

## 1. MCOD and CDC WONDER data generation

1. `python scripts/build_query_manifest.py`
2. `python scripts/download_parse_mcod_1999_2024.py`
3. `python scripts/build_p1_age_denominators_1999_2024.py`
4. `python scripts/build_p1_wonder_export_package.py`
5. Place/manual-check CDC WONDER exports as instructed by the generated package.
6. `python scripts/build_p1_wonder_verified_outputs.py`

`parse_nchs_mcod_year.py` is a parser utility used by downstream scripts.

## 2. Primary MCOD analyses

1. `python scripts/build_p1_1999_2024_results.py`
2. `python scripts/summarize_p1_mcod_1999_2024.py`
3. `python scripts/build_p1_formal_trend_models.py`
4. `python scripts/build_p1_high_impact_supplement_1999_2024.py`
5. `python scripts/enhance_p1_network_results.py`
6. `python scripts/build_p1_covid_sensitivity.py`
7. `python scripts/build_p1_stratified_tables.py`
8. `python scripts/build_p1_priority7_enhancements.py`
9. `python scripts/build_p1_race_ethnicity_module_2018_2024.py`

## 3. External public-data analyses

1. `python scripts/profile_p1_external_public_data.py`
2. `python scripts/build_p1_external_analysis_ready.py`
3. `python scripts/build_p1_triangulation_prep.py`
4. `Rscript scripts/analyze_p1_external_triangulation_survey.R`
5. `python scripts/summarize_p1_external_triangulation.py`
6. `Rscript scripts/analyze_p1_nhis_lung_cancer_comparator.R`
7. `python scripts/summarize_p1_nhis_comparator_analysis.py`

## 4. Result figures/tables

1. `python scripts/make_p1_publication_result_figures.py`
2. `python scripts/make_p1_stratified_figures.py`

Journal-specific DOCX generation, submission folders, manual montage, and later eClinicalMedicine image editing are deliberately excluded.
