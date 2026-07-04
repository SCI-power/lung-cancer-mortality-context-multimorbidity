# Lung-Cancer MCOD Multimorbidity Study: Reproducibility Code

This repository contains the final code needed to generate analysis-ready data and reproduce the statistical analyses for the lung-cancer mortality-context multimorbidity study.

## What is included

- `scripts/`: final data-generation, MCOD analysis, external public-data analysis, sensitivity/stratified analysis, and result figure/table code.
- `documentation/included_final_code_manifest.csv`: included scripts and their purpose.
- `documentation/excluded_nonfinal_scripts_manifest.csv`: reviewed scripts deliberately excluded from this GitHub package.
- `documentation/RUN_ORDER.md`: recommended logical run order.
- `documentation/requirements_python.txt` and `documentation/requirements_R.txt`: package dependencies inferred from the original analysis package.


## Quick start

1. Install dependencies listed in `documentation/requirements_python.txt` and `documentation/requirements_R.txt`.
2. Place required public-use raw files in the expected `raw/` and `external_public_data/raw/` locations.
3. Follow `documentation/RUN_ORDER.md`.


