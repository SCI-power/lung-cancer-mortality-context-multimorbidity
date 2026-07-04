from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from parse_nchs_mcod_year import iter_records, parse_record


PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "raw"
OUT = PROJECT / "outputs"

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

SEX_LABELS = {"1": "Male", "2": "Female"}


def write_dicts(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    age_rows: list[dict] = []
    sex_rows: list[dict] = []
    audit_rows: list[dict] = []
    for year in range(1999, 2025):
        zip_path = RAW / f"Mort{year}us.zip"
        if not zip_path.exists():
            raise FileNotFoundError(zip_path)

        all_deaths = 0
        us_resident_deaths = 0
        lung_deaths = 0
        by_age12: Counter[str] = Counter()
        by_sex: Counter[str] = Counter()

        for idx, line in enumerate(iter_records(zip_path), start=1):
            all_deaths += 1
            if idx % 1_000_000 == 0:
                print(f"{year}: scanned {idx:,} records; lung resident deaths {lung_deaths:,}")
            rec = parse_record(line)
            if rec["resident_status"] == "4":
                continue
            us_resident_deaths += 1
            if not rec["ucd"].startswith("C34"):
                continue
            lung_deaths += 1
            age12 = rec["age12"] or "Unknown"
            sex = rec["sex"] or "Unknown"
            by_age12[age12] += 1
            by_sex[sex] += 1

        for age12, deaths in sorted(by_age12.items()):
            age_rows.append(
                {
                    "year": year,
                    "age12": age12,
                    "age12_label": AGE12_LABELS.get(age12, "Unknown"),
                    "lung_cancer_ucd_deaths": deaths,
                    "proportion_of_lung_cancer_deaths": deaths / lung_deaths if lung_deaths else "",
                }
            )
        for sex, deaths in sorted(by_sex.items()):
            sex_rows.append(
                {
                    "year": year,
                    "sex": sex,
                    "sex_label": SEX_LABELS.get(sex, "Unknown"),
                    "lung_cancer_ucd_deaths": deaths,
                    "proportion_of_lung_cancer_deaths": deaths / lung_deaths if lung_deaths else "",
                }
            )
        audit_rows.append(
            {
                "year": year,
                "all_deaths_in_file": all_deaths,
                "all_deaths_us_residents": us_resident_deaths,
                "lung_cancer_ucd_deaths_us_residents": lung_deaths,
                "age12_sum": sum(by_age12.values()),
                "sex_sum": sum(by_sex.values()),
            }
        )
        print(f"{year}: lung resident deaths {lung_deaths:,}")

    write_dicts(OUT / "P1_mcod_1999_2024_age12_denominators.csv", age_rows)
    write_dicts(OUT / "P1_mcod_1999_2024_sex_denominators.csv", sex_rows)
    write_dicts(OUT / "P1_mcod_1999_2024_stratified_denominator_audit.csv", audit_rows)
    print("Wrote age12 and sex denominators for 1999-2024.")


if __name__ == "__main__":
    main()
