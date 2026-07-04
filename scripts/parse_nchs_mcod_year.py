from __future__ import annotations

import argparse
import csv
import itertools
import os
import re
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[0]
REQUEST_PLAN = PROJECT / "data_request_plan.csv"


def normalize_icd(code: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", code.upper().strip())


def icd_category(code: str) -> str:
    code = normalize_icd(code)
    return code[:3]


def icd_rank(code: str, width: int) -> tuple[str, int] | None:
    code = normalize_icd(code)
    if not code:
        return None
    prefix = code[0]
    digits = re.sub(r"\D", "", code[1:width])
    if not digits:
        return None
    return prefix, int(digits)


def parse_code_terms(raw: str) -> list[str]:
    if not raw:
        return []
    return [x.strip().upper() for x in raw.replace(",", ";").split(";") if x.strip()]


def match_term(code: str, term: str) -> bool:
    code = normalize_icd(code)
    if not code:
        return False

    term = term.strip().upper()
    if "-" in term:
        start, end = [x.strip() for x in term.split("-", 1)]
        start_norm = normalize_icd(start)
        end_norm = normalize_icd(end)
        compare_width = max(len(start_norm), len(end_norm))
        if compare_width <= 3:
            code_norm = icd_category(code)
            compare_width = 3
        else:
            code_norm = normalize_icd(code)[:compare_width]
        start_rank = icd_rank(start_norm, compare_width)
        end_rank = icd_rank(end_norm, compare_width)
        code_rank = icd_rank(code_norm, compare_width)
        if start_rank is None or end_rank is None or code_rank is None:
            return False
        if start_rank[0] != end_rank[0] or start_rank[0] != code_rank[0]:
            return False
        return start_rank[1] <= code_rank[1] <= end_rank[1]

    exact = normalize_icd(term)
    if len(exact) == 3:
        return icd_category(code) == exact
    return code.startswith(exact)


def any_match(code: str, terms: list[str]) -> bool:
    return any(match_term(code, term) for term in terms)


def load_groups(priority: str | None = None) -> list[dict]:
    with REQUEST_PLAN.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        if priority and row["priority"] != priority:
            continue
        if row["multiple_cause_group"] == "lung_cancer_total":
            continue
        out.append(
            {
                "query_id": row["query_id"],
                "group": row["multiple_cause_group"],
                "terms": parse_code_terms(row["multiple_cause_codes"]),
                "priority": row["priority"],
            }
        )
    return out


def iter_records(zip_path: Path):
    try:
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
            if not names:
                raise ValueError(f"No files in {zip_path}")
            with z.open(names[0]) as f:
                for raw in f:
                    yield raw.decode("latin1", errors="ignore").rstrip("\r\n")
        return
    except NotImplementedError:
        # Some NCHS ZIP files use a compression method unsupported by Python's
        # standard zipfile module. Windows bsdtar can stream them without a
        # full extraction step.
        pass

    seven_zip_env = os.environ.get("SEVEN_ZIP")
    seven_zip = Path(seven_zip_env) if seven_zip_env else None
    if seven_zip and seven_zip.exists():
        cmd = [str(seven_zip), "x", "-so", str(zip_path)]
    else:
        tar = shutil.which("tar")
        if tar is None:
            raise RuntimeError("archive streaming requires either Python zipfile support, SEVEN_ZIP, or tar on PATH")
        cmd = [tar, "-xOf", str(zip_path)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    try:
        for raw in proc.stdout:
            yield raw.decode("latin1", errors="ignore").rstrip("\r\n")
    finally:
        proc.stdout.close()
        return_code = proc.wait()
        if return_code != 0:
            stderr = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
            raise RuntimeError(f"archive streaming failed for {zip_path}: {stderr}") from None


def parse_record(line: str) -> dict:
    # Positions are 1-based in NCHS documentation. Python uses 0-based slices.
    resident_status = line[19].strip()
    if len(line) <= 440:
        layout = "legacy_1999_2002_width440"
        year = line[114:118]
        sex = line[58].strip()
        age52 = line[66:68].strip()
        age27 = line[68:70].strip()
        age12 = line[70:72].strip()
        ucd = normalize_icd(line[141:145])
        record_axis_blob = line[340:440]
    else:
        layout = "modern_width490plus"
        year = line[101:105]
        sex = line[68].strip()
        age52 = line[74:76].strip()
        age27 = line[76:78].strip()
        age12 = line[78:80].strip()
        ucd = normalize_icd(line[145:149])
        record_axis_blob = line[343:443]
    record_axis = [normalize_icd(record_axis_blob[i : i + 5]) for i in range(0, 100, 5)]
    record_axis = [c for c in record_axis if c]
    return {
        "year": year,
        "record_layout": layout,
        "resident_status": resident_status,
        "sex": sex,
        "age52": age52,
        "age27": age27,
        "age12": age12,
        "ucd": ucd,
        "record_axis": record_axis,
    }


def process_year(zip_path: Path, year: int, groups: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    total_deaths = 0
    us_resident_deaths = 0
    foreign_resident_deaths = 0
    lung_deaths = 0
    group_counts = Counter()
    group_by_sex = Counter()
    group_by_age12 = Counter()
    pair_counts = Counter()
    lung_ucd_examples = Counter()

    for idx, line in enumerate(iter_records(zip_path), start=1):
        total_deaths += 1
        if idx % 500000 == 0:
            print(f"Processed {idx:,} records; lung deaths {lung_deaths:,}")
        rec = parse_record(line)
        if rec["resident_status"] == "4":
            foreign_resident_deaths += 1
            continue
        us_resident_deaths += 1
        if not rec["ucd"].startswith("C34"):
            continue
        lung_deaths += 1
        lung_ucd_examples[rec["ucd"]] += 1

        present = []
        codes = rec["record_axis"]
        for g in groups:
            if any(any_match(code, g["terms"]) for code in codes):
                group_counts[g["group"]] += 1
                group_by_sex[(g["group"], rec["sex"] or "Unknown")] += 1
                group_by_age12[(g["group"], rec["age12"] or "Unknown")] += 1
                present.append(g["group"])

        for a, b in itertools.combinations(sorted(set(present)), 2):
            pair_counts[(a, b)] += 1
    summary_rows = [
        {
            "year": year,
            "scope": "all_deaths_in_file",
            "deaths": total_deaths,
            "denominator_lung_cancer_ucd_deaths": "",
            "proportion_among_lung_cancer_deaths": "",
        },
        {
            "year": year,
            "scope": "all_deaths_us_residents",
            "deaths": us_resident_deaths,
            "denominator_lung_cancer_ucd_deaths": "",
            "proportion_among_lung_cancer_deaths": "",
        },
        {
            "year": year,
            "scope": "foreign_resident_deaths_excluded",
            "deaths": foreign_resident_deaths,
            "denominator_lung_cancer_ucd_deaths": "",
            "proportion_among_lung_cancer_deaths": "",
        },
        {
            "year": year,
            "scope": "underlying_cause_C34",
            "deaths": lung_deaths,
            "denominator_lung_cancer_ucd_deaths": lung_deaths,
            "proportion_among_lung_cancer_deaths": 1.0,
        },
    ]
    for g in groups:
        count = group_counts[g["group"]]
        summary_rows.append(
            {
                "year": year,
                "scope": g["group"],
                "deaths": count,
                "denominator_lung_cancer_ucd_deaths": lung_deaths,
                "proportion_among_lung_cancer_deaths": count / lung_deaths if lung_deaths else "",
            }
        )

    write_dicts(out_dir / f"P1_mcod_{year}_lung_cancer_comorbidity_counts.csv", summary_rows)

    pair_rows = [
        {
            "year": year,
            "group_a": a,
            "group_b": b,
            "co_mentioned_deaths": c,
            "denominator_lung_cancer_ucd_deaths": lung_deaths,
            "proportion_among_lung_cancer_deaths": c / lung_deaths if lung_deaths else "",
        }
        for (a, b), c in pair_counts.most_common()
    ]
    write_dicts(out_dir / f"P1_mcod_{year}_lung_cancer_pair_counts.csv", pair_rows)

    sex_rows = [
        {
            "year": year,
            "group": group,
            "sex": sex,
            "deaths": count,
        }
        for (group, sex), count in sorted(group_by_sex.items())
    ]
    write_dicts(out_dir / f"P1_mcod_{year}_lung_cancer_group_by_sex.csv", sex_rows)

    age_rows = [
        {
            "year": year,
            "group": group,
            "age12": age,
            "deaths": count,
        }
        for (group, age), count in sorted(group_by_age12.items())
    ]
    write_dicts(out_dir / f"P1_mcod_{year}_lung_cancer_group_by_age12.csv", age_rows)

    examples_rows = [{"ucd": k, "deaths": v} for k, v in lung_ucd_examples.most_common()]
    write_dicts(out_dir / f"P1_mcod_{year}_lung_cancer_ucd_code_distribution.csv", examples_rows)

    print(
        f"Finished {year}: all deaths={total_deaths:,}, "
        f"US-resident deaths={us_resident_deaths:,}, C34 underlying={lung_deaths:,}"
    )
    print(f"Wrote outputs to {out_dir}")


def write_dicts(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--priority", choices=["high", "medium", "low", "all"], default="high")
    parser.add_argument("--out", type=Path, default=PROJECT / "outputs")
    args = parser.parse_args()
    groups = load_groups(priority=None if args.priority == "all" else args.priority)
    process_year(args.zip, args.year, groups, args.out)


if __name__ == "__main__":
    main()
