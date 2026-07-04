from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUEST_PLAN = ROOT / "data_request_plan.csv"
OUT = ROOT / "outputs" / "P1_query_manifest_summary.csv"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with REQUEST_PLAN.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    summary = []
    for row in rows:
        codes = [c.strip() for c in row["multiple_cause_codes"].replace(";", ",").split(",") if c.strip()]
        summary.append(
            {
                "query_id": row["query_id"],
                "priority": row["priority"],
                "multiple_cause_group": row["multiple_cause_group"],
                "n_code_terms": len(codes),
                "codes": ";".join(codes),
                "output_file": row["output_file"],
                "status": "not_run",
            }
        )

    with OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print(f"Wrote {OUT}")
    print(f"Queries: {len(summary)}")
    print("Priority counts:")
    counts = {}
    for row in summary:
        counts[row["priority"]] = counts.get(row["priority"], 0) + 1
    for k in sorted(counts):
        print(f"  {k}: {counts[k]}")


if __name__ == "__main__":
    main()
