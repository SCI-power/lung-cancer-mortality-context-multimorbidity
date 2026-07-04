from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "raw"
OUTPUT = PROJECT / "outputs"
YEARLY = OUTPUT / "yearly_resident_1999_2024"
LOG = OUTPUT / "P1_mcod_download_parse_status_1999_2024.csv"
PARSER = PROJECT / "scripts" / "parse_nchs_mcod_year.py"
BASE_DIR = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/DVS/mortality/"


def discover_official_urls() -> dict[int, str]:
    html = ""
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(BASE_DIR, timeout=60) as response:
                html = response.read().decode("utf-8", errors="ignore")
            break
        except Exception as exc:
            last_error = exc
            print(f"URL discovery attempt {attempt} failed: {exc}")
    if not html:
        print(f"URL discovery unavailable; falling back to direct year URLs. Last error: {last_error}")
        return {}
    links = re.findall(r'href=["\']([^"\']*Mort\d{4}us\.zip)["\']', html, flags=re.IGNORECASE)
    urls: dict[int, str] = {}
    for href in links:
        match = re.search(r"Mort(\d{4})us\.zip", href, flags=re.IGNORECASE)
        if not match:
            continue
        year = int(match.group(1))
        if href.startswith("http"):
            urls[year] = href
        elif href.startswith("/"):
            urls[year] = "https://ftp.cdc.gov" + href
        else:
            urls[year] = BASE_DIR + href
    return urls


def official_url_candidates(year: int, urls: dict[int, str]) -> list[str]:
    candidates = []
    if year in urls:
        candidates.append(urls[year])
    candidates.extend([
        BASE_DIR + f"mort{year}us.zip",
        BASE_DIR + f"Mort{year}us.zip",
    ])
    return list(dict.fromkeys(candidates))


def is_valid_zip(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 10_000_000 and zipfile.is_zipfile(path)


def download_year(year: int, urls: dict[int, str]) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    dest = RAW / f"Mort{year}us.zip"
    if is_valid_zip(dest):
        print(f"{year}: using existing {dest.name} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
        return dest
    if dest.exists():
        print(f"{year}: removing incomplete/invalid {dest.name} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
        dest.unlink()
    tmp = dest.with_suffix(dest.suffix + ".part")
    curl = shutil.which("curl.exe") or shutil.which("curl")
    errors = []
    for url in official_url_candidates(year, urls):
        if tmp.exists():
            tmp.unlink()
        print(f"{year}: downloading {url}")
        try:
            if curl:
                cmd = [
                    curl,
                    "-L",
                    "--fail",
                    "--retry",
                    "8",
                    "--retry-all-errors",
                    "--retry-delay",
                    "5",
                    "--connect-timeout",
                    "30",
                    "--output",
                    str(tmp),
                    url,
                ]
                subprocess.run(cmd, check=True)
            else:
                urllib.request.urlretrieve(url, tmp)
            if not is_valid_zip(tmp):
                size = tmp.stat().st_size if tmp.exists() else 0
                raise RuntimeError(f"downloaded file is incomplete or invalid: {size} bytes")
            tmp.replace(dest)
            print(f"{year}: downloaded {dest.stat().st_size / 1024 / 1024:.1f} MB")
            return dest
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            print(f"{year}: download candidate failed: {exc}")
            if tmp.exists():
                tmp.unlink()
    raise RuntimeError(f"download failed for {year}; " + " | ".join(errors))


def parse_year(year: int, zip_path: Path, force: bool = False) -> None:
    YEARLY.mkdir(parents=True, exist_ok=True)
    out_file = YEARLY / f"P1_mcod_{year}_lung_cancer_comorbidity_counts.csv"
    if out_file.exists() and out_file.stat().st_size > 0 and not force:
        print(f"{year}: parsed resident-scope output exists, skipping")
        return
    cmd = [
        sys.executable,
        str(PARSER),
        "--year",
        str(year),
        "--zip",
        str(zip_path),
        "--priority",
        "all",
        "--out",
        str(YEARLY),
    ]
    print(f"{year}: parsing all nodes to {YEARLY.name}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=1999)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--parse-only", action="store_true")
    parser.add_argument("--force-parse", action="store_true")
    args = parser.parse_args()

    if args.download_only and args.parse_only:
        raise SystemExit("--download-only and --parse-only cannot both be set")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    urls = discover_official_urls()
    status_rows = []
    for year in range(args.start_year, args.end_year + 1):
        try:
            zip_path = RAW / f"Mort{year}us.zip"
            if not args.parse_only:
                zip_path = download_year(year, urls)
            elif not zip_path.exists():
                raise FileNotFoundError(zip_path)
            if not args.download_only:
                parse_year(year, zip_path, force=args.force_parse)
            status_rows.append({"year": year, "status": "ok", "message": ""})
        except Exception as exc:
            message = str(exc).replace("\n", " ")[:500]
            print(f"{year}: FAILED: {message}")
            status_rows.append({"year": year, "status": "failed", "message": message})
            continue

    with LOG.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["year", "status", "message"])
        writer.writeheader()
        writer.writerows(status_rows)
    print(f"Done. Wrote status log: {LOG}")


if __name__ == "__main__":
    main()
