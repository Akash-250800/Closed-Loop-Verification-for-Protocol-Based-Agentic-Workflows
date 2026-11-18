"""Prepare and validate the fixed 30-country redo dataset.

Run from E:\\MCP-Thesis with:
    python 01_prepare_redo_data.py

Inputs (must already exist beside this script):
    wikidata_presidential_record.csv
    hdata_plt_cleaned.csv

Outputs (created under redo_data/):
    country_reference.csv
    wikidata_terms.csv
    hdata_terms.csv
    benchmark_cases.csv
    manifest.json
    PREPARE_REDO_DATA_PASS.txt

This script does not download, invent, or silently repair historical facts. It
checks the fixed Wikidata-derived benchmark against the independent H-DATA file
and derives the three reference tables used by the redo MCP experiments.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import unicodedata
from datetime import date, datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WIKIDATA_FILE = ROOT / "wikidata_presidential_record.csv"
HDATA_FILE = ROOT / "hdata_plt_cleaned.csv"
OUT_DIR = ROOT / "redo_data"

EXPECTED_ISO3 = [
    "FRA", "DEU", "ITA", "PRT", "IRL",
    "AUT", "FIN", "POL", "CZE", "SVK",
    "HUN", "HRV", "SVN", "EST", "LVA",
    "LTU", "GRC", "CYP", "ROU", "USA",
    "MEX", "BRA", "ARG", "CHL", "COL",
    "PER", "URY", "PRY", "ECU", "PAN",
]

HDATA_COUNTRY = {
    "ARG": "argentina", "AUT": "austria", "BRA": "brazil",
    "CHL": "chile", "COL": "colombia", "CYP": "cyprus",
    "CZE": "czech republic", "DEU": "germany", "ECU": "ecuador",
    "EST": "estonia", "FIN": "finland", "FRA": "france",
    "GRC": "greece", "HRV": "croatia", "HUN": "hungary",
    "IRL": "ireland", "ITA": "italy", "LTU": "lithuania",
    "LVA": "latvia", "MEX": "mexico", "PAN": "panama",
    "PER": "peru", "POL": "poland", "PRT": "portugal",
    "PRY": "paraguay", "ROU": "romania", "SVK": "slovakia",
    "SVN": "slovenia", "URY": "uruguay",
    "USA": "united states of america",
}

BENCHMARK_REQUIRED = {
    "iso3", "country", "capital", "countryQID", "capitalQID", "officeQID",
    "wikidataPresidentQID", "wikidataPresident", "hdataPresident",
    "wikidataStart", "wikidataEnd", "hdataStart", "hdataEnd",
    "overlapStart", "overlapEnd", "testDate", "nameSimilarity",
    "wikidataSourcePath", "validationStatus", "agreementType",
}

HDATA_REQUIRED = {
    "territorynamepernament", "name1", "interregnum", "officetitle",
    "officetypology", "u_hdata_leader_date_date_in",
    "u_hdata_leader_date_date_out",
}

ALLOWED_AGREEMENT_TYPES = {
    "EXACT_TWO_SOURCE",
    "SOURCE2_RIGHT_CENSORED_AT_2022",
    "ONE_DAY_END_BOUNDARY_RESOLVED",
}


def die(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig", errors="strict") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def write_csv_atomic(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def parse_date(value: str, field: str, iso3: str) -> date:
    text = (value or "").strip()
    try:
        return date.fromisoformat(text[:10])
    except Exception as exc:
        die(f"{iso3}: invalid {field} date {value!r}: {exc}")


def normalize(value: str) -> str:
    text = (value or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    ignored = {"jr", "sr", "ii", "iii", "iv"}
    return " ".join(token for token in text.split() if token not in ignored)


def name_similarity(first: str, second: str) -> float:
    a, b = normalize(first), normalize(second)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    a_tokens, b_tokens = a.split(), b.split()
    set_a, set_b = set(a_tokens), set(b_tokens)
    seq = SequenceMatcher(None, a, b).ratio()
    union = set_a | set_b
    jaccard = len(set_a & set_b) / len(union) if union else 0.0
    first_match = float(a_tokens[0] == b_tokens[0])
    last_match = float(a_tokens[-1] == b_tokens[-1])
    subset = float(set_a.issubset(set_b) or set_b.issubset(set_a))
    token_score = (
        0.55 * jaccard + 0.15 * first_match + 0.20 * last_match + 0.10 * subset
    )
    return max(seq, token_score)


def validate_qid(value: str, field: str, iso3: str) -> None:
    if not re.fullmatch(r"Q\d+", value or ""):
        die(f"{iso3}: invalid Wikidata QID in {field}: {value!r}")


def select_hdata_match(case: dict[str, str], rows: list[dict[str, str]]) -> dict[str, str]:
    iso3 = case["iso3"]
    target_country = HDATA_COUNTRY[iso3]
    target_date = parse_date(case["testDate"], "testDate", iso3)
    candidates: list[tuple[float, dict[str, str]]] = []

    for row in rows:
        if (row.get("territorynamepernament") or "").strip().lower() != target_country:
            continue
        title = (row.get("officetitle") or "").strip().lower()
        if "president" not in title or "vice" in title:
            continue
        if (row.get("interregnum") or "").strip() in {"1", "1.0"}:
            continue
        if (row.get("officetypology") or "").strip() not in {"", "1", "1.0"}:
            continue
        start_text = (row.get("u_hdata_leader_date_date_in") or "").strip()
        end_text = (row.get("u_hdata_leader_date_date_out") or "").strip()
        if not start_text or not end_text:
            continue
        try:
            start = date.fromisoformat(start_text[:10])
            end = date.fromisoformat(end_text[:10])
        except ValueError:
            continue
        if not (start <= target_date <= end):
            continue
        score = name_similarity(case["wikidataPresident"], row.get("name1", ""))
        if score >= 0.70:
            candidates.append((score, row))

    if len(candidates) != 1:
        details = [f"{r.get('name1')} ({score:.3f})" for score, r in candidates]
        die(
            f"{iso3}: expected exactly one independent H-DATA match at "
            f"{case['testDate']}; found {len(candidates)}: {details}"
        )
    return candidates[0][1]


def validate_agreement(case: dict[str, str], hrow: dict[str, str]) -> None:
    iso3 = case["iso3"]
    agreement = case["agreementType"]
    if agreement not in ALLOWED_AGREEMENT_TYPES:
        die(f"{iso3}: unsupported agreementType {agreement!r}")

    w_start = parse_date(case["wikidataStart"], "wikidataStart", iso3)
    w_end = parse_date(case["wikidataEnd"], "wikidataEnd", iso3)
    h_start = parse_date(hrow["u_hdata_leader_date_date_in"], "H-DATA start", iso3)
    h_end = parse_date(hrow["u_hdata_leader_date_date_out"], "H-DATA end", iso3)
    overlap_start = parse_date(case["overlapStart"], "overlapStart", iso3)
    overlap_end = parse_date(case["overlapEnd"], "overlapEnd", iso3)
    test_date = parse_date(case["testDate"], "testDate", iso3)

    if w_start > w_end or h_start > h_end:
        die(f"{iso3}: source term has start date after end date")
    if not (overlap_start < test_date < overlap_end):
        die(f"{iso3}: testDate must be strictly inside the shared source interval")
    if overlap_start != max(w_start, h_start) or overlap_end != min(w_end, h_end):
        die(f"{iso3}: stored overlap does not equal the actual source intersection")

    # The benchmark's H-DATA values must be exact copies of the independent file.
    if case["hdataStart"] != h_start.isoformat() or case["hdataEnd"] != h_end.isoformat():
        die(f"{iso3}: benchmark H-DATA dates do not match hdata_plt_cleaned.csv")
    if normalize(case["hdataPresident"]) != normalize(hrow.get("name1", "")):
        die(f"{iso3}: benchmark H-DATA president does not match hdata_plt_cleaned.csv")

    if agreement == "EXACT_TWO_SOURCE":
        if w_start != h_start or w_end != h_end:
            die(f"{iso3}: EXACT_TWO_SOURCE row does not have exact term dates")
    elif agreement == "ONE_DAY_END_BOUNDARY_RESOLVED":
        if w_start != h_start or abs((w_end - h_end).days) != 1:
            die(f"{iso3}: expected same start and exactly one-day end-date difference")
    elif agreement == "SOURCE2_RIGHT_CENSORED_AT_2022":
        if w_start != h_start or h_end != date(2022, 12, 31) or w_end <= h_end:
            die(f"{iso3}: invalid right-censoring pattern")


def main() -> int:
    print("=" * 72)
    print("STEP 1/3 — PREPARE AND VALIDATE REDO DATA")
    print("=" * 72)
    print(f"Project folder: {ROOT}")

    for path in (WIKIDATA_FILE, HDATA_FILE):
        if not path.exists():
            die(f"Required input is missing: {path.name}")

    benchmark_fields, benchmark = read_csv(WIKIDATA_FILE)
    hdata_fields, hdata = read_csv(HDATA_FILE)

    missing = BENCHMARK_REQUIRED - set(benchmark_fields)
    if missing:
        die(f"{WIKIDATA_FILE.name} is missing columns: {sorted(missing)}")
    missing = HDATA_REQUIRED - set(hdata_fields)
    if missing:
        die(f"{HDATA_FILE.name} is missing columns: {sorted(missing)}")

    if len(benchmark) != 30:
        die(f"Expected exactly 30 benchmark rows; found {len(benchmark)}")
    iso3_values = [row["iso3"].strip() for row in benchmark]
    if len(set(iso3_values)) != 30 or set(iso3_values) != set(EXPECTED_ISO3):
        die("The benchmark does not contain the exact agreed 30-country set")

    country_rows: list[dict] = []
    wiki_rows: list[dict] = []
    hdata_rows: list[dict] = []
    case_rows: list[dict] = []
    agreement_counts = {key: 0 for key in sorted(ALLOWED_AGREEMENT_TYPES)}

    seen_capitals: set[str] = set()
    seen_country_qids: set[str] = set()

    for case in benchmark:
        iso3 = case["iso3"].strip()
        if case["validationStatus"] != "AGREE":
            die(f"{iso3}: validationStatus must be AGREE")
        for field in ("countryQID", "capitalQID", "officeQID", "wikidataPresidentQID"):
            validate_qid(case[field], field, iso3)
        capital_key = normalize(case["capital"])
        if capital_key in seen_capitals:
            die(f"{iso3}: duplicate normalized capital {case['capital']!r}")
        if case["countryQID"] in seen_country_qids:
            die(f"{iso3}: duplicate country QID {case['countryQID']}")
        seen_capitals.add(capital_key)
        seen_country_qids.add(case["countryQID"])

        hrow = select_hdata_match(case, hdata)
        validate_agreement(case, hrow)
        agreement_counts[case["agreementType"]] += 1

        # Independent source names must identify the same person strongly enough.
        score = name_similarity(case["wikidataPresident"], hrow.get("name1", ""))
        if score < 0.70:
            die(f"{iso3}: cross-source name similarity too low ({score:.3f})")

        country_rows.append({
            "capital": case["capital"],
            "country_code": iso3,
            "country_name": case["country"],
            "country_qid": case["countryQID"],
            "capital_qid": case["capitalQID"],
        })
        wiki_rows.append({
            "country_code": iso3,
            "country_name": case["country"],
            "office_qid": case["officeQID"],
            "president_qid": case["wikidataPresidentQID"],
            "president_name": case["wikidataPresident"],
            "start_date": case["wikidataStart"],
            "end_date": case["wikidataEnd"],
            "wikidata_source_path": case["wikidataSourcePath"],
        })
        hdata_rows.append({
            "country_code": iso3,
            "country_name": case["country"],
            "president_name": hrow.get("name1", "").strip(),
            "start_date": hrow["u_hdata_leader_date_date_in"],
            "end_date": hrow["u_hdata_leader_date_date_out"],
            "office_title": hrow.get("officetitle", "").strip(),
            "territory_code": hrow.get("territorycode", "").strip(),
            "hdata_country_id": hrow.get("u_hdata_leader_date_country_id", "").strip(),
        })
        case_rows.append({
            "task_id": f"redo_{iso3.lower()}",
            "country_code": iso3,
            "country_name": case["country"],
            "capital": case["capital"],
            "target_date": case["testDate"],
            "gold_president_name": case["wikidataPresident"],
            "wikidata_president_name": case["wikidataPresident"],
            "hdata_president_name": hrow.get("name1", "").strip(),
            "agreement_type": case["agreementType"],
        })

    expected_agreement_counts = {
        "EXACT_TWO_SOURCE": 26,
        "ONE_DAY_END_BOUNDARY_RESOLVED": 2,
        "SOURCE2_RIGHT_CENSORED_AT_2022": 2,
    }
    if agreement_counts != expected_agreement_counts:
        die(
            "Agreement-type counts changed unexpectedly. "
            f"Expected {expected_agreement_counts}; found {agreement_counts}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "country_reference.csv": (
            ["capital", "country_code", "country_name", "country_qid", "capital_qid"],
            country_rows,
        ),
        "wikidata_terms.csv": (
            ["country_code", "country_name", "office_qid", "president_qid",
             "president_name", "start_date", "end_date", "wikidata_source_path"],
            wiki_rows,
        ),
        "hdata_terms.csv": (
            ["country_code", "country_name", "president_name", "start_date",
             "end_date", "office_title", "territory_code", "hdata_country_id"],
            hdata_rows,
        ),
        "benchmark_cases.csv": (
            ["task_id", "country_code", "country_name", "capital", "target_date",
             "gold_president_name", "wikidata_president_name",
             "hdata_president_name", "agreement_type"],
            case_rows,
        ),
    }

    for name, (fields, rows) in outputs.items():
        write_csv_atomic(OUT_DIR / name, fields, rows)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "30-country real-data MCP redo preparation",
        "source_files": {
            WIKIDATA_FILE.name: {"sha256": sha256(WIKIDATA_FILE), "rows": len(benchmark)},
            HDATA_FILE.name: {"sha256": sha256(HDATA_FILE), "rows": len(hdata)},
        },
        "provenance": {
            "country_reference.csv": "Derived only from Wikidata QID/capital/country fields in wikidata_presidential_record.csv",
            "wikidata_terms.csv": "Derived only from Wikidata fields in wikidata_presidential_record.csv",
            "hdata_terms.csv": "Selected directly from hdata_plt_cleaned.csv using country, office, name, and test-date matching",
            "benchmark_cases.csv": "One fixed, internally validated test case per country; target dates are strictly inside shared source intervals",
        },
        "countries": EXPECTED_ISO3,
        "country_count": 30,
        "agreement_counts": agreement_counts,
        "outputs": {},
    }
    for name in outputs:
        fields, rows = read_csv(OUT_DIR / name)
        manifest["outputs"][name] = {
            "sha256": sha256(OUT_DIR / name),
            "rows": len(rows),
            "columns": fields,
        }

    manifest_path = OUT_DIR / "manifest.json"
    temp_manifest = manifest_path.with_suffix(".json.tmp")
    temp_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_manifest.replace(manifest_path)

    pass_text = (
        "PREPARE REDO DATA: PASS\n"
        "Countries: 30/30\n"
        "Reference datasets created: 3\n"
        "Independent source agreement checked: Wikidata-derived benchmark + H-DATA\n"
        "Exact two-source term cases: 26\n"
        "Documented boundary/censoring cases: 4\n"
        "No external data was invented by this script.\n"
    )
    (OUT_DIR / "PREPARE_REDO_DATA_PASS.txt").write_text(pass_text, encoding="utf-8")

    print("\nPREPARE REDO DATA: PASS")
    print("Countries: 30 / 30")
    print("Created three experiment references:")
    print("  redo_data/country_reference.csv")
    print("  redo_data/wikidata_terms.csv")
    print("  redo_data/hdata_terms.csv")
    print("Created benchmark cases: redo_data/benchmark_cases.csv")
    print("Created provenance manifest: redo_data/manifest.json")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nPREPARE REDO DATA: FAIL\n{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
