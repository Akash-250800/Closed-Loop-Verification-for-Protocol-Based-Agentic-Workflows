"""Final completeness/integrity verification for the MCP thesis redo.

Run after the targeted Phase C repair has completed:
    python 05_verify_corrected_redo.py
For an archival evidence-only check, without pytest or pip check:
    python 05_verify_corrected_redo.py --evidence-only

PASS means the planned experiments are complete, internally consistent, and
traceable. PASS does NOT mean every LLM trial succeeded; experimental failures
remain valid empirical outcomes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "redo_data"
OUT_DIR = ROOT / "redo_v4_final_verification"

EXPECTED_COUNTRIES = 30
EXPECTED_COUNTRY_CODES = {
    "FRA", "DEU", "ITA", "PRT", "IRL", "AUT", "FIN", "POL", "CZE", "SVK",
    "HUN", "HRV", "SVN", "EST", "LVA", "LTU", "GRC", "CYP", "ROU", "USA",
    "MEX", "BRA", "ARG", "CHL", "COL", "PER", "URY", "PRY", "ECU", "PAN",
}
MULTIREF_FAULTS = {
    "none", "type_mismatch", "wrong_tool", "transient_once", "persistent_corruption"
}
POLICIES = {"no_verification", "naive_retry", "closed_loop"}

EXPECTED = {
    "redo_v3_inventory_baseline": 480,
    "redo_v3_feedback": 72,
    "redo_v3_retry": 176,
    "redo_v3_bfcl": 60,
    "redo_v4_multiref_corrected": 450,
    "redo_v3_two_agent": 30,
}
EXPECTED_TOTAL = sum(EXPECTED.values())


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def as_int(value) -> int:
    if value in (None, ""):
        return 0
    return int(float(value))


def as_float(value) -> float:
    if value in (None, "", "N/A"):
        return 0.0
    return float(value)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def verify_prepared_data() -> dict:
    manifest_path = DATA_DIR / "manifest.json"
    if not manifest_path.exists():
        fail("redo_data/manifest.json is missing")
    manifest = read_json(manifest_path)
    if manifest.get("country_count") != 30:
        fail("Prepared data manifest does not report 30 countries")

    for source_name, info in manifest.get("source_files", {}).items():
        path = ROOT / source_name
        if not path.exists():
            fail(f"Prepared-data source missing: {source_name}")
        if sha256(path) != info.get("sha256"):
            fail(f"Prepared-data source changed after Step 1: {source_name}")
    for name, info in manifest.get("outputs", {}).items():
        path = DATA_DIR / name
        if not path.exists() or sha256(path) != info.get("sha256"):
            fail(f"Prepared reference changed or is missing: redo_data/{name}")

    cases = read_csv(DATA_DIR / "benchmark_cases.csv")
    codes = {r["country_code"] for r in cases}
    if len(cases) != 30 or codes != EXPECTED_COUNTRY_CODES:
        fail("benchmark_cases.csv is not the exact 30-country set")
    return manifest


def verify_legacy(folder_name: str, expected_rows: int) -> tuple[list[dict], dict]:
    folder = ROOT / folder_name
    manifest_path = folder / "manifest.json"
    trials_path = folder / "trials.csv"
    if not manifest_path.exists() or not trials_path.exists():
        fail(f"{folder_name}: missing manifest.json or trials.csv")
    manifest = read_json(manifest_path)
    rows = read_csv(trials_path)
    if manifest.get("completion_status") != "complete":
        fail(f"{folder_name}: manifest is not complete")
    if len(rows) != expected_rows:
        fail(f"{folder_name}: {len(rows)} rows found; expected {expected_rows}")
    return rows, manifest


def verify_inventory_baseline(rows: list[dict]) -> dict:
    tasks = {r["task_id"] for r in rows}
    faults = {r["fault"] for r in rows}
    conditions = {r["condition"] for r in rows}
    if len(tasks) != 20 or len(faults) != 8 or conditions != POLICIES:
        fail("Inventory baseline factor structure is not 20 tasks x 8 faults x 3 policies")
    keys = {(r["task_id"], r["fault"], r["condition"]) for r in rows}
    if len(keys) != 480:
        fail("Inventory baseline contains duplicate or missing task/fault/policy cells")

    # Methodological separation: no-fault rows are natural; every injected-fault row is seeded.
    for row in rows:
        seeded = as_int(row.get("controlled_seed"))
        if row["fault"] == "none" and seeded != 0:
            fail(f"Natural no-fault row was incorrectly seeded: {row['task_id']} {row['condition']}")
        if row["fault"] != "none" and seeded != 1:
            fail(f"Injected-fault row lacks canonical control seed: {row['task_id']} {row['fault']} {row['condition']}")
        if row["fault"] != "none" and as_int(row.get("injected_fault")) != 1:
            fail(f"Controlled fault was not actually injected: {row['task_id']} {row['fault']} {row['condition']}")

    # Direct answer to the client's retry question: an identical valid call must recover
    # after a one-shot transient transport failure under naive retry with budget 2.
    transient_faults = {"transient_api", "transient_timeout", "transient_rate_limit"}
    transient_naive = [
        r for r in rows
        if r["condition"] == "naive_retry" and r["fault"] in transient_faults
    ]
    if len(transient_naive) != 60:
        fail(f"Expected 60 transient naive-retry rows; found {len(transient_naive)}")
    if any(as_int(r.get("task_completed")) != 1 for r in transient_naive):
        bad = [(r["task_id"], r["fault"]) for r in transient_naive if as_int(r.get("task_completed")) != 1]
        fail(f"Naive retry failed a one-shot transient controlled call: {bad[:8]}")
    if any(as_int(r.get("injected_fault_recovered")) != 1 for r in transient_naive):
        fail("A transient naive-retry row completed without being recorded as injected-fault recovery")

    return {
        "unique_tasks": 20, "faults": 8, "policies": 3, "rows": 480,
        "transient_naive_rows": 60,
        "transient_naive_completed": 60,
    }


def verify_feedback(rows: list[dict]) -> dict:
    tasks = {r["task_id"] for r in rows}
    faults = {r["fault"] for r in rows}
    formats = {r["feedback_format"] for r in rows}
    if len(tasks) != 3 or len(faults) != 8 or formats != {"raw_trace", "schema_diff", "natural_language"}:
        fail("Feedback experiment must be 3 tasks x 8 faults x 3 feedback formats")
    if {r["condition"] for r in rows} != {"closed_loop"}:
        fail("Feedback-format experiment must contain closed_loop only")
    keys = {(r["task_id"], r["fault"], r["feedback_format"]) for r in rows}
    if len(keys) != 72:
        fail("Feedback experiment has duplicate or missing cells")
    for row in rows:
        seeded = as_int(row.get("controlled_seed"))
        if row["fault"] == "none" and seeded != 0:
            fail("Feedback no-fault row should be a natural LLM run")
        if row["fault"] != "none" and seeded != 1:
            fail("Feedback injected-fault row lacks canonical control seed")
    return {"unique_tasks": 3, "faults": 8, "feedback_formats": 3, "rows": 72}


def verify_bfcl(rows: list[dict]) -> dict:
    tasks = {r["task_id"] for r in rows}
    conditions = {r["condition"] for r in rows}
    if len(tasks) != 20:
        fail(f"BFCL must contain 20 unique tasks; found {len(tasks)}")
    if conditions != POLICIES:
        fail("BFCL must contain all three policies")
    if {r["fault"] for r in rows} != {"none"}:
        fail("BFCL redo should contain only the no-fault condition")
    counts = Counter(r["task_id"] for r in rows)
    if set(counts.values()) != {3}:
        fail("Each BFCL task must have exactly three policy runs")
    return {"unique_tasks": 20, "policy_runs": 60, "policies_per_task": 3}


def verify_custom(folder_name: str, expected_rows: int) -> tuple[list[dict], dict]:
    folder = ROOT / folder_name
    manifest_path = folder / "manifest.json"
    records_path = folder / "records.json"
    trials_path = folder / "trials.csv"
    for path in (manifest_path, records_path, trials_path):
        if not path.exists():
            fail(f"{folder_name}: required file missing: {path.name}")
    manifest = read_json(manifest_path)
    records = read_json(records_path)
    csv_rows = read_csv(trials_path)
    if manifest.get("completion_status") != "complete":
        fail(f"{folder_name}: manifest is not complete")
    if len(records) != expected_rows or len(csv_rows) != expected_rows:
        fail(
            f"{folder_name}: records/trials counts are {len(records)}/{len(csv_rows)}; "
            f"expected {expected_rows}/{expected_rows}"
        )
    return records, manifest


def verify_retry(rows: list[dict]) -> dict:
    if len({r["task_id"] for r in rows}) != 2:
        fail("Retry experiment must use exactly 2 inventory tasks")
    faults = {r["fault"] for r in rows}
    if len(faults) != 8:
        fail("Retry experiment must contain all 8 legacy fault conditions")
    by_condition = Counter(r["condition"] for r in rows)
    expected_counts = {"no_verification": 16, "naive_retry": 80, "closed_loop": 80}
    if dict(by_condition) != expected_counts:
        fail(f"Retry policy counts are wrong: expected {expected_counts}, found {dict(by_condition)}")

    for row in rows:
        seeded = as_int(row.get("controlled_seed"))
        if row["fault"] == "none" and seeded != 0:
            fail("Retry no-fault row should remain a natural end-to-end LLM run")
        if row["fault"] != "none" and seeded != 1:
            fail("Retry injected-fault row lacks canonical control seed")

    keys = set()
    for row in rows:
        budget = as_int(row.get("scheduled_retry_budget"))
        key = (row["task_id"], row["fault"], row["condition"], budget)
        if key in keys:
            fail(f"Retry experiment duplicate cell: {key}")
        keys.add(key)
        if row["condition"] == "no_verification" and budget != 0:
            fail("No-verification must appear once per task/fault at scheduled budget 0")
        if row["condition"] in {"naive_retry", "closed_loop"} and budget not in {0, 1, 2, 3, 4}:
            fail("Retry budget must be 0..4 for retry-capable policies")

    for task in {r["task_id"] for r in rows}:
        for fault in faults:
            subset = [r for r in rows if r["task_id"] == task and r["fault"] == fault]
            nv = [r for r in subset if r["condition"] == "no_verification"]
            nr = {as_int(r["scheduled_retry_budget"]) for r in subset if r["condition"] == "naive_retry"}
            cl = {as_int(r["scheduled_retry_budget"]) for r in subset if r["condition"] == "closed_loop"}
            if len(nv) != 1 or nr != {0, 1, 2, 3, 4} or cl != {0, 1, 2, 3, 4}:
                fail(f"Retry budget matrix incomplete for {task}/{fault}")
    return {"rows": 176, "tasks": 2, "faults": 8, "design_cells_per_task_fault": 11}


def first_two_fault_step_proposals(row: dict) -> tuple[str | None, str | None]:
    events = [
        e for e in row.get("events", [])
        if as_int(e.get("step")) == 2 and as_int(e.get("attempt")) in {1, 2}
    ]
    events.sort(key=lambda e: as_int(e.get("attempt")))
    if len(events) < 2:
        return None, None
    return events[0].get("proposal"), events[1].get("proposal")


def verify_multiref(rows: list[dict]) -> dict:
    tasks = {r["task_id"] for r in rows}
    countries = {r["country_code"] for r in rows}
    faults = {r["fault"] for r in rows}
    conditions = {r["condition"] for r in rows}
    if len(tasks) != 30 or countries != EXPECTED_COUNTRY_CODES:
        fail("30-country multi-reference experiment does not contain the exact country set")
    if faults != MULTIREF_FAULTS or conditions != POLICIES:
        fail("Multi-reference fault/policy factors are incomplete")
    keys = {(r["task_id"], r["fault"], r["condition"]) for r in rows}
    if len(keys) != 450:
        fail("Multi-reference matrix has duplicate or missing cells")
    counts = Counter((r["fault"], r["condition"]) for r in rows)
    if set(counts.values()) != {30}:
        fail("Each multi-reference fault/policy cell must contain exactly 30 countries")

    # Completed trials must actually have observed both independent sources.
    for row in rows:
        seeded = as_int(row.get("controlled_seed"))
        if row["fault"] == "none" and seeded != 0:
            fail(f"Natural multi-reference no-fault row was incorrectly seeded: {row['task_id']}")
        if row["fault"] != "none" and seeded != 1:
            fail(f"Injected multi-reference row lacks canonical control seed: {row['task_id']} {row['fault']}")
        if row["fault"] != "none" and not as_int(row.get("injected_fault")):
            fail(f"Multi-reference controlled fault was not injected: {row['task_id']} {row['fault']} {row['condition']}")
        if as_int(row.get("task_completed")) and not as_int(row.get("independent_sources_observed")):
            fail(f"Completed multi-reference row lacks both sources: {row['task_id']} {row['fault']} {row['condition']}")
        if row["condition"] == "naive_retry" and as_int(row.get("retry_count")):
            if as_int(row.get("same_candidate_retry_count")) != as_int(row.get("retry_count")):
                fail(f"Naive retry regenerated a candidate instead of replaying it: {row['task_id']} {row['fault']}")

    transient_naive = [
        r for r in rows
        if r["fault"] == "transient_once" and r["condition"] == "naive_retry"
    ]
    if len(transient_naive) != 30 or any(as_int(r.get("task_completed")) != 1 for r in transient_naive):
        fail("All 30 seeded transient_once naive-retry multi-reference rows must recover")
    persistent_naive = [
        r for r in rows
        if r["fault"] == "persistent_corruption" and r["condition"] == "naive_retry"
    ]
    if len(persistent_naive) != 30:
        fail("Expected 30 persistent_corruption naive-retry multi-reference rows")
    if any(as_int(r.get("task_completed")) != 0 for r in persistent_naive):
        fail("Naive retry unexpectedly repaired persistent_corruption without regeneration")

    # Mechanistic proof for the client's retry question: where a retry occurred on the injected
    # fault step, the naive policy must replay byte-identical proposal text.
    checked_transient = 0
    checked_persistent = 0
    for row in rows:
        if row["condition"] != "naive_retry" or not as_int(row.get("injected_fault")):
            continue
        if row["fault"] not in {"transient_once", "persistent_corruption"}:
            continue
        first, second = first_two_fault_step_proposals(row)
        if first is not None and second is not None:
            if first != second:
                fail(f"Naive retry candidate changed between attempts for {row['task_id']} {row['fault']}")
            if row["fault"] == "transient_once":
                checked_transient += 1
            else:
                checked_persistent += 1

    mechanism_file = ROOT / "redo_v3_multiref" / "NAIVE_RETRY_MECHANISM.txt"
    if not mechanism_file.exists():
        fail("Naive retry mechanism documentation is missing")

    return {
        "rows": 450,
        "countries": 30,
        "faults": 5,
        "policies": 3,
        "naive_transient_replay_pairs_observed": checked_transient,
        "naive_persistent_replay_pairs_observed": checked_persistent,
    }


def verify_repair_lineage(corrected_rows: list[dict]) -> dict:
    """Cross-check original controlled trials, repaired natural trials, and provenance.

    This intentionally does not change any empirical outcome.
    """
    original = ROOT / "redo_v3_multiref"
    repaired = ROOT / "redo_v4_phase_c_nofault"
    combined = ROOT / "redo_v4_multiref_corrected"
    paths = [
        original / "records.json", repaired / "records.json", combined / "records.json",
        repaired / "manifest.json", combined / "manifest.json",
        combined / "REPAIR_AUDIT.json",
        DATA_DIR / "benchmark_cases.csv", ROOT / "02_run_redo_experiments_FINAL.py",
    ]
    for path in paths:
        if not path.is_file():
            fail(f"Repair provenance file missing: {path.relative_to(ROOT)}")

    old = read_json(original / "records.json")
    fresh = read_json(repaired / "records.json")
    combo = corrected_rows
    repaired_manifest = read_json(repaired / "manifest.json")
    combo_manifest = read_json(combined / "manifest.json")
    audit = read_json(combined / "REPAIR_AUDIT.json")
    if len(old) != 450 or len(fresh) != 90 or len(combo) != 450:
        fail("Phase C lineage counts do not match 450 original, 90 replacements and 450 combined")
    if repaired_manifest.get("completion_status") != "complete":
        fail("Natural no-fault repair manifest is not complete")
    if combo_manifest.get("completion_status") != "complete":
        fail("Combined Phase C manifest is not complete")
    if audit.get("status") != "PASS":
        fail("Repair provenance audit is not marked PASS")
    if any(audit.get(k) != v for k, v in {
        "original_phase_c": 450, "unchanged_controlled_fault_trials": 360,
        "new_complete_nofault_trials": 90, "combined_phase_c": 450,
        "other_verified_v3_experiments": 818, "research_rows_for_dissertation": 1268,
    }.items()):
        fail("Repair provenance audit has an unexpected row-count declaration")

    original_hash = sha256(original / "records.json")
    replaced_hash = sha256(repaired / "records.json")
    combined_hash = sha256(combined / "records.json")
    runner_hash = sha256(ROOT / "02_run_redo_experiments_FINAL.py")
    prepared_hash = sha256(DATA_DIR / "benchmark_cases.csv")
    expected = {
        "original_phase_c_records": original_hash,
        "replacement_nofault_records": replaced_hash,
        "corrected_runner": runner_hash,
        "prepared_benchmark": prepared_hash,
    }
    for key, digest in expected.items():
        if combo_manifest.get("source_hashes", {}).get(key) != digest:
            fail(f"Combined Phase C manifest SHA256 does not match {key}")
        if audit.get("input_sha256", {}).get(key) != digest:
            fail(f"Repair audit SHA256 does not match {key}")
    if audit.get("combined_sha256") != combined_hash:
        fail("Corrected Phase C records file hash disagrees with repair audit")
    if repaired_manifest.get("runner_sha256") != runner_hash or repaired_manifest.get("benchmark_sha256") != prepared_hash:
        fail("The repair was not run against the expected runner and benchmark")
    if repaired_manifest.get("original_phase_c_sha256") != original_hash:
        fail("The repair was not run against these original Phase C records")

    def index(items: list[dict], context: str) -> dict:
        result = {}
        for row in items:
            key = (row["task_id"], row["fault"], row["condition"])
            if key in result:
                fail(f"{context}: duplicate Phase C key {key}")
            result[key] = row
        return result

    old_idx = index(old, "original")
    new_idx = index(fresh, "repair")
    combined_idx = index(combo, "corrected")
    if set(combined_idx) != set(old_idx):
        fail("Corrected matrix has missing or additional trial design cells")
    if len(new_idx) != 90 or {k[1] for k in new_idx} != {"none"}:
        fail("Replacement file has an invalid set of no-fault cells")
    if set(new_idx) != {k for k in old_idx if k[1] == "none"}:
        fail("Replacement file does not cover the exact original 90 no-fault cells")

    controlled_unchanged = 0
    natural_replaced = 0
    for key, row in combined_idx.items():
        if key in new_idx:
            # The consolidated copy adds a provenance label. All empirical fields
            # and attempt events must remain identical to the replacement record.
            label = row.get("result_origin")
            if label != "new_natural_nofault_grounded_merge":
                fail(f"Repaired natural trial has missing/incorrect origin label: {key}")
            without_label = {k: v for k, v in row.items() if k != "result_origin"}
            if without_label != new_idx[key]:
                fail(f"Repaired natural trial differs from replacement: {key}")
            if as_int(row.get("controlled_seed")) or as_int(row.get("injected_fault")):
                fail(f"Replacement trial was seeded or fault injected: {key}")
            natural_replaced += 1
        else:
            if row != old_idx[key]:
                fail(f"Original controlled trial was altered: {key}")
            if key[1] == "none" or not as_int(row.get("controlled_seed")):
                fail(f"Unchanged controlled trial has invalid fault/seed: {key}")
            controlled_unchanged += 1
    if controlled_unchanged != 360 or natural_replaced != 90:
        fail("Replacement and retention cardinalities are inconsistent")

    # Reconcile the tabular and JSON representation, including outcomes.
    for folder, idx in ((repaired, new_idx), (combined, combined_idx)):
        csv_rows = read_csv(folder / "trials.csv")
        if len(csv_rows) != len(idx):
            fail(f"{folder.name}: trials.csv length does not match records.json")
        seen = set()
        for row in csv_rows:
            key = (row["task_id"], row["fault"], row["condition"])
            if key not in idx or key in seen:
                fail(f"{folder.name}: duplicate or extraneous CSV row {key}")
            seen.add(key)
            original_row = idx[key]
            for field in ("task_completed", "controlled_seed", "injected_fault", "retry_count",
                          "same_candidate_retry_count", "independent_sources_observed"):
                if as_int(row.get(field)) != as_int(original_row.get(field)):
                    fail(f"{folder.name}: CSV/JSON mismatch for {key} field {field}")
        if seen != set(idx):
            fail(f"{folder.name}: CSV records have missing design cells")

        # Full per-trial JSONL event exports must match their records.
        events_path = folder / "attempt_events.jsonl"
        with events_path.open(encoding="utf-8") as fh:
            events = [json.loads(line) for line in fh if line.strip()]
        if index(events, f"{folder.name} JSONL") != idx:
            fail(f"{folder.name}: attempt_events.jsonl differs from canonical records")

        summary_rows = read_csv(folder / "summary.csv")
        for group in summary_rows:
            filtered = [r for r in idx.values() if all(
                str(r.get(k, "")) == str(v) for k, v in group.items()
                if k in {"condition", "fault", "feedback_format"} and v not in ("", None)
            )]
            n = as_int(group.get("n"))
            if n != len(filtered) or not n:
                fail(f"{folder.name}: summary row count disagrees with records")
            rate = sum(as_int(r.get("task_completed")) for r in filtered) / n
            if abs(as_float(group.get("completion_rate")) - rate) > 1e-10:
                fail(f"{folder.name}: summary completion disagrees with records")

    counts = {policy: sum(as_int(r.get("task_completed")) for r in combo if r["condition"] == policy)
              for policy in sorted(POLICIES)}
    nat = {policy: sum(as_int(r.get("task_completed")) for r in fresh if r["condition"] == policy)
           for policy in sorted(POLICIES)}
    for policy in POLICIES:
        info = audit.get("results_by_policy", {}).get(policy, {})
        if info.get("successful") != counts[policy] or info.get("total") != 150:
            fail(f"Audit results_by_policy overall mismatch for {policy}")
        if info.get("natural_nofault_successful") != nat[policy] or info.get("natural_nofault_total") != 30:
            fail(f"Audit natural outcome mismatch for {policy}")
    return {
        "original_rows": len(old), "corrected_rows": len(combo),
        "unchanged_controlled_rows": controlled_unchanged,
        "replaced_natural_rows": natural_replaced, "overall": counts, "nofault": nat,
        "sha256": {**expected, "corrected_combined_records": combined_hash},
        "outcome_note": "PASS certifies data lineage and consistency, not success of every trial",
    }


def verify_two_agent(rows: list[dict]) -> dict:
    tasks = {r["task_id"] for r in rows}
    countries = {r["country_code"] for r in rows}
    if len(tasks) != 30 or countries != EXPECTED_COUNTRY_CODES:
        fail("Two-agent experiment does not contain the exact 30-country set")
    if len(tasks) != len(rows):
        fail("Two-agent experiment has duplicate task rows")
    for row in rows:
        if as_int(row.get("task_completed")):
            if not as_int(row.get("agents_selected_independent_sources")):
                fail(f"Completed two-agent task did not use independent sources: {row['task_id']}")
            if row.get("agent_a_source_choice") == row.get("agent_b_source_choice"):
                fail(f"Completed two-agent task used the same source twice: {row['task_id']}")
    return {
        "rows": 30,
        "countries": 30,
        "completed": sum(as_int(r.get("task_completed")) for r in rows),
        "independent_source_successes": sum(as_int(r.get("agents_selected_independent_sources")) for r in rows),
    }


def completion_summary(rows: list[dict], group_fields: tuple[str, ...]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field, "") for field in group_fields)].append(row)
    output = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        record = {field: value for field, value in zip(group_fields, key)}
        complete = [as_int(r.get("task_completed")) for r in group]
        retries = [as_int(r.get("retry_count")) for r in group]
        latencies = [as_float(r.get("latency_seconds")) for r in group]
        token_values = [
            as_float(r.get("total_tokens")) for r in group
            if r.get("total_tokens") not in (None, "", "N/A")
        ]
        record.update({
            "n": len(group),
            "completed": sum(complete),
            "completion_rate": sum(complete) / len(group),
            "mean_retries": statistics.mean(retries) if retries else 0,
            "mean_latency_seconds": statistics.mean(latencies) if latencies else 0,
            "mean_tokens": statistics.mean(token_values) if len(token_values) == len(group) else "N/A",
        })
        output.append(record)
    return output


def run_check_command(command: list[str]) -> tuple[int, str]:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    text = (result.stdout or "") + (result.stderr or "")
    return result.returncode, text.strip()


def create_hash_manifest() -> dict:
    names = [
        "agents.py", "experiment.py", "verification.py", "feedback.py", "taxonomy.py",
        "benchmarks.py", "server.py", "requirements.txt",
        "wikidata_presidential_record.csv", "hdata_plt_cleaned.csv",
        "01_prepare_redo_data.py", "02_run_redo_experiments_FINAL.py", "03_final_verify_redo_FINAL.py", "04_repair_phase_c_nofault.py",
        "05_verify_corrected_redo.py",
    ]
    paths = [ROOT / name for name in names]
    paths += sorted(DATA_DIR.glob("*.csv")) + [
        DATA_DIR / "manifest.json",
        DATA_DIR / "inventory_control_seeds.json",
        DATA_DIR / "multiref_control_seeds.json",
    ]
    # Include the real experimental evidence in the integrity manifest as well as code
    # and prepared inputs. Never generate substitutes for missing expected files.
    for folder_name in EXPECTED:
        folder = ROOT / folder_name
        paths += [folder / "manifest.json", folder / "trials.csv"]
        if folder_name in {"redo_v3_retry", "redo_v4_multiref_corrected", "redo_v3_two_agent"}:
            paths.append(folder / "records.json")
        for optional_name in ("summary.csv", "attempt_events.jsonl", "NAIVE_RETRY_MECHANISM.txt",
                              "REPAIR_AUDIT.json", "REPAIR_CHECK.txt"):
            optional_path = folder / optional_name
            if optional_path.exists():
                paths.append(optional_path)
    # Preserve the unchanged original Phase C and the exact 90-replacement source.
    for folder_name in ("redo_v3_multiref", "redo_v4_phase_c_nofault"):
        folder = ROOT / folder_name
        for file_name in ("manifest.json", "trials.csv", "records.json",
                          "summary.csv", "attempt_events.jsonl"):
            paths.append(folder / file_name)
        mechanism = folder / "NAIVE_RETRY_MECHANISM.txt"
        if mechanism.exists():
            paths.append(mechanism)
    result = {}
    for path in paths:
        if not path.exists():
            fail(f"Hash-manifest file missing: {path.relative_to(ROOT)}")
        result[str(path.relative_to(ROOT))] = {
            "sha256": sha256(path), "bytes": path.stat().st_size
        }
    return result


def self_test() -> int:
    assert EXPECTED_TOTAL == 1268
    assert sum(EXPECTED.values()) == 1268
    assert EXPECTED["redo_v4_multiref_corrected"] == 30 * 5 * 3
    assert EXPECTED["redo_v3_bfcl"] == 20 * 3
    assert EXPECTED["redo_v3_retry"] == 2 * 8 * 11
    if (DATA_DIR / "benchmark_cases.csv").exists():
        cases = read_csv(DATA_DIR / "benchmark_cases.csv")
        assert len(cases) == 30
        assert {r["country_code"] for r in cases} == EXPECTED_COUNTRY_CODES
    print("05_verify_corrected_redo.py SELF-TEST: PASS")
    print(f"Expected total redo research rows: {EXPECTED_TOTAL}")
    return 0


def main(evidence_only: bool = False) -> int:
    print("=" * 78)
    print("CORRECTED REDO — FINAL CONSOLIDATED VERIFICATION")
    print("=" * 78)
    print(f"Project folder: {ROOT}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data_manifest = verify_prepared_data()

    inventory, inv_manifest = verify_legacy("redo_v3_inventory_baseline", 480)
    feedback, fb_manifest = verify_legacy("redo_v3_feedback", 72)
    bfcl, bfcl_manifest = verify_legacy("redo_v3_bfcl", 60)
    retry, retry_manifest = verify_custom("redo_v3_retry", 176)
    multiref, multiref_manifest = verify_custom("redo_v4_multiref_corrected", 450)
    repair_evidence = verify_repair_lineage(multiref)
    two_agent, two_agent_manifest = verify_custom("redo_v3_two_agent", 30)

    evidence = {
        "inventory": verify_inventory_baseline(inventory),
        "feedback": verify_feedback(feedback),
        "retry": verify_retry(retry),
        "bfcl": verify_bfcl(bfcl),
        "multiref": verify_multiref(multiref),
        "phase_c_repair": repair_evidence,
        "two_agent": verify_two_agent(two_agent),
    }

    actual_total = sum(len(rows) for rows in (inventory, feedback, retry, bfcl, multiref, two_agent))
    if actual_total != EXPECTED_TOTAL:
        fail(f"Grand total is {actual_total}; expected {EXPECTED_TOTAL}")

    # Final software tests run after the evidence exists.
    # A global `pip check` can report unrelated Anaconda/system-package conflicts.
    # Record it for transparency, but do not reject valid thesis evidence because
    # an unrelated package elsewhere in the environment has a dependency warning.
    if evidence_only:
        pip_code, pip_text = None, "NOT RUN (evidence-only mode)"
        pytest_code, pytest_text = None, "NOT RUN (evidence-only mode)"
    else:
        pip_code, pip_text = run_check_command([sys.executable, "-m", "pip", "check"])
        pytest_code, pytest_text = run_check_command([sys.executable, "-m", "pytest", "-q"])
        if pytest_code != 0:
            fail("pytest failed:\n" + pytest_text)

    counts_rows = [
        {"experiment": name, "expected_rows": EXPECTED[name], "actual_rows": EXPECTED[name], "status": "PASS"}
        for name in EXPECTED
    ]
    counts_rows.append({"experiment": "GRAND_TOTAL", "expected_rows": EXPECTED_TOTAL, "actual_rows": actual_total, "status": "PASS"})
    write_csv(OUT_DIR / "EXPERIMENT_COUNTS.csv", counts_rows)

    write_csv(
        OUT_DIR / "MULTIREF_POLICY_SUMMARY.csv",
        completion_summary(multiref, ("fault", "condition")),
    )
    write_csv(
        OUT_DIR / "RETRY_BUDGET_SUMMARY.csv",
        completion_summary(retry, ("condition", "scheduled_retry_budget")),
    )
    write_csv(
        OUT_DIR / "BFCL_POLICY_SUMMARY.csv",
        completion_summary(bfcl, ("condition",)),
    )
    write_csv(
        OUT_DIR / "FEEDBACK_FORMAT_SUMMARY.csv",
        completion_summary(feedback, ("feedback_format",)),
    )
    write_csv(
        OUT_DIR / "TWO_AGENT_SUMMARY.csv",
        [{
            "n": len(two_agent),
            "completed": sum(as_int(r.get("task_completed")) for r in two_agent),
            "completion_rate": sum(as_int(r.get("task_completed")) for r in two_agent) / len(two_agent),
            "independent_source_selections": sum(as_int(r.get("agents_selected_independent_sources")) for r in two_agent),
        }],
    )

    write_csv(
        OUT_DIR / "MULTIREF_POLICY_OVERALL.csv",
        completion_summary(multiref, ("condition",)),
    )
    write_csv(
        OUT_DIR / "MULTIREF_NATURAL_NOFAULT.csv",
        completion_summary([r for r in multiref if r["fault"] == "none"], ("condition",)),
    )
    atomic_text(OUT_DIR / "PHASE_C_REPAIR_LINEAGE.json", json.dumps(repair_evidence, indent=2))

    hash_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": create_hash_manifest(),
    }
    atomic_text(
        OUT_DIR / "PACKAGE_HASHES_REDO.json",
        json.dumps(hash_manifest, indent=2, ensure_ascii=False),
    )

    report_lines = [
        "MCP THESIS REDO — FINAL COMPLETENESS AND INTEGRITY CHECK",
        "=" * 61,
        "",
        ("EVIDENCE STATUS: PASS (software tests not run)" if evidence_only else "FINAL STATUS: PASS"),
        "",
        "PASS means the planned evidence is complete and internally consistent.",
        "PASS does not mean every LLM trial succeeded; failed trials remain empirical outcomes.",
        "",
        "DATA",
        "- Fixed benchmark countries: 30/30",
        "- Real reference origins: Wikidata-derived benchmark + independent H-DATA",
        f"- Agreement structure: {data_manifest.get('agreement_counts')}",
        "",
        "EXPERIMENT COUNTS",
        f"- Inventory policy baseline: {len(inventory)}",
        f"- Feedback-format ablation: {len(feedback)}",
        f"- Corrected retry-budget ablation: {len(retry)}",
        f"- BFCL: 20 unique tasks, {len(bfcl)} policy-runs",
        f"- 30-country two-source policy matrix: {len(multiref)}",
        f"- Two-agent source-choice trials: {len(two_agent)}",
        f"- GRAND TOTAL: {actual_total}",
        "",
        "TARGETED PHASE C REPAIR",
        "- The original redo_v3_multiref evidence is preserved and hash-checked.",
        "- Corrected Phase C evidence comes from redo_v4_multiref_corrected only.",
        "- The targeted redo_v4_phase_c_nofault contains exactly 90 replacement rows.",
        "- All 360 controlled fault rows are identical to the original records.",
        "- The 90 replacement empirical records match, with added consolidated provenance labels.",
        f"- Phase C no verification complete: {repair_evidence['overall']['no_verification']}/150",
        f"- Phase C naive retry complete: {repair_evidence['overall']['naive_retry']}/150",
        f"- Phase C closed loop complete: {repair_evidence['overall']['closed_loop']}/150",
        f"- Natural no fault no verification: {repair_evidence['nofault']['no_verification']}/30",
        f"- Natural no fault naive retry: {repair_evidence['nofault']['naive_retry']}/30",
        f"- Natural no fault closed loop: {repair_evidence['nofault']['closed_loop']}/30",
        "",
        "RETRY-BUDGET CORRECTION",
        "- The old 237-vs-240 ambiguity is not reused.",
        "- New design has exactly 176 cells.",
        "- No-verification is measured once per task/fault because retry budget is not applicable.",
        "- Naive-retry and closed-loop are each measured at budgets 0,1,2,3,4.",
        "",
        "CONTROLLED-FAULT METHODOLOGY",
        "- No-fault rows are genuine end-to-end LLM runs.",
        "- Injected-fault rows begin from canonical task-specification control calls/chains.",
        "- Canonical controls are not counted as LLM-generated calls; natural no-fault rows are LLM-driven.",
        "- This prevents unrelated natural generation errors from being mislabeled as recovery failures.",
        f"- Inventory transient naive-retry recoveries: {evidence['inventory']['transient_naive_completed']}/60",
        "",
        "NAIVE RETRY MECHANISM",
        "- Naive retry replays the identical failed proposal; it does not regenerate it.",
        "- transient_once blocks only attempt 1, so the same valid call can succeed on attempt 2.",
        "- persistent_corruption keeps the corrupted country code on every naive retry.",
        f"- Observed transient proposal-replay pairs: {evidence['multiref']['naive_transient_replay_pairs_observed']}",
        f"- Observed persistent proposal-replay pairs: {evidence['multiref']['naive_persistent_replay_pairs_observed']}",
        "",
        "BFCL WORDING",
        "- Unique BFCL tasks: 20",
        "- Policy-runs: 60 (20 tasks x 3 policies)",
        "",
        "TWO-AGENT DESIGN",
        "- Both agents receive the full MCP catalog.",
        "- Agent A chooses the first evidence source.",
        "- Agent B must independently select the other source.",
        "- Final decision is evaluated against both retrieved source records.",
        f"- Completed two-agent trials: {evidence['two_agent']['completed']}/30",
        f"- Independent-source selections: {evidence['two_agent']['independent_source_successes']}/30",
        "",
        "SOFTWARE CHECKS",
        ("- pip check: NOT RUN" if evidence_only else ("- pip check: PASS" if pip_code == 0 else "- pip check: WARNING (review captured output)")),
        ("- pytest: NOT RUN" if evidence_only else "- pytest: PASS"),
        f"- pip check output: {pip_text.replace(chr(10), ' | ')}",
        f"- pytest output: {pytest_text.replace(chr(10), ' | ')}",
        "",
        "Generated verification files are in redo_v4_final_verification/.",
    ]
    atomic_text(OUT_DIR / "FINAL_REDO_CHECK.txt", "\n".join(report_lines) + "\n")

    print("\n" + ("CORRECTED REDO EVIDENCE VERIFICATION: PASS (software checks not run)"
                  if evidence_only else "CORRECTED FINAL REDO VERIFICATION: PASS"))
    print(f"Grand total research rows: {actual_total} / {EXPECTED_TOTAL}")
    print("BFCL: 20 unique tasks / 60 policy-runs")
    print("Multi-reference: 30 countries / 450 policy-fault runs")
    print("Two-agent: 30 country trials")
    print(f"Report: {OUT_DIR / 'FINAL_REDO_CHECK.txt'}")
    return 0


def cli() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--evidence-only", action="store_true", help="Offline archival check without pip or pytest")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    try:
        return main(evidence_only=args.evidence_only)
    except Exception as exc:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        message = f"CORRECTED FINAL REDO VERIFICATION: FAIL\n{type(exc).__name__}: {exc}\n"
        atomic_text(OUT_DIR / "FINAL_REDO_CHECK.txt", message)
        print("\n" + message, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
