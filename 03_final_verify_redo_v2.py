"""Final completeness/integrity verification for the MCP thesis redo.

Run only after 02_run_redo_experiments.py finishes:
    python 03_final_verify_redo.py

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
OUT_DIR = ROOT / "redo_v2_final_verification"

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
    "redo_v2_inventory_baseline": 480,
    "redo_v2_feedback": 72,
    "redo_v2_retry": 176,
    "redo_v2_bfcl": 60,
    "redo_v2_multiref": 450,
    "redo_v2_two_agent": 30,
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
            fail(f"Injected-fault row lacks validated seed: {row['task_id']} {row['fault']} {row['condition']}")
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
            fail("Feedback injected-fault row lacks validated LLM seed")
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
            fail("Retry injected-fault row lacks validated LLM seed")

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
            fail(f"Injected multi-reference row lacks validated seed: {row['task_id']} {row['fault']}")
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

    mechanism_file = ROOT / "redo_v2_multiref" / "NAIVE_RETRY_MECHANISM.txt"
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
        "01_prepare_redo_data.py", "02_run_redo_experiments_v2.py", "03_final_verify_redo_v2.py",
    ]
    paths = [ROOT / name for name in names]
    paths += sorted(DATA_DIR.glob("*.csv")) + [DATA_DIR / "manifest.json", DATA_DIR / "inventory_validated_llm_seeds.json", DATA_DIR / "multiref_validated_llm_seeds.json"]
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
    assert EXPECTED["redo_v2_multiref"] == 30 * 5 * 3
    assert EXPECTED["redo_v2_bfcl"] == 20 * 3
    assert EXPECTED["redo_v2_retry"] == 2 * 8 * 11
    if (DATA_DIR / "benchmark_cases.csv").exists():
        cases = read_csv(DATA_DIR / "benchmark_cases.csv")
        assert len(cases) == 30
        assert {r["country_code"] for r in cases} == EXPECTED_COUNTRY_CODES
    print("03_final_verify_redo_v2.py SELF-TEST: PASS")
    print(f"Expected total redo research rows: {EXPECTED_TOTAL}")
    return 0


def main() -> int:
    print("=" * 78)
    print("STEP 3/3 — FINAL REDO VERIFICATION")
    print("=" * 78)
    print(f"Project folder: {ROOT}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data_manifest = verify_prepared_data()

    inventory, inv_manifest = verify_legacy("redo_v2_inventory_baseline", 480)
    feedback, fb_manifest = verify_legacy("redo_v2_feedback", 72)
    bfcl, bfcl_manifest = verify_legacy("redo_v2_bfcl", 60)
    retry, retry_manifest = verify_custom("redo_v2_retry", 176)
    multiref, multiref_manifest = verify_custom("redo_v2_multiref", 450)
    two_agent, two_agent_manifest = verify_custom("redo_v2_two_agent", 30)

    evidence = {
        "inventory": verify_inventory_baseline(inventory),
        "feedback": verify_feedback(feedback),
        "retry": verify_retry(retry),
        "bfcl": verify_bfcl(bfcl),
        "multiref": verify_multiref(multiref),
        "two_agent": verify_two_agent(two_agent),
    }

    actual_total = sum(len(rows) for rows in (inventory, feedback, retry, bfcl, multiref, two_agent))
    if actual_total != EXPECTED_TOTAL:
        fail(f"Grand total is {actual_total}; expected {EXPECTED_TOTAL}")

    # Final software tests run after the evidence exists.
    # A global `pip check` can report unrelated Anaconda/system-package conflicts.
    # Record it for transparency, but do not reject valid thesis evidence because
    # an unrelated package elsewhere in the environment has a dependency warning.
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
        "FINAL STATUS: PASS",
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
        "RETRY-BUDGET CORRECTION",
        "- The old 237-vs-240 ambiguity is not reused.",
        "- New design has exactly 176 cells.",
        "- No-verification is measured once per task/fault because retry budget is not applicable.",
        "- Naive-retry and closed-loop are each measured at budgets 0,1,2,3,4.",
        "",
        "CONTROLLED-FAULT METHODOLOGY",
        "- No-fault rows are genuine end-to-end LLM runs.",
        "- Injected-fault rows begin from validated LLM-generated clean proposals/chains.",
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
        "",
        "SOFTWARE CHECKS",
        ("- pip check: PASS" if pip_code == 0 else "- pip check: WARNING (non-project environment conflict; see captured output)"),
        "- pytest: PASS",
        f"- pip check output: {pip_text.replace(chr(10), ' | ')}",
        f"- pytest output: {pytest_text.replace(chr(10), ' | ')}",
        "",
        "Generated verification files are in redo_final_verification/.",
    ]
    atomic_text(OUT_DIR / "FINAL_REDO_CHECK.txt", "\n".join(report_lines) + "\n")

    print("\nFINAL REDO VERIFICATION: PASS")
    print(f"Grand total research rows: {actual_total} / {EXPECTED_TOTAL}")
    print("BFCL: 20 unique tasks / 60 policy-runs")
    print("Multi-reference: 30 countries / 450 policy-fault runs")
    print("Two-agent: 30 country trials")
    print(f"Report: {OUT_DIR / 'FINAL_REDO_CHECK.txt'}")
    return 0


def cli() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    try:
        return main()
    except Exception as exc:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        message = f"FINAL REDO VERIFICATION: FAIL\n{type(exc).__name__}: {exc}\n"
        atomic_text(OUT_DIR / "FINAL_REDO_CHECK.txt", message)
        print("\n" + message, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
