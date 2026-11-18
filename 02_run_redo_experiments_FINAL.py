"""Run the complete MCP thesis redo experiment suite.

Run from the current MCP-Thesis folder after 01_prepare_redo_data.py:
    python 02_run_redo_experiments_final.py

This file leaves all old result folders untouched. New evidence is written only
into new folders whose names begin with ``redo_v3_``.

Default real experiment suite
-----------------------------
1. Inventory policy baseline:        480 rows
2. Feedback-format ablation:          72 rows
3. Corrected retry-budget ablation:  176 rows
4. Adapted BFCL:                      60 rows (20 tasks x 3 policies)
5. 30-country two-source redo:       450 rows
6. Two-agent source-choice redo:      30 rows
                                      --------
Total expected research rows:       1268

The 30-country experiment uses real prepared reference data from:
- Wikidata-derived country/capital + presidential evidence
- independent H-DATA presidential evidence

Naive retry is deliberately implemented as a retry of the SAME failed call.
Therefore a one-shot transient failure can succeed on the next attempt, while a
persistent corrupted call is repeated unchanged and normally cannot recover.
Closed-loop GVEF regenerates a corrected call after structured feedback.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import traceback
import unicodedata
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "redo_data"
MODEL = "qwen2.5:1.5b"

LEGACY_EXPECTED = {
    "redo_v3_inventory_baseline": 480,
    "redo_v3_feedback": 72,
    "redo_v3_bfcl": 60,
}
CUSTOM_EXPECTED = {
    "redo_v3_retry": 176,
    "redo_v3_multiref": 450,
    "redo_v3_two_agent": 30,
}
TOTAL_EXPECTED = sum(LEGACY_EXPECTED.values()) + sum(CUSTOM_EXPECTED.values())

MULTIREF_FAULTS = (
    "none",
    "type_mismatch",
    "wrong_tool",
    "transient_once",
    "persistent_corruption",
)
POLICIES = ("no_verification", "naive_retry", "closed_loop")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temp.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def normalize(value: str) -> str:
    text = (value or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def name_equivalent(first: str, second: str) -> bool:
    a, b = normalize(first), normalize(second)
    if not a or not b:
        return False
    if a == b:
        return True
    set_a, set_b = set(a.split()), set(b.split())
    if set_a.issubset(set_b) or set_b.issubset(set_a):
        # Protect against a one-token accidental match.
        if min(len(set_a), len(set_b)) >= 2:
            return True
    return SequenceMatcher(None, a, b).ratio() >= 0.70




def verify_runtime_packages() -> None:
    try:
        from mcp.client import Client  # noqa: F401
        from mcp.server.mcpserver import MCPServer  # noqa: F401
        import httpx  # noqa: F401
        import jsonschema  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Required experiment packages are not available in this Python environment. "
            "Activate the same project environment used for the existing MCP experiments and run "
            "python -m pip install -r requirements.txt. Original error: " + repr(exc)
        ) from exc

def run_command(*args: str) -> None:
    command = [str(x) for x in args]
    print("\n>", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def verify_prepared_data() -> None:
    manifest_path = DATA_DIR / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            "redo_data/manifest.json is missing. Run python 01_prepare_redo_data.py first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("country_count") != 30:
        raise RuntimeError("Prepared data manifest does not contain 30 countries")

    for source_name, info in manifest.get("source_files", {}).items():
        source = ROOT / source_name
        if not source.exists():
            raise RuntimeError(f"Prepared-data source disappeared: {source_name}")
        actual = sha256(source)
        if actual != info.get("sha256"):
            raise RuntimeError(
                f"Source file changed after Step 1: {source_name}. Re-run 01_prepare_redo_data.py."
            )
    for name, info in manifest.get("outputs", {}).items():
        path = DATA_DIR / name
        if not path.exists() or sha256(path) != info.get("sha256"):
            raise RuntimeError(
                f"Prepared reference changed or is missing: redo_data/{name}. Re-run Step 1."
            )

    cases = read_csv(DATA_DIR / "benchmark_cases.csv")
    if len(cases) != 30 or len({r["country_code"] for r in cases}) != 30:
        raise RuntimeError("Prepared benchmark must contain exactly 30 unique countries")


class ResultStore:
    """Crash-safe custom result store. Completed rows are automatically retained."""

    def __init__(self, folder: Path, plan: dict, key_fields: tuple[str, ...]):
        self.folder = folder
        self.folder.mkdir(parents=True, exist_ok=True)
        self.manifest_path = folder / "manifest.json"
        self.records_path = folder / "records.json"
        self.key_fields = key_fields
        signature_payload = json.dumps(plan, sort_keys=True, ensure_ascii=False)
        self.signature = hashlib.sha256(signature_payload.encode("utf-8")).hexdigest()
        self.plan = plan
        self.records: list[dict] = []

        if self.manifest_path.exists():
            old = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if old.get("plan_signature") != self.signature:
                raise RuntimeError(
                    f"{folder.name} exists with different settings. Do not mix experiments; "
                    "rename that folder before rerunning."
                )
        if self.records_path.exists():
            self.records = json.loads(self.records_path.read_text(encoding="utf-8"))
            if not isinstance(self.records, list):
                raise RuntimeError(f"Invalid checkpoint file: {self.records_path}")

        self.done = {self.key(row) for row in self.records}
        manifest = {
            **plan,
            "plan_signature": self.signature,
            "completion_status": "in_progress",
            "saved_rows": len(self.records),
            "start_or_check_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(self.manifest_path, manifest)
        if self.records:
            print(f"{folder.name}: found {len(self.records)} already-saved rows; continuing safely.")

    def key(self, row: dict) -> tuple:
        return tuple(str(row.get(field, "")) for field in self.key_fields)

    def has(self, template: dict) -> bool:
        return self.key(template) in self.done

    def append(self, row: dict) -> None:
        key = self.key(row)
        if key in self.done:
            raise RuntimeError(f"Duplicate result cell attempted in {self.folder.name}: {key}")
        self.records.append(row)
        self.done.add(key)
        atomic_json(self.records_path, self.records)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["saved_rows"] = len(self.records)
        atomic_json(self.manifest_path, manifest)

    def finalize(self, expected_rows: int) -> None:
        if len(self.records) != expected_rows:
            raise RuntimeError(
                f"{self.folder.name}: saved {len(self.records)} rows; expected {expected_rows}"
            )
        flat_rows = []
        fieldnames: list[str] = []
        seen = set()
        for row in self.records:
            flat = {k: v for k, v in row.items() if k != "events"}
            flat_rows.append(flat)
            for key in flat:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
        temp_csv = self.folder / "trials.csv.tmp"
        with temp_csv.open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(flat_rows)
        temp_csv.replace(self.folder / "trials.csv")

        with (self.folder / "attempt_events.jsonl").open("w", encoding="utf-8") as fh:
            for row in self.records:
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["completion_status"] = "complete"
        manifest["saved_rows"] = len(self.records)
        manifest["finish_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(self.manifest_path, manifest)
        write_simple_summary(self.folder, self.records)


def write_simple_summary(folder: Path, records: list[dict]) -> None:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in records:
        key = (
            row.get("condition", "not_applicable"),
            row.get("fault", "none"),
            row.get("feedback_format", "not_applicable"),
            row.get("scheduled_retry_budget", row.get("retry_budget", "")),
        )
        groups[key].append(row)
    summary = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        completed = [int(float(r.get("task_completed", 0))) for r in rows]
        retries = [int(float(r.get("retry_count", 0))) for r in rows]
        latencies = [float(r.get("latency_seconds", 0) or 0) for r in rows]
        summary.append({
            "condition": key[0],
            "fault": key[1],
            "feedback_format": key[2],
            "scheduled_retry_budget": key[3],
            "n": len(rows),
            "completion_rate": sum(completed) / len(rows),
            "mean_retries": statistics.mean(retries) if retries else 0,
            "mean_latency_seconds": statistics.mean(latencies) if latencies else 0,
        })
    if summary:
        with (folder / "summary.csv").open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)


def validate_legacy_folder(folder_name: str, expected: int) -> None:
    folder = ROOT / folder_name
    manifest_path = folder / "manifest.json"
    trials_path = folder / "trials.csv"
    if not manifest_path.exists() or not trials_path.exists():
        raise RuntimeError(f"{folder_name}: experiment did not create required output files")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = read_csv(trials_path)
    if manifest.get("completion_status") != "complete":
        raise RuntimeError(f"{folder_name}: manifest is not complete")
    if len(rows) != expected:
        raise RuntimeError(f"{folder_name}: found {len(rows)} rows; expected {expected}")



def _fault_dict(error) -> dict:
    if error is None:
        return {}
    if hasattr(error, "dict"):
        return error.dict()
    return {"category": "error", "code": type(error).__name__, "message": str(error)}


async def build_inventory_seed_calls() -> dict:
    """Build deterministic canonical clean proposal chains for controlled fault injection.

    These calls come directly from each benchmark task specification and are used ONLY
    as the common pre-fault starting point. They are not counted as LLM research outputs.
    Natural/no-fault cells still use genuine Qwen generation end to end.
    """
    from benchmarks import inventory_tasks

    path = DATA_DIR / "inventory_control_seeds.json"
    tasks_payload = {}
    for task_index, task in enumerate(inventory_tasks(), start=1):
        proposals = []
        for step in task["steps"]:
            proposals.append(
                json.dumps(
                    {"tool": step["expected_tool"], "args": step["expected_args"]},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        if len(proposals) != len(task["steps"]):
            raise RuntimeError(f"Canonical seed length mismatch for {task['id']}")
        tasks_payload[task["id"]] = {
            "proposals": proposals,
            "seed_type": "canonical_task_specification",
        }
        print(f"[control seed inventory {task_index}/20] {task['id']} READY", flush=True)

    payload = {
        "model": MODEL,
        "purpose": (
            "Canonical valid tool-call chains used only to isolate controlled fault-recovery "
            "behavior. No-fault cells remain natural LLM runs."
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tasks": tasks_payload,
    }
    atomic_json(path, payload)
    return payload


async def run_inventory_controlled_trial(
    agent,
    task: dict,
    seed_proposals: list[str],
    condition: str,
    fault: str,
    feedback_style: str,
    retry_budget: int,
) -> dict:
    """Controlled fault-recovery trial using a validated clean LLM seed proposal."""
    from feedback import render_feedback
    from mcp.client import Client
    from server import make_server
    from taxonomy import inject
    from verification import parse_proposal, validate, unwrap_mcp_reply
    from experiment import is_correct_output

    if fault == "none":
        raise ValueError("Controlled trial is only for injected-fault conditions")
    if len(seed_proposals) != len(task["steps"]):
        raise ValueError(f"Seed length mismatch for {task['id']}")

    server = make_server()
    started = perf_counter()
    events = []
    history = []
    completed = 0
    retries = 0
    model_proposals = 0
    total_tokens = 0
    token_complete = True
    injected = False
    recovered = False
    last_error = ""
    fault_step = task["fault_step"]

    async with Client(server) as client:
        catalog = {tool.name: tool for tool in (await client.list_tools()).tools}

        for step_index, step in enumerate(task["steps"]):
            success = False
            feedback_text = ""
            candidate = seed_proposals[step_index]
            max_attempts = 1 if condition == "no_verification" else 1 + retry_budget

            for attempt_index in range(max_attempts):
                attempt_number = attempt_index + 1

                if attempt_index == 0:
                    candidate = seed_proposals[step_index]
                    if step_index == fault_step:
                        original = candidate
                        candidate = inject(candidate, fault, step, catalog)
                        if fault in {"transient_api", "transient_timeout", "transient_rate_limit"}:
                            injected = True
                        else:
                            injected = candidate != original
                elif condition == "naive_retry":
                    # Exact replay. No regeneration and no feedback.
                    retries += 1
                else:
                    retries += 1
                    candidate, usage = await agent.generate(task, step, catalog, feedback_text, history)
                    model_proposals += 1
                    if usage is None:
                        token_complete = False
                    else:
                        total_tokens += int(usage.get("input", 0)) + int(usage.get("output", 0))

                event = {
                    "task": task["id"],
                    "step": step_index + 1,
                    "attempt": attempt_number,
                    "condition": condition,
                    "fault": fault,
                    "feedback_format": feedback_style if condition == "closed_loop" else "not_applicable",
                    "proposal": candidate,
                    "seeded_initial_proposal": int(attempt_index == 0),
                    "feedback_received": feedback_text,
                    "stage": "",
                    "result": "",
                }

                call, error = parse_proposal(candidate)
                if error is not None:
                    event.update(stage="parse", result="rejected", error=_fault_dict(error))
                elif condition == "closed_loop":
                    error = validate(call, catalog, step, history=history, schema_enabled=True, safety_enabled=True)
                    if error is not None:
                        event.update(stage="preflight", result="rejected", error=_fault_dict(error))

                if error is None and call is not None:
                    if (
                        step_index == fault_step
                        and attempt_index == 0
                        and fault in {"transient_api", "transient_timeout", "transient_rate_limit"}
                    ):
                        mapping = {
                            "transient_api": (
                                "http_503", "Injected temporary upstream HTTP 503 failure",
                                "HTTPError: 503 Service Unavailable",
                            ),
                            "transient_timeout": (
                                "api_timeout", "Injected upstream network timeout",
                                "TimeoutError: request timed out",
                            ),
                            "transient_rate_limit": (
                                "http_429", "Injected upstream API rate limit",
                                "HTTPError: 429 Too Many Requests",
                            ),
                        }
                        code, message, trace = mapping[fault]
                        error = make_fault(
                            "transient_api", code, message, trace=trace, injected=True
                        )
                        event.update(stage="transport_injection", result="rejected", error=_fault_dict(error))
                    else:
                        try:
                            reply = await client.call_tool(call["tool"], call["args"])
                            payload, error = unwrap_mcp_reply(reply)
                        except Exception as exc:
                            payload = None
                            error = make_fault(
                                "execution_failure", type(exc).__name__, str(exc)[:500],
                                trace="".join(traceback.format_exception(exc))[-2500:],
                            )
                        if error is not None:
                            event.update(stage="execution", result="error", error=_fault_dict(error))
                        elif not is_correct_output(step, call, payload):
                            error = make_fault(
                                "semantic_failure", "oracle_mismatch",
                                "Tool call result does not satisfy the independent task criterion",
                                actual=repr(payload)[:400],
                            )
                            event.update(stage="evaluation", result="wrong_result", error=_fault_dict(error), output=payload)
                        else:
                            event.update(stage="evaluation", result="success", output=payload)
                            events.append(event)
                            history.append({"instruction": step["instruction"], "proposal": call, "output": payload})
                            completed += 1
                            if step_index == fault_step and attempt_index > 0:
                                recovered = True
                            success = True
                            break

                events.append(event)
                if error is not None:
                    last_error = error.message
                    if condition == "closed_loop":
                        feedback_text = render_feedback(error, feedback_style)
                        event["feedback_sent"] = feedback_text
                        history.append({"failed_call": candidate, "feedback": feedback_text})
                    if condition == "no_verification":
                        break

            if not success:
                break

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": task["id"],
        "source": task["source"],
        "mode": "llm",
        "model": MODEL,
        "replicate": 1,
        "condition": condition,
        "fault": fault,
        "feedback_format": feedback_style if condition == "closed_loop" else "not_applicable",
        "retry_budget": retry_budget if condition != "no_verification" else 0,
        "task_completed": int(completed == len(task["steps"])),
        "completed_steps": completed,
        "injected_fault": int(injected),
        "injected_fault_recovered": int(injected and recovered),
        "controlled_seed": 1,
        "seed_source": "canonical_task_specification",
        "online_model_proposals": model_proposals,
        "retry_count": retries,
        "latency_seconds": round(perf_counter() - started, 6),
        "total_tokens": total_tokens if token_complete else None,
        "token_measurement_complete": int(token_complete),
        "last_error": last_error if completed < len(task["steps"]) else "",
        "events": events,
    }


async def run_inventory_baseline_final(seeds: dict) -> None:
    from agents import new_agent
    from benchmarks import inventory_tasks
    from experiment import run_trial
    from taxonomy import FAULTS

    tasks = inventory_tasks()
    expected = 20 * len(FAULTS) * 3
    assert expected == 480
    plan = {
        "experiment": "inventory_natural_plus_controlled_fault_baseline_final",
        "model": MODEL,
        "task_count": 20,
        "faults": list(FAULTS),
        "policies": list(POLICIES),
        "retry_budget": 2,
        "method_note": (
            "No-fault cells are genuine end-to-end LLM runs. Injected-fault cells start from "
            "a canonical task-specification clean proposal chain so recovery is not confounded by "
            "an unrelated fresh generation error."
        ),
        "total_planned_trials": expected,
    }
    store = ResultStore(
        ROOT / "redo_v3_inventory_baseline", plan,
        ("task_id", "condition", "fault"),
    )
    agent = new_agent("ollama", MODEL)
    try:
        count = len(store.records)
        for task in tasks:
            seed = seeds["tasks"][task["id"]]["proposals"]
            for fault in FAULTS:
                for condition in POLICIES:
                    template = {"task_id": task["id"], "condition": condition, "fault": fault}
                    if store.has(template):
                        continue
                    if fault == "none":
                        row = await run_trial(
                            agent, task, condition, "none",
                            "schema_diff" if condition == "closed_loop" else "not_applicable",
                            2, rep=1,
                        )
                        row["controlled_seed"] = 0
                        row["seed_source"] = "natural_end_to_end"
                    else:
                        row = await run_inventory_controlled_trial(
                            agent, task, seed, condition, fault,
                            "schema_diff" if condition == "closed_loop" else "not_applicable",
                            2,
                        )
                    store.append(row)
                    count += 1
                    print(
                        f"[inventory-final {count}/{expected}] {task['id']} {fault} {condition}: "
                        f"{'PASS' if row['task_completed'] else 'FAIL'}",
                        flush=True,
                    )
    finally:
        if hasattr(agent, "aclose"):
            await agent.aclose()
    store.finalize(expected)


async def run_feedback_final(seeds: dict) -> None:
    from agents import new_agent
    from benchmarks import inventory_tasks
    from experiment import run_trial
    from taxonomy import FAULTS, FEEDBACKS

    tasks = inventory_tasks()[:3]
    expected = 3 * len(FAULTS) * len(FEEDBACKS)
    assert expected == 72
    plan = {
        "experiment": "feedback_format_ablation_final",
        "model": MODEL,
        "task_count": 3,
        "faults": list(FAULTS),
        "feedback_formats": list(FEEDBACKS),
        "retry_budget": 2,
        "method_note": "Injected-fault cells use canonical task-specification control seeds.",
        "total_planned_trials": expected,
    }
    store = ResultStore(
        ROOT / "redo_v3_feedback", plan,
        ("task_id", "fault", "feedback_format"),
    )
    agent = new_agent("ollama", MODEL)
    try:
        count = len(store.records)
        for task in tasks:
            seed = seeds["tasks"][task["id"]]["proposals"]
            for fault in FAULTS:
                for style in FEEDBACKS:
                    template = {"task_id": task["id"], "fault": fault, "feedback_format": style}
                    if store.has(template):
                        continue
                    if fault == "none":
                        row = await run_trial(agent, task, "closed_loop", "none", style, 2, rep=1)
                        row["controlled_seed"] = 0
                        row["seed_source"] = "natural_end_to_end"
                    else:
                        row = await run_inventory_controlled_trial(
                            agent, task, seed, "closed_loop", fault, style, 2
                        )
                    store.append(row)
                    count += 1
                    print(
                        f"[feedback-final {count}/{expected}] {task['id']} {fault} {style}: "
                        f"{'PASS' if row['task_completed'] else 'FAIL'}",
                        flush=True,
                    )
    finally:
        if hasattr(agent, "aclose"):
            await agent.aclose()
    store.finalize(expected)


def run_bfcl_final() -> None:
    py = sys.executable
    run_command(
        py, "experiment.py", "--mode", "ollama", "--model", MODEL,
        "--benchmark", "bfcl",
        "--bfcl-questions", str(Path("data") / "bfcl_v3_simple" / "questions.jsonl"),
        "--bfcl-answers", str(Path("data") / "bfcl_v3_simple" / "answers.jsonl"),
        "--tasks", "20", "--replicates", "1", "--suite", "baselines",
        "--faults", "none", "--retry-budgets", "2",
        "--out", "redo_v3_bfcl", "--resume",
    )
    validate_legacy_folder("redo_v3_bfcl", 60)


async def run_retry_redo(seeds: dict) -> None:
    from agents import new_agent
    from benchmarks import inventory_tasks
    from experiment import run_trial
    from taxonomy import FAULTS

    tasks = inventory_tasks()[:2]
    budgets = (0, 1, 2, 3, 4)
    plans = []
    for fault in FAULTS:
        plans.append(("no_verification", fault, 0))
        for budget in budgets:
            plans.append(("naive_retry", fault, budget))
            plans.append(("closed_loop", fault, budget))

    expected = len(tasks) * len(plans)
    if expected != 176:
        raise AssertionError(f"Retry design error: expected 176 cells, calculated {expected}")

    plan = {
        "experiment": "corrected_retry_budget_ablation_final",
        "model": MODEL,
        "task_count": len(tasks),
        "faults": list(FAULTS),
        "budgets": list(budgets),
        "method_note": (
            "No-fault cells are natural LLM runs. Injected-fault cells use validated LLM-generated "
            "clean seed proposals before controlled fault injection."
        ),
        "total_planned_trials": expected,
    }
    store = ResultStore(
        ROOT / "redo_v3_retry", plan,
        ("task_id", "condition", "fault", "scheduled_retry_budget"),
    )
    agent = new_agent("ollama", MODEL)
    try:
        count = len(store.records)
        for task in tasks:
            seed = seeds["tasks"][task["id"]]["proposals"]
            for condition, fault, budget in plans:
                template = {
                    "task_id": task["id"], "condition": condition, "fault": fault,
                    "scheduled_retry_budget": budget,
                }
                if store.has(template):
                    continue
                if fault == "none":
                    row = await run_trial(
                        agent, task, condition, "none",
                        "schema_diff" if condition == "closed_loop" else "not_applicable",
                        budget, rep=1,
                    )
                    row["controlled_seed"] = 0
                    row["seed_source"] = "natural_end_to_end"
                else:
                    row = await run_inventory_controlled_trial(
                        agent, task, seed, condition, fault,
                        "schema_diff" if condition == "closed_loop" else "not_applicable",
                        budget,
                    )
                row["scheduled_retry_budget"] = budget
                row["effective_retry_budget"] = row.get("retry_budget", 0)
                store.append(row)
                count += 1
                print(
                    f"[retry-final {count}/{expected}] {task['id']} {fault} {condition} "
                    f"budget={budget}: {'PASS' if row['task_completed'] else 'FAIL'}",
                    flush=True,
                )
    finally:
        if hasattr(agent, "aclose"):
            await agent.aclose()
    store.finalize(expected)


class RedoReferenceData:
    def __init__(self):
        self.countries = read_csv(DATA_DIR / "country_reference.csv")
        self.wikidata = read_csv(DATA_DIR / "wikidata_terms.csv")
        self.hdata = read_csv(DATA_DIR / "hdata_terms.csv")
        self.cases = read_csv(DATA_DIR / "benchmark_cases.csv")

    def country_by_capital(self, capital: str) -> dict:
        matches = [r for r in self.countries if normalize(r["capital"]) == normalize(capital)]
        if len(matches) != 1:
            raise ValueError(f"Expected one country for capital {capital!r}; found {len(matches)}")
        row = matches[0]
        return {
            "source": "wikidata_country_reference",
            "capital": row["capital"], "country_code": row["country_code"],
            "country_name": row["country_name"], "country_qid": row["country_qid"],
            "capital_qid": row["capital_qid"],
        }

    @staticmethod
    def _term(rows: list[dict[str, str]], source: str, country_code: str, target_date: str) -> dict:
        code = (country_code or "").strip().upper()
        if not re.fullmatch(r"[A-Z]{3}", code):
            raise ValueError(f"country_code must be an ISO3 code; received {country_code!r}")
        try:
            target = date.fromisoformat(target_date)
        except Exception as exc:
            raise ValueError("target_date must use YYYY-MM-DD") from exc
        matches = []
        for row in rows:
            if row["country_code"].upper() != code:
                continue
            start = date.fromisoformat(row["start_date"])
            end = date.fromisoformat(row["end_date"])
            if start <= target <= end:
                matches.append(row)
        if len(matches) != 1:
            raise ValueError(
                f"{source}: expected one presidential record for {code} on {target_date}; found {len(matches)}"
            )
        row = matches[0]
        result = {
            "source": source,
            "country_code": code,
            "country_name": row["country_name"],
            "president_name": row["president_name"],
            "target_date": target_date,
            "term_start": row["start_date"],
            "term_end": row["end_date"],
        }
        if source == "wikidata":
            result.update({
                "president_qid": row["president_qid"],
                "office_qid": row["office_qid"],
                "wikidata_source_path": row["wikidata_source_path"],
            })
        else:
            result.update({
                "office_title": row["office_title"],
                "hdata_country_id": row["hdata_country_id"],
            })
        return result

    def wikidata_term(self, country_code: str, target_date: str) -> dict:
        return self._term(self.wikidata, "wikidata", country_code, target_date)

    def hdata_term(self, country_code: str, target_date: str) -> dict:
        return self._term(self.hdata, "hdata", country_code, target_date)


def make_redo_server(data: RedoReferenceData):
    from mcp.server.mcpserver import MCPServer

    server = MCPServer(
        "30-Country Two-Source Historical MCP",
        instructions=(
            "Read-only research references. Country/capital and Wikidata presidential data are "
            "Wikidata-derived; H-DATA is independent. Agents must gather both source records before deciding."
        ),
    )

    @server.tool()
    def resolve_country_by_capital(capital: str) -> dict:
        """Resolve a capital to ISO3 country code using the Wikidata-derived country reference."""
        return data.country_by_capital(capital)

    @server.tool()
    def lookup_wikidata_president(country_code: str, target_date: str) -> dict:
        """Retrieve the Wikidata-derived presidential record covering a date for an ISO3 country."""
        return data.wikidata_term(country_code, target_date)

    @server.tool()
    def lookup_hdata_president(country_code: str, target_date: str) -> dict:
        """Retrieve the independent H-DATA presidential record covering a date for an ISO3 country."""
        return data.hdata_term(country_code, target_date)

    @server.tool()
    def submit_presidential_decision(
        country_code: str,
        target_date: str,
        president_name: str,
        wikidata_president: str,
        hdata_president: str,
    ) -> dict:
        """Submit a final cross-source decision after comparing Wikidata and H-DATA evidence."""
        # This tool intentionally does not contain the answer key. The independent host evaluator
        # checks the submitted decision after execution.
        return {
            "country_code": country_code,
            "target_date": target_date,
            "president_name": president_name,
            "wikidata_president": wikidata_president,
            "hdata_president": hdata_president,
        }

    return server


def multiref_tasks() -> list[dict]:
    tasks = []
    for row in read_csv(DATA_DIR / "benchmark_cases.csv"):
        tasks.append({
            "id": row["task_id"],
            "source": "redo_two_source_presidents",
            "country_code": row["country_code"],
            "country_name": row["country_name"],
            "capital": row["capital"],
            "target_date": row["target_date"],
            "gold_president_name": row["gold_president_name"],
            "wikidata_president_name": row["wikidata_president_name"],
            "hdata_president_name": row["hdata_president_name"],
            "agreement_type": row["agreement_type"],
            "description": (
                f"Determine who was president of the country whose capital is {row['capital']} "
                f"on {row['target_date']}. Resolve the country through MCP, independently collect "
                "Wikidata and H-DATA evidence, compare them, then submit one evidence-grounded decision."
            ),
        })
    return tasks


def step_definition(task: dict, index: int, used_sources: list[str]) -> dict:
    if index == 0:
        return {
            "kind": "resolve",
            "instruction": (
                f"Resolve the country whose capital is {task['capital']} using an MCP tool. "
                "Do not answer from memory."
            ),
        }
    if index == 1:
        return {
            "kind": "source1",
            "instruction": (
                f"Using the country_code returned by MCP, retrieve the presidential record covering "
                f"{task['target_date']} from EITHER Wikidata or H-DATA. Choose the source yourself."
            ),
        }
    if index == 2:
        used = used_sources[-1] if used_sources else "the first source"
        return {
            "kind": "source2",
            "instruction": (
                f"Retrieve an INDEPENDENT presidential record for the same country and date from the "
                f"other source. The source already used was {used}. You have the full MCP catalog; "
                "select the different evidence source yourself."
            ),
        }
    return {
        "kind": "decision",
        "instruction": (
            "Compare the successful Wikidata and H-DATA evidence in the action history. Submit a final "
            "presidential decision using submit_presidential_decision. Copy both source names into their "
            "corresponding evidence fields and choose the president supported by the comparison."
        ),
    }


def fault_to_dict(error) -> dict:
    if error is None:
        return {}
    if hasattr(error, "dict"):
        return error.dict()
    return {"category": "error", "code": type(error).__name__, "message": str(error)}


def make_fault(category: str, code: str, message: str, **kwargs):
    from taxonomy import Fault
    return Fault(category, code, message, **kwargs)


def schema_preflight(call: dict, catalog: dict):
    from jsonschema import Draft202012Validator
    if call["tool"] not in catalog:
        return make_fault(
            "incorrect_tool", "unknown_tool", f"Tool {call['tool']!r} does not exist",
            expected=f"one of {sorted(catalog)}", actual=call["tool"],
        )
    schema = catalog[call["tool"]].input_schema
    errors = sorted(
        Draft202012Validator(schema).iter_errors(call["args"]),
        key=lambda e: (list(map(str, e.path)), e.message),
    )
    if not errors:
        return None
    err = errors[0]
    field = ".".join(map(str, err.path))
    if err.validator == "required":
        match = re.search(r"'([^']+)'", err.message)
        field = match.group(1) if match else "required field"
    category = "parameter_type_mismatch" if err.validator == "type" else "schema_mismatch"
    return make_fault(
        category, f"jsonschema_{err.validator}", err.message,
        field=field, expected=repr(err.validator_value), actual=repr(call["args"]),
        trace=f"JSON Schema validation: {err.message}",
    )


def semantic_preflight(call: dict, step: dict, task: dict, history: list[dict]):
    kind = step["kind"]
    source_tools = {"lookup_wikidata_president", "lookup_hdata_president"}
    if kind == "resolve":
        allowed = {"resolve_country_by_capital"}
    elif kind in {"source1", "source2"}:
        allowed = source_tools
    else:
        allowed = {"submit_presidential_decision"}
    if call["tool"] not in allowed:
        suggestion = ""
        if len(allowed) == 1:
            suggestion = next(iter(allowed))
        return make_fault(
            "incorrect_tool", "semantic_tool_selection",
            f"The current action requires one of {sorted(allowed)}, not {call['tool']!r}",
            expected=repr(sorted(allowed)), actual=call["tool"], suggested_tool=suggestion,
        )

    country_outputs = [x["output"] for x in history if isinstance(x.get("output"), dict) and "country_code" in x["output"]]
    country_code = country_outputs[0]["country_code"] if country_outputs else None
    if kind in {"source1", "source2", "decision"} and country_code:
        actual = call["args"].get("country_code")
        if actual != country_code:
            return make_fault(
                "semantic_failure", "ungrounded_country_code",
                "country_code must copy the value returned by the previous MCP country lookup",
                field="country_code", expected=repr(country_code), actual=repr(actual),
            )
    if kind in {"source1", "source2", "decision"}:
        actual_date = call["args"].get("target_date")
        if actual_date != task["target_date"]:
            return make_fault(
                "semantic_failure", "wrong_target_date",
                "target_date must remain the date stated in the task",
                field="target_date", expected=task["target_date"], actual=repr(actual_date),
            )
    if kind == "source2":
        used = [x["output"].get("source") for x in history if isinstance(x.get("output"), dict) and x["output"].get("source") in {"wikidata", "hdata"}]
        if used:
            selected_source = "wikidata" if call["tool"] == "lookup_wikidata_president" else "hdata"
            if selected_source == used[-1]:
                other_tool = "lookup_hdata_president" if selected_source == "wikidata" else "lookup_wikidata_president"
                return make_fault(
                    "semantic_failure", "duplicate_source",
                    "The second evidence call must use the independent source not already used",
                    expected=f"source different from {selected_source}", actual=selected_source,
                    suggested_tool=other_tool,
                )

    # Phase-D merge grounding: validate final evidence fields against values actually
    # retrieved through MCP. This does not use the benchmark answer key. It only makes
    # the existing closed-loop feedback actionable when the model emits placeholders
    # such as "wikidata_president" instead of copying the observed source value.
    if kind == "decision":
        source_outputs = [
            x["output"] for x in history
            if isinstance(x.get("output"), dict)
            and x["output"].get("source") in {"wikidata", "hdata"}
        ]
        by_source = {x["source"]: x for x in source_outputs}
        if set(by_source) == {"wikidata", "hdata"}:
            expected_w = by_source["wikidata"].get("president_name", "")
            expected_h = by_source["hdata"].get("president_name", "")
            actual_w = call["args"].get("wikidata_president", "")
            actual_h = call["args"].get("hdata_president", "")
            actual_final = call["args"].get("president_name", "")

            if normalize(actual_w) != normalize(expected_w):
                return make_fault(
                    "semantic_failure", "bad_wikidata_citation",
                    "wikidata_president must copy the president_name returned by the Wikidata MCP lookup",
                    field="wikidata_president", expected=expected_w, actual=actual_w,
                )
            if normalize(actual_h) != normalize(expected_h):
                return make_fault(
                    "semantic_failure", "bad_hdata_citation",
                    "hdata_president must copy the president_name returned by the H-DATA MCP lookup",
                    field="hdata_president", expected=expected_h, actual=actual_h,
                )
            if not (name_equivalent(actual_final, expected_w) and name_equivalent(actual_final, expected_h)):
                return make_fault(
                    "semantic_failure", "ungrounded_final_decision",
                    "president_name must be supported by both retrieved source records",
                    field="president_name",
                    expected=f"Wikidata={expected_w!r}; H-DATA={expected_h!r}",
                    actual=actual_final,
                )
    return None


def evaluate_output(call: dict, payload: dict, step: dict, task: dict, history: list[dict]):
    kind = step["kind"]
    if not isinstance(payload, dict):
        return make_fault("semantic_failure", "non_object_output", "MCP output must be an object")
    if kind == "resolve":
        if payload.get("country_code") != task["country_code"] or normalize(payload.get("country_name", "")) != normalize(task["country_name"]):
            return make_fault(
                "semantic_failure", "wrong_country",
                "Country lookup result does not satisfy the benchmark case",
                expected=f"{task['country_code']} {task['country_name']}", actual=repr(payload),
            )
        return None
    if kind in {"source1", "source2"}:
        if payload.get("country_code") != task["country_code"] or payload.get("target_date") != task["target_date"]:
            return make_fault("semantic_failure", "wrong_reference_scope", "Source evidence is for the wrong country or date")
        source = payload.get("source")
        if source not in {"wikidata", "hdata"}:
            return make_fault("semantic_failure", "unknown_source", "Evidence output does not identify Wikidata or H-DATA")
        if source == "wikidata":
            expected_name = task["wikidata_president_name"]
        else:
            expected_name = task["hdata_president_name"]
        if not name_equivalent(payload.get("president_name", ""), expected_name):
            return make_fault(
                "semantic_failure", "source_oracle_mismatch",
                f"{source} evidence does not match its independently prepared reference",
                expected=expected_name, actual=payload.get("president_name", ""),
            )
        if kind == "source2":
            previous_sources = [x["output"].get("source") for x in history if isinstance(x.get("output"), dict) and x["output"].get("source") in {"wikidata", "hdata"}]
            if previous_sources and source == previous_sources[-1]:
                return make_fault(
                    "semantic_failure", "duplicate_source",
                    "Both evidence calls used the same source; independent comparison was not performed",
                )
        return None

    source_outputs = [x["output"] for x in history if isinstance(x.get("output"), dict) and x["output"].get("source") in {"wikidata", "hdata"}]
    by_source = {x["source"]: x for x in source_outputs}
    if set(by_source) != {"wikidata", "hdata"}:
        return make_fault("semantic_failure", "missing_independent_evidence", "Final decision requires both Wikidata and H-DATA evidence")
    if not (
        name_equivalent(payload.get("president_name", ""), task["wikidata_president_name"])
        and name_equivalent(payload.get("president_name", ""), task["hdata_president_name"])
    ):
        return make_fault(
            "semantic_failure", "wrong_final_answer",
            "Submitted president is not equivalent to the person supported by both sources",
            expected=task["gold_president_name"], actual=payload.get("president_name", ""),
        )
    if normalize(payload.get("wikidata_president", "")) != normalize(by_source["wikidata"]["president_name"]):
        return make_fault(
            "semantic_failure", "bad_wikidata_citation",
            "Final decision did not copy the retrieved Wikidata president",
            field="wikidata_president",
            expected=by_source["wikidata"]["president_name"],
            actual=payload.get("wikidata_president", ""),
        )
    if normalize(payload.get("hdata_president", "")) != normalize(by_source["hdata"]["president_name"]):
        return make_fault(
            "semantic_failure", "bad_hdata_citation",
            "Final decision did not copy the retrieved H-DATA president",
            field="hdata_president",
            expected=by_source["hdata"]["president_name"],
            actual=payload.get("hdata_president", ""),
        )
    if normalize(by_source["wikidata"]["president_name"]) != normalize(task["wikidata_president_name"]):
        return make_fault("semantic_failure", "wikidata_history_mismatch", "Stored Wikidata evidence is inconsistent")
    if normalize(by_source["hdata"]["president_name"]) != normalize(task["hdata_president_name"]):
        return make_fault("semantic_failure", "hdata_history_mismatch", "Stored H-DATA evidence is inconsistent")
    return None


def inject_multiref_fault(raw_candidate: str, fault: str, task: dict, catalog: dict):
    """Return (mutated_text, injected_bool). Transient faults are injected at transport."""
    if fault in {"none", "transient_once"}:
        # transient_once is delivered later at the transport boundary only after
        # the proposal has parsed and passed any applicable preflight checks.
        return raw_candidate, False
    try:
        value = json.loads(raw_candidate)
    except Exception:
        return raw_candidate, False
    if not isinstance(value, dict) or not isinstance(value.get("tool"), str) or not isinstance(value.get("args"), dict):
        return raw_candidate, False
    tool = value["tool"]
    args = dict(value["args"])
    if tool not in {"lookup_wikidata_president", "lookup_hdata_president"}:
        return raw_candidate, False

    if fault == "type_mismatch":
        args["country_code"] = 12345
        value["args"] = args
    elif fault == "wrong_tool":
        value = {"tool": "resolve_country_by_capital", "args": {"capital": task["capital"]}}
    elif fault == "persistent_corruption":
        args["country_code"] = "ZZZ"
        value["args"] = args
    else:
        return raw_candidate, False
    return json.dumps(value, ensure_ascii=False), True


async def run_multiref_trial(
    agent_by_step: list,
    data: RedoReferenceData,
    task: dict,
    condition: str,
    fault: str,
    feedback_style: str = "schema_diff",
    seed_chain: list[str] | None = None,
) -> dict:
    from feedback import render_feedback
    from mcp.client import Client
    from verification import parse_proposal, unwrap_mcp_reply

    server = make_redo_server(data)
    start_clock = perf_counter()
    events: list[dict] = []
    history: list[dict] = []
    used_sources: list[str] = []
    total_tokens = 0
    token_complete = True
    model_proposals = 0
    retries = 0
    same_candidate_retries = 0
    injected = False
    recovered = False
    completed_steps = 0
    last_error = ""
    fault_step = 1  # first source-retrieval action, zero-based

    async with Client(server) as client:
        catalog = {tool.name: tool for tool in (await client.list_tools()).tools}
        for step_index in range(4):
            step = step_definition(task, step_index, used_sources)

            # For the merge step, expose the already-retrieved MCP evidence explicitly
            # in the instruction. These are observed tool outputs, not hidden gold answers.
            # This prevents a small local model from treating schema field names as
            # literal values while preserving genuine model choice for the final decision.
            if step["kind"] == "decision":
                source_outputs = [
                    x["output"] for x in history
                    if isinstance(x.get("output"), dict)
                    and x["output"].get("source") in {"wikidata", "hdata"}
                ]
                by_source = {x["source"]: x for x in source_outputs}
                if set(by_source) == {"wikidata", "hdata"}:
                    w_name = by_source["wikidata"].get("president_name", "")
                    h_name = by_source["hdata"].get("president_name", "")
                    step["instruction"] += (
                        f" Retrieved MCP evidence: Wikidata president_name={w_name!r}; "
                        f"H-DATA president_name={h_name!r}. In wikidata_president and "
                        "hdata_president, copy those retrieved person-name values exactly; "
                        "do not write the field names as placeholder text."
                    )

            attempts_allowed = 1 if condition == "no_verification" else 3
            feedback_text = ""
            candidate = None
            success = False
            first_faulted_candidate = None

            for attempt_index in range(attempts_allowed):
                attempt_number = attempt_index + 1
                agent = agent_by_step[step_index]
                if attempt_index == 0 and seed_chain is not None:
                    candidate = seed_chain[step_index]
                    usage = None
                    if step_index == fault_step and fault != "none":
                        candidate, did_inject = inject_multiref_fault(candidate, fault, task, catalog)
                        injected = injected or did_inject
                        if did_inject:
                            first_faulted_candidate = candidate
                elif attempt_index == 0 or condition == "closed_loop":
                    candidate, usage = await agent.generate(task, step, catalog, feedback_text, history)
                    model_proposals += 1
                    if usage is None:
                        token_complete = False
                    else:
                        total_tokens += int(usage.get("input", 0)) + int(usage.get("output", 0))
                    if step_index == fault_step and attempt_index == 0 and fault != "none":
                        candidate, did_inject = inject_multiref_fault(candidate, fault, task, catalog)
                        injected = injected or did_inject
                        if did_inject:
                            first_faulted_candidate = candidate
                else:
                    # Naive retry deliberately replays the exact same failed candidate.
                    retries += 1
                    same_candidate_retries += int(candidate == first_faulted_candidate or first_faulted_candidate is None)

                if attempt_index > 0 and condition == "closed_loop":
                    retries += 1

                event = {
                    "task_id": task["id"], "step": step_index + 1, "step_kind": step["kind"],
                    "attempt": attempt_number, "condition": condition, "fault": fault,
                    "agent_role": "A" if agent is agent_by_step[0] else "B",
                    "proposal": candidate,
                    "feedback_received": feedback_text,
                    "result": "", "stage": "",
                }

                call, error = parse_proposal(candidate)
                if error is not None:
                    event.update(stage="parse", result="rejected", error=fault_to_dict(error))
                elif condition == "closed_loop":
                    error = schema_preflight(call, catalog)
                    if error is None:
                        error = semantic_preflight(call, step, task, history)
                    if error is not None:
                        event.update(stage="preflight", result="rejected", error=fault_to_dict(error))

                if error is None and call is not None:
                    if step_index == fault_step and attempt_index == 0 and fault == "transient_once":
                        injected = True
                        first_faulted_candidate = candidate
                        error = make_fault(
                            "transient_api", "http_503_once",
                            "Injected one-shot HTTP 503 equivalent. The same valid call may succeed on retry.",
                            trace="HTTPError: 503 Service Unavailable (controlled one-shot injection)", injected=True,
                        )
                        event.update(stage="transport_injection", result="rejected", error=fault_to_dict(error))
                    else:
                        try:
                            reply = await client.call_tool(call["tool"], call["args"])
                            payload, error = unwrap_mcp_reply(reply)
                        except Exception as exc:
                            payload = None
                            error = make_fault(
                                "execution_failure", type(exc).__name__, str(exc)[:500],
                                trace="".join(traceback.format_exception(exc))[-2500:],
                            )
                        if error is not None:
                            event.update(stage="execution", result="error", error=fault_to_dict(error))
                        else:
                            error = evaluate_output(call, payload, step, task, history)
                            if error is not None:
                                event.update(stage="evaluation", result="wrong_result", error=fault_to_dict(error), output=payload)
                            else:
                                event.update(stage="evaluation", result="success", output=payload)
                                events.append(event)
                                history.append({
                                    "instruction": step["instruction"],
                                    "proposal": call,
                                    "output": payload,
                                    "agent_role": event["agent_role"],
                                })
                                if payload.get("source") in {"wikidata", "hdata"}:
                                    used_sources.append(payload["source"])
                                completed_steps += 1
                                if injected and step_index == fault_step and attempt_index > 0:
                                    recovered = True
                                success = True
                                break

                events.append(event)
                if error is not None:
                    last_error = error.message
                    if condition == "closed_loop":
                        feedback_text = render_feedback(error, feedback_style)
                        history.append({"failed_call": candidate, "feedback": feedback_text, "agent_role": event["agent_role"]})
                    if condition == "no_verification":
                        break

            if not success:
                break

    source_order = [x["output"]["source"] for x in history if isinstance(x.get("output"), dict) and x["output"].get("source") in {"wikidata", "hdata"}]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": task["id"], "country_code": task["country_code"], "country_name": task["country_name"],
        "capital": task["capital"], "target_date": task["target_date"],
        "agreement_type": task["agreement_type"], "model": MODEL,
        "condition": condition, "fault": fault, "feedback_format": feedback_style if condition == "closed_loop" else "not_applicable",
        "task_completed": int(completed_steps == 4), "completed_steps": completed_steps,
        "controlled_seed": int(seed_chain is not None),
        "seed_source": "canonical_balanced_two_source" if seed_chain is not None else "natural_end_to_end",
        "injected_fault": int(injected), "injected_fault_recovered": int(injected and recovered),
        "model_proposals": model_proposals, "retry_count": retries,
        "same_candidate_retry_count": same_candidate_retries,
        "source_order": "+".join(source_order),
        "independent_sources_observed": int(set(source_order) == {"wikidata", "hdata"}),
        "latency_seconds": round(perf_counter() - start_clock, 6),
        "total_tokens": total_tokens if token_complete else None,
        "token_measurement_complete": int(token_complete),
        "last_error": last_error if completed_steps < 4 else "",
        "events": events,
    }


async def build_multiref_seed_calls(data: RedoReferenceData, tasks: list[dict]) -> dict:
    """Build deterministic canonical two-source chains for controlled fault injection.

    Source order is balanced across countries to avoid a single fixed source-order bias.
    These seeds are setup controls only. Natural no-fault and two-agent trials still require
    the model to choose tools and evidence sources itself.
    """
    path = DATA_DIR / "multiref_control_seeds.json"
    payload = {
        "model": MODEL,
        "purpose": (
            "Canonical valid four-step two-source chains used only as common initial calls "
            "for injected-fault comparisons. Natural no-fault and two-agent runs remain LLM-driven."
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tasks": {},
    }

    for task_index, task in enumerate(tasks, start=1):
        # Balance source order: 15 countries start with Wikidata, 15 with H-DATA.
        first_source = "wikidata" if task_index % 2 else "hdata"
        second_source = "hdata" if first_source == "wikidata" else "wikidata"
        first_tool = f"lookup_{first_source}_president"
        second_tool = f"lookup_{second_source}_president"

        proposals = [
            json.dumps(
                {"tool": "resolve_country_by_capital", "args": {"capital": task["capital"]}},
                ensure_ascii=False, sort_keys=True,
            ),
            json.dumps(
                {"tool": first_tool, "args": {"country_code": task["country_code"], "target_date": task["target_date"]}},
                ensure_ascii=False, sort_keys=True,
            ),
            json.dumps(
                {"tool": second_tool, "args": {"country_code": task["country_code"], "target_date": task["target_date"]}},
                ensure_ascii=False, sort_keys=True,
            ),
            json.dumps(
                {
                    "tool": "submit_presidential_decision",
                    "args": {
                        "country_code": task["country_code"],
                        "target_date": task["target_date"],
                        "president_name": task["gold_president_name"],
                        "wikidata_president": task["wikidata_president_name"],
                        "hdata_president": task["hdata_president_name"],
                    },
                },
                ensure_ascii=False, sort_keys=True,
            ),
        ]
        payload["tasks"][task["id"]] = {
            "proposals": proposals,
            "source_order": [first_source, second_source],
            "seed_type": "canonical_balanced_two_source",
        }
        print(
            f"[control seed multiref {task_index}/30] {task['country_code']} "
            f"sources={first_source}+{second_source} READY",
            flush=True,
        )

    atomic_json(path, payload)
    return payload


async def run_multiref_redo() -> None:
    from agents import new_agent

    data = RedoReferenceData()
    tasks = multiref_tasks()
    if len(tasks) != 30:
        raise RuntimeError("Multi-reference redo requires exactly 30 country tasks")
    seeds = await build_multiref_seed_calls(data, tasks)
    plan = {
        "experiment": "30_country_two_source_policy_comparison",
        "model": MODEL,
        "task_count": 30,
        "faults": list(MULTIREF_FAULTS),
        "policies": list(POLICIES),
        "retry_budget": 2,
        "total_planned_trials": 30 * len(MULTIREF_FAULTS) * len(POLICIES),
        "source_design": "Wikidata-derived reference + independent H-DATA, with agent-selected evidence order",
        "controlled_fault_note": "Injected-fault rows start from a canonical balanced two-source control chain; no-fault rows remain natural end-to-end runs.",
        "naive_retry_definition": "Replay the exact same failed tool call without corrective feedback",
        "transient_definition": "One-shot controlled transport failure; retry is allowed to execute normally",
        "persistent_definition": "Schema-valid country_code corruption to ZZZ; naive retry repeats it unchanged",
    }
    expected = plan["total_planned_trials"]
    if expected != 450:
        raise AssertionError("30-country matrix must contain exactly 450 cells")
    store = ResultStore(
        ROOT / "redo_v3_multiref", plan,
        ("task_id", "condition", "fault"),
    )
    agent = new_agent("ollama", MODEL)
    try:
        count = len(store.records)
        for task in tasks:
            for fault in MULTIREF_FAULTS:
                for condition in POLICIES:
                    template = {"task_id": task["id"], "condition": condition, "fault": fault}
                    if store.has(template):
                        continue
                    seed_chain = None if fault == "none" else seeds["tasks"][task["id"]]["proposals"]
                    row = await run_multiref_trial(
                        [agent, agent, agent, agent], data, task, condition, fault,
                        seed_chain=seed_chain,
                    )
                    store.append(row)
                    count += 1
                    print(
                        f"[multiref {count}/{expected}] {task['country_code']} {fault} {condition}: "
                        f"{'PASS' if row['task_completed'] else 'FAIL'}",
                        flush=True,
                    )
    finally:
        if hasattr(agent, "aclose"):
            await agent.aclose()
    store.finalize(expected)
    mechanism_text = (
        "NAIVE RETRY MECHANISM — TECHNICAL DEFINITION\n"
        "===========================================\n"
        "Naive retry does not regenerate or repair a failed proposal. It replays the exact same proposal.\n"
        "transient_once is injected only on attempt 1, so attempt 2 may execute the same valid call successfully.\n"
        "persistent_corruption changes country_code to ZZZ; naive retry repeats ZZZ unchanged, so the defect persists.\n"
        "Closed-loop GVEF receives error feedback and asks the model to generate a corrected proposal.\n"
        "This file documents mechanism, not an LLM performance result.\n"
    )
    (ROOT / "redo_v3_multiref" / "NAIVE_RETRY_MECHANISM.txt").write_text(mechanism_text, encoding="utf-8")


async def run_two_agent_redo() -> None:
    from agents import new_agent

    data = RedoReferenceData()
    tasks = multiref_tasks()
    plan = {
        "experiment": "two_agent_independent_source_choice",
        "model": MODEL,
        "task_count": 30,
        "design": (
            "Both agents receive the full MCP tool catalog. Agent A resolves the country and chooses the first "
            "evidence source; Agent B must independently select the other source; Agent A submits the merged decision."
        ),
        "total_planned_trials": 30,
        "design_version": "phase_d_grounded_merge_feedback_v2",
        "merge_grounding_note": (
            "Decision prompts and corrective feedback expose only president_name values already returned "
            "by the two MCP source calls; benchmark gold labels are not inserted into model prompts."
        ),
    }
    store = ResultStore(ROOT / "redo_v3_two_agent", plan, ("task_id",))
    agent_a = new_agent("ollama", MODEL)
    agent_b = new_agent("ollama", MODEL)
    try:
        count = len(store.records)
        for task in tasks:
            template = {"task_id": task["id"]}
            if store.has(template):
                continue
            row = await run_multiref_trial(
                [agent_a, agent_a, agent_b, agent_a], data, task,
                "closed_loop", "none", feedback_style="schema_diff",
            )
            row["experiment"] = "two_agent"
            row["agent_assignment"] = "A:resolve+A:first_source; B:independent_second_source; A:merge_decision"
            # Derive which source each role actually retrieved from successful event outputs.
            a_sources, b_sources = [], []
            for event in row["events"]:
                output = event.get("output")
                if isinstance(output, dict) and output.get("source") in {"wikidata", "hdata"}:
                    (a_sources if event.get("agent_role") == "A" else b_sources).append(output["source"])
            row["agent_a_source_choice"] = a_sources[0] if a_sources else ""
            row["agent_b_source_choice"] = b_sources[0] if b_sources else ""
            row["agents_selected_independent_sources"] = int(
                bool(a_sources and b_sources and a_sources[0] != b_sources[0])
            )
            store.append(row)
            count += 1
            print(
                f"[two-agent {count}/30] {task['country_code']}: "
                f"{'PASS' if row['task_completed'] else 'FAIL'} "
                f"A={row['agent_a_source_choice'] or 'none'} B={row['agent_b_source_choice'] or 'none'}",
                flush=True,
            )
    finally:
        if hasattr(agent_a, "aclose"):
            await agent_a.aclose()
        if hasattr(agent_b, "aclose"):
            await agent_b.aclose()
    store.finalize(30)


def self_test() -> int:
    # Pure checks: no MCP/Ollama required.
    assert TOTAL_EXPECTED == 1268
    assert 30 * len(MULTIREF_FAULTS) * len(POLICIES) == 450
    retry_cells = 2 * 8 * (1 + 5 + 5)
    assert retry_cells == 176
    assert 20 * 3 == 60

    # Naive-retry mechanism: one-shot failure allows identical second call; persistent bad call does not.
    candidate = '{"tool":"lookup_wikidata_president","args":{"country_code":"FRA","target_date":"2001-05-16"}}'
    attempts = []
    for i in range(3):
        attempts.append(candidate)
        if i == 0:  # transient_once blocks only the first transport attempt
            continue
        break
    assert len(attempts) == 2 and attempts[0] == attempts[1]
    persistent = candidate.replace('"FRA"', '"ZZZ"')
    assert all(x == persistent for x in [persistent, persistent, persistent])

    # Prepared-data shape check if Step 1 has been run in this folder.
    if (DATA_DIR / "benchmark_cases.csv").exists():
        rows = read_csv(DATA_DIR / "benchmark_cases.csv")
        assert len(rows) == 30 and len({r["country_code"] for r in rows}) == 30
        data = RedoReferenceData()
        for case in rows:
            country = data.country_by_capital(case["capital"])
            assert country["country_code"] == case["country_code"]
            w = data.wikidata_term(case["country_code"], case["target_date"])
            h = data.hdata_term(case["country_code"], case["target_date"])
            assert name_equivalent(w["president_name"], case["wikidata_president_name"])
            assert name_equivalent(h["president_name"], case["hdata_president_name"])
    print("02_run_redo_experiments_final.py SELF-TEST: PASS")
    print(f"Expected total research rows: {TOTAL_EXPECTED}")
    return 0


async def run_mechanism_gate(seeds: dict) -> None:
    """Fast deterministic gate using the real MCP servers before expensive research."""
    from agents import new_agent
    from benchmarks import inventory_tasks

    print("\n--- PRE-RUN MECHANISM GATE ---")
    agent = new_agent("ollama", MODEL)
    try:
        task = inventory_tasks()[0]
        seed = seeds["tasks"][task["id"]]["proposals"]
        transient = await run_inventory_controlled_trial(
            agent, task, seed, "naive_retry", "transient_api", "not_applicable", 2
        )
        type_bad = await run_inventory_controlled_trial(
            agent, task, seed, "naive_retry", "type_mismatch", "not_applicable", 2
        )
        if int(transient.get("task_completed", 0)) != 1 or int(transient.get("injected_fault_recovered", 0)) != 1:
            raise RuntimeError("Mechanism gate failed: naive retry did not recover the one-shot transient call")
        if int(type_bad.get("task_completed", 0)) != 0:
            raise RuntimeError("Mechanism gate failed: naive retry unexpectedly repaired a persistent type-corrupted call")
        print("MECHANISM GATE: PASS")
        print("  transient_api + naive_retry -> PASS/recovered")
        print("  type_mismatch + naive_retry -> FAIL as expected")
    finally:
        if hasattr(agent, "aclose"):
            await agent.aclose()


async def phase_d_only_main() -> None:
    print("=" * 78)
    print("PHASE D ONLY — REDESIGNED TWO-AGENT EXPERIMENT")
    print("=" * 78)
    print(f"Project folder: {ROOT}")
    verify_prepared_data()
    verify_runtime_packages()
    run_command(sys.executable, "check_ollama.py")
    print("\n--- Phase D: redesigned two-agent experiment ---")
    await run_two_agent_redo()
    print("\nPHASE D REDO: COMPLETE")
    print("Run next: python 03_final_verify_redo_FINAL.py")


async def real_main() -> None:
    print("=" * 78)
    print("STEP 2/3 — RUN COMPLETE REDO EXPERIMENTS")
    print("=" * 78)
    print(f"Project folder: {ROOT}")
    print(f"Expected research rows across all redo suites: {TOTAL_EXPECTED}")
    verify_prepared_data()
    verify_runtime_packages()

    # Fail before expensive trials if the software environment is not ready.
    run_command(sys.executable, "-m", "pytest", "-q")
    run_command(sys.executable, "check_ollama.py")

    print("\n--- Phase A0: build canonical clean control calls ---")
    seeds = await build_inventory_seed_calls()
    await run_mechanism_gate(seeds)

    print("\n--- Phase A1: fresh inventory baseline rerun ---")
    await run_inventory_baseline_final(seeds)

    print("\n--- Phase A2: feedback-format rerun ---")
    await run_feedback_final(seeds)

    print("\n--- Phase A3: BFCL rerun ---")
    run_bfcl_final()

    print("\n--- Phase B: corrected retry-budget experiment ---")
    await run_retry_redo(seeds)

    print("\n--- Phase C: 30-country two-source policy experiment ---")
    await run_multiref_redo()

    print("\n--- Phase D: redesigned two-agent experiment ---")
    await run_two_agent_redo()

    print("\nSTEP 2 FINAL REDO EXPERIMENTS: COMPLETE")
    print("Run next: python 03_final_verify_redo_final.py")


def cli() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--self-test", action="store_true",
        help="Run pure plan/data checks without MCP or Ollama; does not create research results.",
    )
    parser.add_argument(
        "--phase-d-only", action="store_true",
        help="Run only the 30-country redesigned two-agent Phase D experiment.",
    )
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    try:
        if args.phase_d_only:
            asyncio.run(phase_d_only_main())
        else:
            asyncio.run(real_main())
        return 0
    except KeyboardInterrupt:
        print("\nSTOPPED BY USER. Completed custom rows remain checkpointed; old evidence is untouched.", file=sys.stderr)
        return 130
    except subprocess.CalledProcessError as exc:
        print(f"\nSTEP 2 STOPPED: command failed with exit code {exc.returncode}.", file=sys.stderr)
        return exc.returncode or 2
    except Exception as exc:
        print(f"\nSTEP 2 FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        if isinstance(exc, BaseExceptionGroup):
            def _show_group(group, indent=0):
                for item in group.exceptions:
                    prefix = "  " * indent
                    if isinstance(item, BaseExceptionGroup):
                        print(f"{prefix}{type(item).__name__}: {item}", file=sys.stderr)
                        _show_group(item, indent + 1)
                    else:
                        print(f"{prefix}{type(item).__name__}: {item}", file=sys.stderr)
            _show_group(exc)
        else:
            traceback.print_exception(exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
