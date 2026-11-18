"""Targeted, auditable Phase C repair: rerun ONLY 30 countries x 3 no-fault policies.

Never rewrites existing redo_v3_* results. The 360 controlled-fault Phase C
trials are reused byte-for-byte as JSON objects. The corrected 90 no-fault
trials use exactly the grounded-merge implementation already used in Phase D.

Commands (in the activated project environment):
    python 04_repair_phase_c_nofault.py --check
    python 04_repair_phase_c_nofault.py
    python 04_repair_phase_c_nofault.py --verify-only

Safe to Ctrl+C and rerun: each newly completed country/policy trial is saved.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import importlib.util
import inspect
import json
import logging
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "02_run_redo_experiments_FINAL.py"
DATA = ROOT / "redo_data"
OLD = ROOT / "redo_v3_multiref"
NEW = ROOT / "redo_v4_phase_c_nofault"
COMBINED = ROOT / "redo_v4_multiref_corrected"
POLICIES = ("no_verification", "naive_retry", "closed_loop")
EXPECTED = 90


class _MissingReferenceTracebackFilter(logging.Filter):
    """Hide only duplicate MCP server stack traces for unknown model-generated codes.

    The trial's MCP error, FAIL outcome and full event history remain unchanged in
    redo_v4_phase_c_nofault/records.json. Other exceptions are NOT suppressed.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "mcp.server.mcpserver.server" or not record.exc_info:
            return True
        if "raised an unexpected exception" not in record.getMessage():
            return True
        exc = record.exc_info[1]
        visited = set()
        while exc is not None and id(exc) not in visited:
            visited.add(id(exc))
            if (isinstance(exc, ValueError)
                    and "expected one presidential record for" in str(exc)
                    and "found 0" in str(exc)):
                return False
            exc = exc.__cause__ if exc.__cause__ is not None else exc.__context__
        return True


def diagnose_saved_country(country_code: str = "IRL") -> int:
    """Inspect saved model failures without Ollama, MCP, or changing results."""
    path = NEW / "records.json"
    if not path.is_file():
        print("No Phase C repair checkpoint exists yet; nothing to diagnose.")
        return 1
    records = load_json(path)
    rows = [r for r in records if r.get("country_code") == country_code]
    print(f"SAVED DIAGNOSTIC: {country_code} - {len(rows)} policy trial(s); "
          f"total completed checkpoint rows: {len(records)}")
    for row in rows:
        print(f"\n[{row.get('condition')}] "
              f"{'PASS' if row.get('task_completed') else 'FAIL'} "
              f"completed_steps={row.get('completed_steps')} "
              f"last_error={row.get('last_error')!r}")
        failures = [e for e in row.get("events", []) if e.get("result") != "success"]
        for event in failures[-4:]:
            raw = event.get("proposal", "")
            try:
                call = json.loads(raw)
                call_summary = f"{call.get('tool', '<none>')} {call.get('args', {})}"
            except (TypeError, ValueError, AttributeError):
                call_summary = str(raw)[:160]
            err = event.get("error") or {}
            print(f"  step={event.get('step')} attempt={event.get('attempt')} "
                  f"stage={event.get('stage')} "
                  f"code={err.get('code', '-')}: {call_summary[:180]}")
    print("\nThese are saved observations. Diagnostic mode makes zero model calls and edits zero files.")
    return 0


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_runner():
    if not RUNNER.is_file():
        raise RuntimeError("Missing current 02_run_redo_experiments_FINAL.py in this directory")
    spec = importlib.util.spec_from_file_location("mcp_phase_c_existing_runner", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Refuse an older runner: use only the corrected Phase D merge implementation.
    engine = inspect.getsource(mod.run_multiref_trial)
    grounding = inspect.getsource(mod.semantic_preflight)
    if ("Retrieved MCP evidence:" not in engine
            or "wikidata_president must copy" not in grounding
            or "hdata_president must copy" not in grounding):
        raise RuntimeError(
            "Runner is missing the Phase D grounded-merge repair. "
            "Use the SAME corrected 02_run_redo_experiments_FINAL.py that produced Phase D."
        )
    return mod


def preflight(mod):
    if not (DATA / "manifest.json").is_file():
        raise RuntimeError("Missing redo_data/manifest.json")
    manifest = load_json(DATA / "manifest.json")
    if manifest.get("country_count") != 30:
        raise RuntimeError("Prepared dataset is not the original 30-country set")
    for fname, evidence in manifest.get("outputs", {}).items():
        path = DATA / fname
        if not path.is_file() or digest(path) != evidence["sha256"]:
            raise RuntimeError(f"Prepared data hash changed: redo_data/{fname}")
    old_path = OLD / "records.json"
    old_manifest_path = OLD / "manifest.json"
    if not old_path.is_file() or not old_manifest_path.is_file():
        raise RuntimeError("Original redo_v3_multiref is missing. DO NOT create substitute rows.")
    old_manifest = load_json(old_manifest_path)
    original = load_json(old_path)
    if old_manifest.get("completion_status") != "complete" or len(original) != 450:
        raise RuntimeError("Original Phase C must be a completed 450-row experiment")
    tasks = mod.multiref_tasks()
    if len(tasks) != 30 or len({t['id'] for t in tasks}) != 30:
        raise RuntimeError("Expected exactly 30 distinct prepared benchmark cases")
    expected_keys = {(t['id'], p, f) for t in tasks for p in POLICIES for f in mod.MULTIREF_FAULTS}
    existing_keys = [(r['task_id'], r['condition'], r['fault']) for r in original]
    if len(set(existing_keys)) != 450 or set(existing_keys) != expected_keys:
        raise RuntimeError("Original Phase C trial-cell grid does not match the plan")
    keep = [r for r in original if r['fault'] != 'none']
    original_nofault = [r for r in original if r['fault'] == 'none']
    if len(keep) != 360 or len(original_nofault) != 90:
        raise RuntimeError("Unexpected Phase C no-fault vs fault-injection trial split")
    if any(r.get('controlled_seed') != 1 or r.get('seed_source') != 'canonical_balanced_two_source'
           for r in keep):
        raise RuntimeError("Cannot safely reuse controlled-fault rows: unexpected provenance")
    if any(r.get('controlled_seed') != 0 or r.get('seed_source') != 'natural_end_to_end'
           for r in original_nofault):
        raise RuntimeError("Unexpected original natural-run provenance")
    print("PREFLIGHT: PASS - 30 countries, 450 original trials, 360 controlled-fault rows retained")
    return tasks, original, keep


def new_plan():
    return {
        'experiment': 'phase_c_nofault_targeted_grounded_merge_repair',
        'design_version': 'phase_d_grounded_merge_feedback_v2',
        'model': 'qwen2.5:1.5b',
        'task_count': 30,
        'policies': list(POLICIES),
        'fault': 'none',
        'total_planned_trials': EXPECTED,
        'runner_sha256': digest(RUNNER),
        'benchmark_sha256': digest(DATA / 'benchmark_cases.csv'),
        'original_phase_c_sha256': digest(OLD / 'records.json'),
        'method_note': (
            'Fresh complete 30-country x 3-policy natural no-fault comparison under the '
            'corrected Phase D evidence-copy prompt and grounded feedback. '
            'No task answer is inserted into the model decision; observed MCP outputs are used. '
            'Original controlled fault-injection trials are preserved unchanged.'
        ),
    }


def verify_replacement_rows(mod, tasks, rows):
    if len(rows) != EXPECTED:
        raise RuntimeError(f"Replacement has {len(rows)}/90 trials; can resume with same command")
    expected = {(task['id'], condition, 'none') for task in tasks for condition in POLICIES}
    keys = [(r.get('task_id'), r.get('condition'), r.get('fault')) for r in rows]
    if set(keys) != expected or len(set(keys)) != EXPECTED:
        raise RuntimeError("Replacement no-fault trial-cell coverage mismatch")
    for r in rows:
        if (r.get('controlled_seed') != 0 or r.get('seed_source') != 'natural_end_to_end'
                or r.get('injected_fault') != 0 or r.get('model') != mod.MODEL
                or r.get('task_completed') not in (0, 1)):
            raise RuntimeError(f"Invalid replacement provenance/outcome at {r.get('task_id')}")
        if not isinstance(r.get('events'), list) or not r['events']:
            raise RuntimeError(f"Missing observable attempt events for {r.get('task_id')}")
        # Verify final success against its actual MCP source outputs, not merely a success flag.
        if r['task_completed']:
            ev = r['events']
            by_source = {e['output']['source']: e['output']['president_name'] for e in ev
                         if e.get('result') == 'success' and isinstance(e.get('output'), dict)
                         and e['output'].get('source') in ('wikidata', 'hdata')}
            decision = [e for e in ev if e.get('step_kind') == 'decision' and e.get('result') == 'success']
            if set(by_source) != {'wikidata', 'hdata'} or len(decision) != 1:
                raise RuntimeError(f"Successful trial without two verified sources: {r['task_id']}")
            out = decision[0]['output']
            if (mod.normalize(out.get('wikidata_president', '')) != mod.normalize(by_source['wikidata'])
                    or mod.normalize(out.get('hdata_president', '')) != mod.normalize(by_source['hdata'])
                    or not (mod.name_equivalent(out.get('president_name', ''), by_source['wikidata'])
                            and mod.name_equivalent(out.get('president_name', ''), by_source['hdata']))):
                raise RuntimeError(f"Misgraded successful decision: {r['task_id']}")


def create_combined(mod, tasks, original, keep, rerun):
    verify_replacement_rows(mod, tasks, rerun)
    # Construct in exactly the original plan order; old fault rows are untouched dicts.
    rerun_by_key = {(r['task_id'], r['condition'], 'none'): r for r in rerun}
    combined = []
    for old in original:
        key = (old['task_id'], old['condition'], old['fault'])
        if old['fault'] == 'none':
            replacement = dict(rerun_by_key[key])
            replacement['result_origin'] = 'new_natural_nofault_grounded_merge'
            combined.append(replacement)
        else:
            combined.append(old)  # copied unchanged, without editing fields or status
    assert len(combined) == 450
    assert [r for r in combined if r['fault'] != 'none'] == keep
    success_by_policy = {
        p: {'successful': sum(r['task_completed'] for r in combined if r['condition'] == p),
            'total': 150,
            'natural_nofault_successful': sum(r['task_completed'] for r in rerun if r['condition'] == p),
            'natural_nofault_total': 30}
        for p in POLICIES
    }
    input_hashes = {
        'original_phase_c_records': digest(OLD / 'records.json'),
        'replacement_nofault_records': digest(NEW / 'records.json'),
        'corrected_runner': digest(RUNNER),
        'prepared_benchmark': digest(DATA / 'benchmark_cases.csv'),
    }
    plan = {
        'experiment': 'phase_c_combined_corrected_30_country_comparison',
        'repair_version': 'grounded_merge_v2_targeted_nofault',
        'model': mod.MODEL,
        'task_count': 30,
        'total_planned_trials': 450,
        'previously_saved_controlled_fault_rows': 360,
        'new_natural_nofault_rows': 90,
        'method_note': (
            'Consolidated audit artifact, NOT 450 newly run trials. 360 controlled seeded fault '
            'trials are unchanged from redo_v3_multiref/records.json. 90 complete natural '
            'no-fault cells were rerun using the same grounded merge fix as Phase D. '
            'The original 450 records remain available; do not add old and new totals.'
        ),
        'source_hashes': input_hashes,
    }
    # Use a separate destination; fail on prior conflicting combined output.
    expected_sig = hashlib.sha256(json.dumps(plan, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if (COMBINED / 'manifest.json').exists():
        existing = load_json(COMBINED / 'manifest.json')
        if existing.get('plan_signature') != expected_sig:
            raise RuntimeError('Existing combined output has a different input signature; do not overwrite it.')
    store = mod.ResultStore(COMBINED, plan, ('task_id', 'condition', 'fault'))
    if store.records:
        if store.records != combined:
            raise RuntimeError('Existing combined records differ from reconstructed verified evidence.')
    else:
        # Provenance locked before output; these are copies of actual existing records.
        store.records = combined
        store.done = {store.key(row) for row in combined}
        atomic_json(store.records_path, combined)
    store.finalize(450)
    assert load_json(COMBINED / 'records.json') == combined
    report = {
        'status': 'PASS',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'original_phase_c': 450,
        'unchanged_controlled_fault_trials': 360,
        'new_complete_nofault_trials': 90,
        'combined_phase_c': 450,
        'other_verified_v3_experiments': 788 + 30,
        'research_rows_for_dissertation': 1268,
        'replacement_policies': list(POLICIES),
        'results_by_policy': success_by_policy,
        'input_sha256': input_hashes,
        'combined_sha256': digest(COMBINED / 'records.json'),
        'original_evidence_modified': False,
        'note': 'PASS checks row coverage and provenance, not that every LLM call succeeds.',
    }
    atomic_json(COMBINED / 'REPAIR_AUDIT.json', report)
    lines = [
        'PHASE C TARGETED REPAIR: PASS',
        'Original Phase C untouched: 450 rows',
        'Unchanged canonical fault-injection rows: 360',
        'Replaced natural no-fault rows: 90 (30 countries x 3 policies)',
        'Corrected combined Phase C: 450 rows',
        'Research total remains 1,268 (not 1,358)',
        'Corrected Phase C directory: redo_v4_multiref_corrected',
        'Use this new directory for Phase C results; retain redo_v3_multiref as historical original.',
    ]
    for p, vals in success_by_policy.items():
        lines.append(f"{p}: {vals['successful']}/{vals['total']} overall; "
                     f"no-fault {vals['natural_nofault_successful']}/30")
    (COMBINED / 'REPAIR_CHECK.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n' + '\n'.join(lines), flush=True)


async def execute(mod, tasks, original, keep):
    from agents import new_agent
    # Current output folder's plan hashes lock it to the old source + corrected runner.
    store = mod.ResultStore(NEW, new_plan(), ('task_id', 'condition', 'fault'))
    # Keep actual trial failures while omitting only redundant MCP exception traces.
    logging.getLogger("mcp.server.mcpserver.server").addFilter(
        _MissingReferenceTracebackFilter()
    )
    agent = new_agent('ollama', mod.MODEL)
    try:
        for task in tasks:
            for policy in POLICIES:
                key = {'task_id': task['id'], 'condition': policy, 'fault': 'none'}
                if store.has(key):
                    continue
                # Exactly one LLM agent for all four steps; do NOT run Phase D or any fault trials.
                row = await mod.run_multiref_trial(
                    [agent, agent, agent, agent], mod.RedoReferenceData(), task,
                    policy, 'none', feedback_style='schema_diff', seed_chain=None,
                )
                store.append(row)
                print(f"[Phase C repair {len(store.records)}/90] {task['country_code']} "
                      f"{policy}: {'PASS' if row['task_completed'] else 'FAIL'}", flush=True)
    finally:
        if hasattr(agent, 'aclose'):
            await agent.aclose()
    verify_replacement_rows(mod, tasks, store.records)
    store.finalize(90)
    create_combined(mod, tasks, original, keep, store.records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Validate scope and corrected code, no model calls')
    parser.add_argument('--verify-only', action='store_true', help='Rebuild/verify existing complete repair; no model calls')
    parser.add_argument('--diagnose', action='store_true', help='Display Ireland checkpoint failures; no model calls')
    args = parser.parse_args()
    if args.diagnose:
        return diagnose_saved_country()
    try:
        mod = load_runner()
        tasks, original, keep = preflight(mod)
        print('Only rerun: 90 natural no-fault Phase C trials (30 countries x 3 policies)')
        print('Do NOT rerun: 360 controlled Phase C trials; 30 Phase D trials; all other suites')
        if args.check:
            print('TARGETED REPAIR SELF-CHECK: PASS (zero LLM calls)')
            return 0
        if args.verify_only:
            replacement = load_json(NEW / 'records.json')
            create_combined(mod, tasks, original, keep, replacement)
            return 0
        asyncio.run(execute(mod, tasks, original, keep))
        return 0
    except KeyboardInterrupt:
        print('\nSTOPPED. Completed repair rows are checkpointed. Repeat the SAME command to resume.', file=sys.stderr)
        return 130
    except Exception as exc:
        print(f'PHASE C REPAIR STOPPED: {type(exc).__name__}: {exc}', file=sys.stderr)
        traceback.print_exception(exc, file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
