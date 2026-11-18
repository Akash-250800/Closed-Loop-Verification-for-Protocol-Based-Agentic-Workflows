"""One-command guarded multi-reference run for the deadline build.

Runs a 9-cell gate first. The 36-cell main run starts only if the gate shows:
- all no-fault policies complete;
- closed-loop recovers both injected faults;
- no-verification and naive-retry do not falsely pass the corrupted calls.
"""
from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
MODEL = "qwen2.5:1.5b"
GATE = ROOT / "results_multiref_gate"
MAIN = ROOT / "results_multiref_final"


def run(*args):
    print("\n>", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True)


def archive(path: Path):
    if not path.exists():
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = path.with_name(path.name + "_previous_" + stamp)
    path.rename(target)
    print(f"Preserved old results: {target.name}", flush=True)


def read_trials(path: Path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def as_int(row, key):
    return int(float(row[key]))


def validate_gate():
    rows = read_trials(GATE / "trials.csv")
    if len(rows) != 9:
        raise RuntimeError(f"Gate produced {len(rows)} rows, expected 9")
    by = {(r["fault"], r["condition"]): r for r in rows}

    for condition in ("no_verification", "naive_retry", "closed_loop"):
        if as_int(by[("none", condition)], "task_completed") != 1:
            raise RuntimeError(f"Gate failed: no-fault {condition} did not complete")

    for fault in ("type_mismatch", "wrong_tool"):
        closed = by[(fault, "closed_loop")]
        if as_int(closed, "task_completed") != 1:
            raise RuntimeError(f"Gate failed: closed_loop did not complete {fault}")
        if as_int(closed, "injected_fault_recovered") != 1:
            raise RuntimeError(f"Gate failed: closed_loop did not record recovery for {fault}")
        if as_int(by[(fault, "no_verification")], "task_completed") != 0:
            raise RuntimeError(f"Gate failed: no_verification unexpectedly completed {fault}")
        if as_int(by[(fault, "naive_retry")], "task_completed") != 0:
            raise RuntimeError(f"Gate failed: naive_retry unexpectedly completed {fault}")

    print("\nGATE PASS: recovery behavior is correct. Starting final run.", flush=True)


def validate_final():
    rows = read_trials(MAIN / "trials.csv")
    if len(rows) != 36:
        raise RuntimeError(f"Final run produced {len(rows)} rows, expected 36")
    closed_faulted = [r for r in rows if r["condition"] == "closed_loop" and r["fault"] != "none"]
    recovered = sum(as_int(r, "task_completed") for r in closed_faulted)
    print(f"\nFINAL CHECK: closed-loop completed {recovered}/{len(closed_faulted)} faulted multi-reference tasks.", flush=True)
    if recovered != len(closed_faulted):
        raise RuntimeError("Final run completed, but at least one closed-loop faulted task failed. Keep the results; do not rerun automatically.")


def main():
    # Do not delete evidence. Existing folders are renamed before new runs.
    archive(GATE)
    archive(MAIN)

    run(PYTHON, "-m", "pytest", "-q")
    run(PYTHON, "check_ollama.py")

    run(
        PYTHON, "experiment.py",
        "--mode", "ollama", "--model", MODEL,
        "--benchmark", "multiref", "--tasks", "1", "--replicates", "1",
        "--suite", "baselines", "--faults", "none,type_mismatch,wrong_tool",
        "--retry-budgets", "2", "--out", GATE.name,
    )
    validate_gate()

    run(
        PYTHON, "experiment.py",
        "--mode", "ollama", "--model", MODEL,
        "--benchmark", "multiref", "--tasks", "4", "--replicates", "1",
        "--suite", "baselines", "--faults", "none,type_mismatch,wrong_tool",
        "--retry-budgets", "2", "--out", MAIN.name,
    )
    run(PYTHON, "analyse.py", "--results", MAIN.name)
    validate_final()

    print("\nMULTI-REFERENCE FINAL RUN COMPLETE AND VALIDATED", flush=True)
    print(f"Results: {MAIN}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"\nSTOP: command failed with exit code {exc.returncode}. Existing results were preserved.", file=sys.stderr)
        raise SystemExit(exc.returncode)
    except Exception as exc:
        print(f"\nSTOP: {exc}", file=sys.stderr)
        raise SystemExit(2)
