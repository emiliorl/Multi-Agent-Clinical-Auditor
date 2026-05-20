#!/usr/bin/env python3
"""Consecutive batch runner: audit multiple MIMIC-IV patients in sequence."""

import argparse
import csv
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

DEMO_CSV = (
    Path(__file__).parent
    / "data"
    / "mimic-iv-clinical-database-demo-2.2"
    / "demo_subject_id.csv"
)
REPORTS_DIR = Path(__file__).parent / "reports"
PYTHON = sys.executable
_BOX_W = 74


def _load_patient_ids() -> list[str]:
    ids: list[str] = []
    with open(DEMO_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("subject_id", "").strip()
            if pid:
                ids.append(pid)
    return ids


def _extract_metrics(stdout: str, stderr: str, duration: float) -> dict:
    combined = stdout + "\n" + stderr

    # κ value — printed by main.py as "κ=0.923" or "kappa=0.923"
    kappa: float | None = None
    m = re.search(r"[kκ](?:appa)?=([0-9]+\.[0-9]+)", combined, re.IGNORECASE)
    if m:
        kappa = float(m.group(1))

    # Iteration count
    iterations: int | None = None
    m = re.search(r"iteration[s]? (\d+)", combined, re.IGNORECASE)
    if m:
        iterations = int(m.group(1))
    else:
        m = re.search(r"after (\d+) iteration", combined, re.IGNORECASE)
        if m:
            iterations = int(m.group(1))

    if "[ESCALATION]" in stdout:
        status = "ESCALATED"
    elif "[RESULT]" in stdout:
        status = "CONSENSUS"
    elif kappa is not None:
        status = "CONSENSUS" if kappa > 0.80 else "ESCALATED"
    else:
        status = "UNKNOWN"

    return {
        "kappa": kappa,
        "iterations": iterations,
        "status": status,
        "duration_s": round(duration, 1),
    }


def _run_patient(patient_id: str) -> dict:
    import sys
    import os
    start = time.time()
    
    # Force the child process to use UTF-8 encoding for standard I/O on Windows
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    
    proc = subprocess.Popen(
        [PYTHON, "main.py", "--patient", patient_id],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=Path(__file__).parent,
    )
    
    accumulated_output = []
    
    # Read output line by line in real-time and stream to terminal
    if proc.stdout:
        for line in iter(proc.stdout.readline, ""):
            # Safely encode/decode using the terminal's native encoding to prevent CP1252 write crashes
            terminal_encoding = sys.stdout.encoding or "utf-8"
            safe_line = line.encode(terminal_encoding, errors="replace").decode(terminal_encoding)
            sys.stdout.write(safe_line)
            sys.stdout.flush()
            accumulated_output.append(line)
            
    proc.wait()
    duration = time.time() - start
    full_output = "".join(accumulated_output)
    
    # Extract metrics using the full merged output
    metrics = _extract_metrics(full_output, "", duration)
    metrics["patient_id"] = patient_id
    metrics["returncode"] = proc.returncode
    tail = full_output[-800:].strip()
    metrics["error"] = tail if (proc.returncode != 0 or metrics["status"] == "UNKNOWN") else None
    return metrics


def _status_icon(status: str, returncode: int) -> str:
    if returncode != 0 and status == "UNKNOWN":
        return "✗ FAILED"
    if status == "ESCALATED":
        return "⚠ ESCALATED"
    return "✓ CONSENSUS"


def _print_summary(results: list[dict]) -> None:
    print()
    print("═" * _BOX_W)
    print("  CLINICAL AUDIT BATCH SUMMARY".center(_BOX_W))
    print("═" * _BOX_W)

    col = "{:<14}{:<16}{:<11}{:<13}{:<10}{}"
    print("  " + col.format("Patient ID", "Status", "κ Score", "Iterations", "Duration", "Note"))
    print("  " + "─" * (_BOX_W - 2))

    consensus_count = escalated_count = failed_count = 0

    for r in results:
        pid = r["patient_id"]
        kappa_s = f"{r['kappa']:.3f}" if r["kappa"] is not None else "—"
        iters_s = str(r["iterations"]) if r["iterations"] is not None else "—"
        dur_s = f"{r['duration_s']}s"
        note = ""

        if r["returncode"] != 0 and r["status"] == "UNKNOWN":
            note = "See report for error"
            failed_count += 1
        elif r["status"] == "ESCALATED":
            note = "Physician review required"
            escalated_count += 1
        else:
            consensus_count += 1

        print("  " + col.format(pid, r["status"], kappa_s, iters_s, dur_s, note))

    print("  " + "─" * (_BOX_W - 2))
    total = len(results)
    valid_kappas = [r["kappa"] for r in results if r["kappa"] is not None]
    avg_kappa = sum(valid_kappas) / len(valid_kappas) if valid_kappas else 0.0
    total_time = sum(r["duration_s"] for r in results)
    print(
        f"  Patients: {total}  |  "
        f"Consensus: {consensus_count}  |  "
        f"Escalated: {escalated_count}  |  "
        f"Failed: {failed_count}"
    )
    print(
        f"  Avg κ: {avg_kappa:.3f}  |  "
        f"Total time: {total_time:.0f}s ({total_time / 60:.1f} min)"
    )
    print("═" * _BOX_W)
    print()


def _save_report(results: list[dict], batch_ts: str) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    report_path = REPORTS_DIR / f"batch_{batch_ts}.md"

    ts_human = datetime.strptime(batch_ts, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    valid_kappas = [r["kappa"] for r in results if r["kappa"] is not None]
    avg_kappa = sum(valid_kappas) / len(valid_kappas) if valid_kappas else 0.0
    consensus = sum(1 for r in results if r["status"] == "CONSENSUS")
    escalated = sum(1 for r in results if r["status"] == "ESCALATED")
    failed = sum(1 for r in results if r["returncode"] != 0 and r["status"] == "UNKNOWN")
    total_time = sum(r["duration_s"] for r in results)

    lines = [
        "# Clinical Audit Batch Report",
        "",
        f"**Run date**: {ts_human}  ",
        f"**Patients audited**: {len(results)}  ",
        f"**Total runtime**: {total_time:.0f}s ({total_time / 60:.1f} min)",
        "",
        "## Results",
        "",
        "| Patient ID | Status | κ Score | Iterations | Duration |",
        "|:-----------|:-------|--------:|-----------:|---------:|",
    ]
    for r in results:
        kappa_s = f"{r['kappa']:.3f}" if r["kappa"] is not None else "—"
        iters_s = str(r["iterations"]) if r["iterations"] is not None else "—"
        lines.append(
            f"| {r['patient_id']} | {r['status']} | {kappa_s} | {iters_s} | {r['duration_s']}s |"
        )

    lines += [
        "",
        "## Summary",
        "",
        f"- **Consensus reached**: {consensus}/{len(results)}",
        f"- **Escalated for review**: {escalated}",
        f"- **Failed**: {failed}",
        f"- **Average κ**: {avg_kappa:.3f}",
    ]

    error_cases = [r for r in results if r.get("error")]
    if error_cases:
        lines += ["", "## Error Details", ""]
        for r in error_cases:
            lines += [
                f"### Patient {r['patient_id']} (exit code {r['returncode']})",
                "```",
                r["error"] or "(no output captured)",
                "```",
                "",
            ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Consecutive batch clinical auditor")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--count",
        type=int,
        help="Number of patients to audit (reads from demo_subject_id.csv)",
    )
    group.add_argument(
        "--patients",
        help="Comma-separated list of patient IDs to audit",
    )
    args = parser.parse_args()

    if args.patients:
        patient_ids = [p.strip() for p in args.patients.split(",") if p.strip()]
    else:
        all_ids = _load_patient_ids()
        if args.count > len(all_ids):
            print(
                f"[WARN] Requested {args.count} patients but only {len(all_ids)} available — "
                f"running all {len(all_ids)}."
            )
        patient_ids = all_ids[: args.count]

    if not patient_ids:
        print("[ERROR] No patient IDs to process.", file=sys.stderr)
        sys.exit(1)

    batch_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    total = len(patient_ids)
    print(f"\n[BATCH] Starting consecutive audit for {total} patient(s) — {batch_ts}")

    results: list[dict] = []
    for i, pid in enumerate(patient_ids, 1):
        print(f"\n[{i}/{total}] Auditing patient {pid} ...")
        try:
            result = _run_patient(pid)
        except Exception as exc:
            result = {
                "patient_id": pid,
                "kappa": None,
                "iterations": None,
                "status": "UNKNOWN",
                "duration_s": 0.0,
                "returncode": -1,
                "error": str(exc),
            }
        results.append(result)
        kappa_s = f"κ={result['kappa']:.3f}" if result["kappa"] is not None else "κ=?"
        print(f"  → {result['status']} | {kappa_s} | {result['duration_s']}s")
        if result["status"] == "UNKNOWN" and result.get("error"):
            for line in result["error"].splitlines()[-6:]:
                print(f"     {line}")

    _print_summary(results)
    report_path = _save_report(results, batch_ts)
    print(f"[BATCH] Report saved → {report_path}\n")


if __name__ == "__main__":
    main()
