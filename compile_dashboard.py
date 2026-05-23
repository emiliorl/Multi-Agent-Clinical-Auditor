#!/usr/bin/env python3
"""
compile_dashboard.py - Reads reasoning_log.jsonl and disagreement_log.jsonl,
writes reports/patients.json (consumed by the live dashboard) and a consolidated
Markdown report.

The HTML dashboard shell (reports/dashboard.html) is data-free and never needs
to be regenerated — it fetches patients.json at runtime and polls every 15 s.
"""

if __name__ == "__main__":
    import os
    import sys
    import subprocess
    from pathlib import Path

    venv_dir = Path(__file__).parent.resolve() / "venv"
    if venv_dir.exists():
        try:
            is_in_venv = Path(sys.executable).resolve().is_relative_to(venv_dir)
        except AttributeError:
            try:
                Path(sys.executable).resolve().relative_to(venv_dir)
                is_in_venv = True
            except ValueError:
                is_in_venv = False

        if not is_in_venv:
            venv_python = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            if venv_python.exists():
                sys.exit(subprocess.call([str(venv_python)] + sys.argv))

import json
import re
from datetime import datetime
from pathlib import Path

REPORTS_DIR = Path(__file__).parent / "reports"
LOGS_DIR    = Path(__file__).parent / "logs"
REASONING_LOG    = LOGS_DIR / "reasoning_log.jsonl"
DISAGREEMENT_LOG = LOGS_DIR / "disagreement_log.jsonl"


def safe_parse_jsonl(file_path: Path) -> list[dict]:
    records = []
    if not file_path.exists():
        return records
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(re.sub(r"\bNaN\b", "null", line)))
            except json.JSONDecodeError as exc:
                print(f"[WARN] Failed to parse line {i} in {file_path.name}: {exc}")
    return records


def build_consolidated_data() -> tuple[list[dict], dict]:
    """Merge logs, deduplicate (latest run per patient), compute stats."""
    patient_latest: dict[str, dict] = {}

    for run in safe_parse_jsonl(REASONING_LOG):
        pid = run.get("patient_id")
        if not pid:
            continue
        ts_str = run.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            ts = datetime.min
        try:
            kappa_val = float(run["kappa_score"]) if run.get("kappa_score") is not None else None
        except (ValueError, TypeError):
            kappa_val = None
        record = {
            "patient_id": pid,
            "status": "CONSENSUS",
            "kappa": kappa_val,
            "iterations": run.get("iterations", 1),
            "timestamp": ts,
            "timestamp_str": ts_str,
            "icd_codes": run.get("final_icd_codes", []),
            "grounding": run.get("grounding_table", []),
            "diagnosis": "Consensus Reached (See patient details)",
            "evidence": [],
            "diagnostician_position": run.get("diagnostician_position"),
            "auditor_position": run.get("auditor_position"),
            "contradiction_points": [],
        }
        if pid not in patient_latest or ts > patient_latest[pid]["timestamp"]:
            patient_latest[pid] = record

    for run in safe_parse_jsonl(DISAGREEMENT_LOG):
        pid = run.get("patient_id")
        if not pid:
            continue
        ts_str = run.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            ts = datetime.min
        try:
            kappa_val = float(run["final_kappa"]) if run.get("final_kappa") is not None else None
        except (ValueError, TypeError):
            kappa_val = None
        diag_final = run.get("diagnostician_final_position") or {}
        record = {
            "patient_id": pid,
            "status": "ESCALATED",
            "kappa": kappa_val,
            "iterations": run.get("iterations", 3),
            "timestamp": ts,
            "timestamp_str": ts_str,
            "icd_codes": diag_final.get("icd_codes_cited", []),
            "grounding": [],
            "diagnosis": diag_final.get("diagnosis_hypothesis", "") or "Requires Physician Review",
            "evidence": diag_final.get("evidence_chain", []),
            "diagnostician_position": diag_final if diag_final else None,
            "auditor_position": run.get("auditor_final_position"),
            "contradiction_points": run.get("contradiction_points", []),
        }
        if pid not in patient_latest or ts > patient_latest[pid]["timestamp"]:
            patient_latest[pid] = record

    sorted_patients = [patient_latest[pid] for pid in sorted(patient_latest)]
    total          = len(sorted_patients)
    consensus_count = sum(1 for p in sorted_patients if p["status"] == "CONSENSUS")
    escalated_count = total - consensus_count
    valid_kappas   = [p["kappa"] for p in sorted_patients if p["kappa"] is not None]
    avg_kappa      = sum(valid_kappas) / len(valid_kappas) if valid_kappas else 0.0

    stats = {
        "total_audited":      total,
        "consensus_count":    consensus_count,
        "escalated_count":    escalated_count,
        "consensus_rate_pct": round(consensus_count / total * 100, 1) if total else 0.0,
        "avg_kappa":          round(avg_kappa, 3),
    }
    return sorted_patients, stats


def generate_patients_json(patients: list[dict], stats: dict) -> Path:
    """Write reports/patients.json — the live data file fetched by the dashboard."""
    REPORTS_DIR.mkdir(exist_ok=True)
    serialized = [{k: v for k, v in p.items() if k != "timestamp"} for p in patients]
    output = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stats": stats,
        "patients": serialized,
    }
    json_path = REPORTS_DIR / "patients.json"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return json_path


def generate_markdown_report(patients: list[dict], stats: dict) -> Path:
    """Write reports/consolidated_report.md."""
    REPORTS_DIR.mkdir(exist_ok=True)
    lines = [
        "# Consolidated Clinical Audit Report",
        "",
        f"**Compiled at**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Total Patients Audited**: {stats['total_audited']}  ",
        f"**Consensus Rate**: {stats['consensus_rate_pct']}% ({stats['consensus_count']}/{stats['total_audited']})  ",
        f"**Average κ Agreement Score**: {stats['avg_kappa']:.3f}  ",
        "",
        "## Summary Metrics",
        "",
        f"- **Consensus Reached**: {stats['consensus_count']}",
        f"- **Escalated for Physician Review**: {stats['escalated_count']}",
        f"- **Average Agreement Score (κ)**: {stats['avg_kappa']:.3f}",
        "",
        "## Patient Audit Status Table",
        "",
        "| Patient ID | Audit Status | Agreement (κ) | Iterations | Grounded Codes | Last Audited Time |",
        "|:-----------|:-------------|--------------:|-----------:|:---------------|:------------------|",
    ]
    for p in patients:
        kappa_s  = f"{p['kappa']:.3f}" if p["kappa"] is not None else "—"
        codes_s  = ", ".join(p["icd_codes"][:8]) + ("..." if len(p["icd_codes"]) > 8 else "")
        ts_clean = p["timestamp_str"].split(".")[0].replace("T", " ") if p["timestamp_str"] else "—"
        status_label = "🟢 CONSENSUS" if p["status"] == "CONSENSUS" else "🔴 ESCALATED"
        lines.append(
            f"| **{p['patient_id']}** | {status_label} | {kappa_s} | {p['iterations']} "
            f"| `{codes_s or 'None'}` | {ts_clean} |"
        )
    lines += [
        "",
        "---",
        "*Compiled from reasoning_log.jsonl and disagreement_log.jsonl — "
        "always shows the latest audit per patient.*",
    ]
    report_path = REPORTS_DIR / "consolidated_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def generate_html_dashboard(_patients: list[dict], _stats: dict) -> Path:
    """Return the path to the static dashboard shell (data-free HTML).

    The shell fetches patients.json at runtime, so it never needs regeneration.
    """
    dashboard_path = REPORTS_DIR / "dashboard.html"
    if not dashboard_path.exists():
        raise FileNotFoundError(
            f"{dashboard_path} not found. Restore it from version control."
        )
    return dashboard_path


def main() -> None:
    print("[DASHBOARD] Scanning JSONL logs to compile clinical audits...")
    patients, stats = build_consolidated_data()

    if not patients:
        print("[ERROR] No clinical audits found in logs directory. Please run batch_run.py first.")
        return

    md_path   = generate_markdown_report(patients, stats)
    json_path = generate_patients_json(patients, stats)
    html_path = generate_html_dashboard(patients, stats)

    print("═" * 60)
    print("  CLINICAL AUDIT LOGS CONSOLIDATION COMPLETE".center(60))
    print("═" * 60)
    print(f"  Total Audited Patients : {stats['total_audited']}")
    print(f"  Consensus Reached      : {stats['consensus_count']} ({stats['consensus_rate_pct']}%)")
    print(f"  Physician Escalated    : {stats['escalated_count']}")
    print(f"  Mean Agreement κ Score : {stats['avg_kappa']:.3f}")
    print("─" * 60)
    print(f"  [+] Markdown Report    : {md_path.absolute()}")
    print(f"  [+] Patient Data JSON  : {json_path.absolute()}")
    print(f"  [+] HTML Dashboard     : {html_path.absolute()}")
    print("═" * 60)
    print()


if __name__ == "__main__":
    main()
