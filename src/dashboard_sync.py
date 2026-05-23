"""Rebuild reports/patients.json from JSONL logs.

Called after every patient audit so the dashboard auto-refreshes via fetch().
Each patient record includes a full `runs` list (newest-first) so the dashboard
can render per-run history without re-fetching.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).parent.parent
REPORTS_DIR = _ROOT / "reports"
LOGS_DIR = _ROOT / "logs"
REASONING_LOG = LOGS_DIR / "reasoning_log.jsonl"
DISAGREEMENT_LOG = LOGS_DIR / "disagreement_log.jsonl"


def _parse_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(re.sub(r"\bNaN\b", "null", line)))
            except json.JSONDecodeError:
                pass
    return records


def _safe_kappa(value) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
        return None if not math.isfinite(f) else f
    except (TypeError, ValueError):
        return None


def refresh_patients_json() -> None:
    """Rebuild reports/patients.json from the current JSONL logs (best-effort)."""
    patient_runs: dict[str, list[dict]] = {}

    for run in _parse_jsonl(REASONING_LOG):
        pid = run.get("patient_id")
        if not pid:
            continue
        ts_str = run.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            ts = datetime.min
        record: dict = {
            "run_id": run.get("run_id"),
            "status": "CONSENSUS",
            "kappa": _safe_kappa(run.get("kappa_score")),
            "iterations": run.get("iterations", 1),
            "timestamp_str": ts_str,
            "icd_codes": run.get("final_icd_codes", []),
            "grounding": run.get("grounding_table", []),
            "diagnosis": "Consensus Reached (See patient details)",
            "evidence": [],
            "diagnostician_position": run.get("diagnostician_position"),
            "auditor_position": run.get("auditor_position"),
            "contradiction_points": [],
            "_ts": ts,
        }
        patient_runs.setdefault(pid, []).append(record)

    for run in _parse_jsonl(DISAGREEMENT_LOG):
        pid = run.get("patient_id")
        if not pid:
            continue
        ts_str = run.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            ts = datetime.min
        diag_final = run.get("diagnostician_final_position") or {}
        record = {
            "run_id": run.get("run_id"),
            "status": "ESCALATED",
            "kappa": _safe_kappa(run.get("final_kappa")),
            "iterations": run.get("iterations", 3),
            "timestamp_str": ts_str,
            "icd_codes": diag_final.get("icd_codes_cited", []),
            "grounding": [],
            "diagnosis": diag_final.get("diagnosis_hypothesis", "Requires Physician Review"),
            "evidence": diag_final.get("evidence_chain", []),
            "diagnostician_position": diag_final if diag_final else None,
            "auditor_position": run.get("auditor_final_position"),
            "contradiction_points": run.get("contradiction_points", []),
            "_ts": ts,
        }
        patient_runs.setdefault(pid, []).append(record)

    patients_out: list[dict] = []
    for pid in sorted(patient_runs.keys()):
        runs = sorted(patient_runs[pid], key=lambda r: r["_ts"], reverse=True)
        for r in runs:
            r.pop("_ts", None)
        latest = runs[0]
        patients_out.append({
            "patient_id": pid,
            "status": latest["status"],
            "kappa": latest["kappa"],
            "iterations": latest["iterations"],
            "timestamp_str": latest["timestamp_str"],
            "icd_codes": latest["icd_codes"],
            "grounding": latest["grounding"],
            "diagnosis": latest["diagnosis"],
            "evidence": latest["evidence"],
            "diagnostician_position": latest["diagnostician_position"],
            "auditor_position": latest["auditor_position"],
            "contradiction_points": latest["contradiction_points"],
            "run_count": len(runs),
            "runs": runs,
        })

    total = len(patients_out)
    consensus_count = sum(1 for p in patients_out if p["status"] == "CONSENSUS")
    escalated_count = total - consensus_count
    valid_kappas = [p["kappa"] for p in patients_out if p["kappa"] is not None]
    avg_kappa = round(sum(valid_kappas) / len(valid_kappas), 3) if valid_kappas else 0.0

    output = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stats": {
            "total_audited": total,
            "consensus_count": consensus_count,
            "escalated_count": escalated_count,
            "consensus_rate_pct": round(consensus_count / total * 100, 1) if total > 0 else 0.0,
            "avg_kappa": avg_kappa,
        },
        "patients": patients_out,
    }

    REPORTS_DIR.mkdir(exist_ok=True)
    with open(REPORTS_DIR / "patients.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
