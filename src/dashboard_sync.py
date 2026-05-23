"""Rebuild reports/patients.json from JSONL logs.

Called after every patient audit so the dashboard auto-refreshes via fetch().
This is intentionally lightweight — no HTML rendering, just JSON.
"""

from __future__ import annotations

import json
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


def refresh_patients_json() -> None:
    """Rebuild reports/patients.json from the current JSONL logs (best-effort)."""
    patient_latest: dict[str, dict] = {}

    for run in _parse_jsonl(REASONING_LOG):
        pid = run.get("patient_id")
        if not pid:
            continue
        ts_str = run.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            ts = datetime.min
        kappa = run.get("kappa_score")
        try:
            kappa_val = float(kappa) if kappa is not None else None
        except (ValueError, TypeError):
            kappa_val = None
        record: dict = {
            "patient_id": pid,
            "status": "CONSENSUS",
            "kappa": kappa_val,
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
        if pid not in patient_latest or ts > patient_latest[pid]["_ts"]:
            patient_latest[pid] = record

    for run in _parse_jsonl(DISAGREEMENT_LOG):
        pid = run.get("patient_id")
        if not pid:
            continue
        ts_str = run.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            ts = datetime.min
        kappa = run.get("final_kappa")
        try:
            kappa_val = float(kappa) if kappa is not None else None
        except (ValueError, TypeError):
            kappa_val = None
        diag_final = run.get("diagnostician_final_position") or {}
        record = {
            "patient_id": pid,
            "status": "ESCALATED",
            "kappa": kappa_val,
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
        if pid not in patient_latest or ts > patient_latest[pid]["_ts"]:
            patient_latest[pid] = record

    sorted_patients = sorted(patient_latest.values(), key=lambda p: p["patient_id"])
    for p in sorted_patients:
        p.pop("_ts", None)

    total = len(sorted_patients)
    consensus_count = sum(1 for p in sorted_patients if p["status"] == "CONSENSUS")
    escalated_count = total - consensus_count
    valid_kappas = [p["kappa"] for p in sorted_patients if p["kappa"] is not None]
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
        "patients": sorted_patients,
    }

    REPORTS_DIR.mkdir(exist_ok=True)
    with open(REPORTS_DIR / "patients.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
