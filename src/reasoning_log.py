"""Immutable append-only JSONL writers for reasoning and disagreement logs."""

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from src.models import AgentDiagnosis

_REASONING_LOG = os.path.join("logs", "reasoning_log.jsonl")
_DISAGREEMENT_LOG = os.path.join("logs", "disagreement_log.jsonl")


def _ensure_logs_dir() -> None:
    os.makedirs("logs", exist_ok=True)


def log_consensus(
    *,
    patient_id: str,
    kappa_score: float,
    iterations: int,
    diagnostician_output: AgentDiagnosis,
    auditor_output: AgentDiagnosis,
    grounding_table: list[dict[str, Any]],
) -> str:
    """Append a successful consensus run to reasoning_log.jsonl. Returns the run_id."""
    _ensure_logs_dir()
    run_id = str(uuid.uuid4())
    final_codes = sorted(
        set(diagnostician_output.icd_codes_cited) | set(auditor_output.icd_codes_cited)
    )
    verification_tokens = [
        entry["verification_token"]
        for entry in grounding_table
        if entry.get("verification_token")
    ]
    record = {
        "run_id": run_id,
        "patient_id": patient_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kappa_score": kappa_score,
        "iterations": iterations,
        "final_icd_codes": final_codes,
        "grounding_table": grounding_table,
        "verification_tokens": verification_tokens,
    }
    with open(_REASONING_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return run_id


def log_disagreement(
    *,
    patient_id: str,
    final_kappa: float,
    iterations: int,
    contradiction_points: list[str],
    diagnostician_final: AgentDiagnosis,
    auditor_final: AgentDiagnosis,
) -> str:
    """Append a ConsensusFailure escalation to disagreement_log.jsonl. Returns the run_id."""
    _ensure_logs_dir()
    run_id = str(uuid.uuid4())
    record = {
        "run_id": run_id,
        "patient_id": patient_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "final_kappa": final_kappa,
        "iterations": iterations,
        "requires_physician_review": True,
        "contradiction_points": contradiction_points,
        "diagnostician_final_position": diagnostician_final.model_dump(),
        "auditor_final_position": auditor_final.model_dump(),
    }
    with open(_DISAGREEMENT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return run_id
