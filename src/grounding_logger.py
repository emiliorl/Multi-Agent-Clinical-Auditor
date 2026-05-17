"""Append-only JSONL writer for per-code grounding attempts."""

import json
import os
from datetime import datetime, timezone

_LOG_PATH = os.path.join("logs", "grounding_log.jsonl")


def log_grounding_attempt(
    *,
    patient_id: str,
    icd_code: str,
    icd_version: str,
    stage_matched: int | None,   # 1, 2, 3, or None for failed
    similarity_score: float | None,
    verification_token: str | None,
) -> None:
    os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
    record = {
        "patient_id": patient_id,
        "icd_code": icd_code,
        "icd_version": icd_version,
        "stage_matched": stage_matched,
        "similarity_score": similarity_score,
        "verification_token": verification_token,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
