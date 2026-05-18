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


def read_grounding_results_for_patient(
    patient_id: str,
    since: datetime,
) -> list[dict]:
    """Return grounding log entries for *patient_id* written at or after *since*.

    Each returned dict has keys: icd_code, icd_version, stage, verification_token.
    When multiple entries exist for the same code (e.g. from a reflection retry),
    the latest one wins so the table reflects the final grounding outcome.
    """
    if not os.path.exists(_LOG_PATH):
        return []

    latest: dict[str, dict] = {}
    with open(_LOG_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("patient_id") != patient_id:
                continue
            ts = datetime.fromisoformat(entry["timestamp"])
            # Make both tz-aware or both tz-naive before comparing.
            if since.tzinfo is not None and ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            elif since.tzinfo is None and ts.tzinfo is not None:
                ts = ts.replace(tzinfo=None)
            if ts < since:
                continue
            code = entry.get("icd_code", "")
            if code not in latest or entry["timestamp"] > latest[code]["timestamp"]:
                latest[code] = entry

    return [
        {
            "icd_code": e["icd_code"],
            "icd_version": e.get("icd_version"),
            "stage": e.get("stage_matched"),
            "verification_token": e.get("verification_token"),
        }
        for e in latest.values()
    ]
