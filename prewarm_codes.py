#!/usr/bin/env python3
"""
Pre-warm the ICD code cache from the local MIMIC-IV dictionary.

Run this once before batch_run.py to populate data/icd_cache.db with every
ICD code in diagnoses_icd.csv.gz.  Resolution priority:

  1. Already in cache        → skip (no work needed)
  2. In d_icd_diagnoses.csv  → write directly, zero network calls
  3. Not in local dict       → fall through to the live 3-stage API pipeline

After this script finishes, all audit runs read exclusively from cache.
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
            if os.name == "nt":
                venv_python = venv_dir / "Scripts" / "python.exe"
            else:
                venv_python = venv_dir / "bin" / "python"

            if venv_python.exists():
                args = [str(venv_python)] + sys.argv
                sys.exit(subprocess.call(args))

import concurrent.futures
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.icd_client import _CACHE_DB
from src.tools import _ground_single_code
from src.logger import get_logger

logger = get_logger(__name__)

_HOSP_DIR = Path("data/mimic-iv-clinical-database-demo-2.2/hosp")
_DICT_FILE = _HOSP_DIR / "d_icd_diagnoses.csv.gz"
_DIAG_FILE = _HOSP_DIR / "diagnoses_icd.csv.gz"

_db_write_lock = Lock()


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_local_dict() -> dict[tuple[str, str], str]:
    """Return {(icd_code, icd_version): long_title} from the MIMIC ICD dictionary."""
    df = pd.read_csv(_DICT_FILE, compression="gzip", dtype=str)
    result: dict[tuple[str, str], str] = {}
    for _, row in df.iterrows():
        code = str(row["icd_code"]).strip()
        try:
            version = str(int(float(str(row["icd_version"]))))
        except (ValueError, TypeError):
            version = str(row["icd_version"]).strip()
        title = str(row.get("long_title", "")).strip()
        result[(code, version)] = title
    return result


def _load_all_codes() -> list[tuple[str, str]]:
    """Return sorted unique (icd_code, icd_version) pairs from diagnoses_icd.csv.gz."""
    codes: set[tuple[str, str]] = set()
    for chunk in pd.read_csv(_DIAG_FILE, compression="gzip", chunksize=50_000, dtype=str):
        for _, row in chunk.iterrows():
            code = str(row.get("icd_code", "")).strip()
            try:
                version = str(int(float(str(row.get("icd_version", "9")))))
            except (ValueError, TypeError):
                version = "9"
            if code:
                codes.add((code, version))
    return sorted(codes)


def _already_cached(db: sqlite3.Connection, code: str, version: str) -> bool:
    row = db.execute(
        "SELECT 1 FROM icd_cache "
        "WHERE icd_code=? AND icd_version=? AND lookup_type='exact' AND term IS NULL",
        (code, version),
    ).fetchone()
    return row is not None


def _write_local_hit(db: sqlite3.Connection, code: str, version: str, title: str) -> None:
    token = f"LOCAL_SIG_{code.replace('.', '_')}"
    response = json.dumps({
        "code": code,
        "title": title,
        "source": "MIMIC_LOCAL",
        "token": token,
    })
    with _db_write_lock:
        db.execute(
            "INSERT OR REPLACE INTO icd_cache "
            "(icd_code, icd_version, lookup_type, term, response, cached_at) "
            "VALUES (?, ?, 'exact', NULL, ?, ?)",
            (code, version, response, datetime.now(timezone.utc).isoformat()),
        )


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    for path in (_DICT_FILE, _DIAG_FILE):
        if not path.exists():
            print(f"[ERROR] File not found: {path}", file=sys.stderr)
            sys.exit(1)

    print("\n[PREWARM] Loading MIMIC ICD dictionary …")
    local_dict = _load_local_dict()
    print(f"[PREWARM] Local dictionary: {len(local_dict):,} entries")

    print("[PREWARM] Scanning diagnoses_icd.csv.gz for unique codes …")
    all_codes = _load_all_codes()
    print(f"[PREWARM] Unique codes in dataset: {len(all_codes):,}")

    os.makedirs(os.path.dirname(_CACHE_DB), exist_ok=True)
    db = sqlite3.connect(_CACHE_DB, check_same_thread=False)
    # ensure table exists (icd_client creates it, but we may run before it)
    db.execute("""
        CREATE TABLE IF NOT EXISTS icd_cache (
            icd_code    TEXT NOT NULL,
            icd_version TEXT NOT NULL,
            lookup_type TEXT NOT NULL,
            term        TEXT,
            response    TEXT NOT NULL,
            cached_at   TEXT NOT NULL,
            PRIMARY KEY (icd_code, icd_version, lookup_type, term)
        )
    """)
    db.commit()

    # Partition codes
    already_done: list[tuple[str, str]] = []
    local_hits: list[tuple[str, str]] = []
    api_needed: list[tuple[str, str]] = []

    for code, version in all_codes:
        if _already_cached(db, code, version):
            already_done.append((code, version))
        elif (code, version) in local_dict:
            local_hits.append((code, version))
        else:
            api_needed.append((code, version))

    print(f"\n[PREWARM] Already cached :  {len(already_done):>4}")
    print(f"[PREWARM] Local dict hits :  {len(local_hits):>4}  → write directly, no network")
    print(f"[PREWARM] API pipeline    :  {len(api_needed):>4}  → 3-stage grounding")

    # ── write local hits ─────────────────────────────────────────────────────
    if local_hits:
        print(f"\n[PREWARM] Writing {len(local_hits):,} local entries to cache …")
        for code, version in local_hits:
            _write_local_hit(db, code, version, local_dict[(code, version)])
        db.commit()
        print(f"[PREWARM] Local entries written.")

    # ── API grounding for remaining codes ─────────────────────────────────────
    api_verified = 0
    api_unverified = 0

    if api_needed:
        print(f"\n[PREWARM] Grounding {len(api_needed)} codes via live API pipeline …")
        max_workers = min(10, len(api_needed))
        done = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(_ground_single_code, code, ver, "PREWARM"): (code, ver)
                for code, ver in api_needed
            }
            for future in concurrent.futures.as_completed(future_map):
                code, ver = future_map[future]
                done += 1
                try:
                    result = future.result()
                    if result.get("status") == "VERIFIED":
                        api_verified += 1
                    else:
                        api_unverified += 1
                        logger.debug("Unverified: ICD-%s %s", ver, code)
                except Exception as exc:
                    api_unverified += 1
                    logger.warning("Exception grounding ICD-%s %s: %s", ver, code, exc)

                if done % 20 == 0 or done == len(api_needed):
                    pct = done / len(api_needed) * 100
                    print(
                        f"  [{done:>4}/{len(api_needed)}]  {pct:5.1f}%"
                        f"  verified: {api_verified}  unverified: {api_unverified}"
                    )

    # ── summary ──────────────────────────────────────────────────────────────
    total_ready = len(already_done) + len(local_hits) + api_verified
    total = len(all_codes)

    print()
    print("━" * 52)
    print("  PREWARM COMPLETE")
    print("━" * 52)
    print(f"  Pre-existing cache   : {len(already_done):>4}")
    print(f"  Local dict resolved  : {len(local_hits):>4}")
    print(f"  API verified         : {api_verified:>4}")
    print(f"  Unresolved (no match): {api_unverified:>4}")
    print(f"  ─────────────────────────────")
    print(f"  Total codes ready    : {total_ready:>4} / {total}")
    print("━" * 52)
    print()
    print("  Run batch_run.py — all grounding will now hit the local cache.")
    print()

    db.close()


if __name__ == "__main__":
    main()
