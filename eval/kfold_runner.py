"""K-fold cross-validation runner for the Multi-Agent Clinical Auditor.

Design goals (after the $0.10 silent-crash incident on 2026-05-27):
  • Fail LOUDLY on any unknown exception — halt the run, write .PAUSED sentinel,
    exit non-zero. No more silently swallowed errors.
  • Checkpoint every patient to <run>.checkpoint.jsonl so re-runs are free.
  • Auto-resume from .PAUSED, or explicit --resume <run_id>.
  • Pre-flight validation before the first $ is spent.
  • Classify known failures (AFC cap, transient retries exhausted, scanner data
    errors, ConsensusFailure escalations) into a small taxonomy. Anything
    outside the taxonomy halts.

Usage:
  python eval/kfold_runner.py                       # k=5, auto-discover patients
  python eval/kfold_runner.py --k 10
  python eval/kfold_runner.py --patients eval/patient_ids.txt
  python eval/kfold_runner.py --delay 5             # seconds between patients
  python eval/kfold_runner.py --smoke               # 1-patient preflight only
  python eval/kfold_runner.py --resume 20260527_140000   # continue a paused run
"""
from __future__ import annotations

if __name__ == "__main__":
    import os
    import sys
    import subprocess
    from pathlib import Path

    # Force UTF-8 in the child process so κ in logs/prints doesn't crash on Windows cp1252.
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")

    venv_dir = Path(__file__).parent.parent.resolve() / "venv"
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

import argparse
import csv
import gzip
import json
import os
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

# Force UTF-8 on Windows BEFORE src.* imports so the logger's StreamHandler
# (created on first import of src.logger) captures the reconfigured stdout/stderr.
# Without this, any "κ" in log/print output crashes the run on cp1252 consoles.
if os.name == "nt":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from src.agents import diagnostician, auditor, manager
from src.consensus_gate import run_consensus_gate, ConsensusFailure
from src.dashboard_sync import refresh_patients_json
from src.grounding_logger import read_grounding_results_for_patient
from src.logger import get_logger
from src.reasoning_log import log_consensus, log_disagreement
from src.tools import EHRPatternScanner

logger = get_logger(__name__)

MIMIC_DIAGNOSES = "data/mimic-iv-clinical-database-demo-2.2/hosp/diagnoses_icd.csv.gz"
RESULTS_DIR = Path("eval/results")


# ─────────────────────────────────────────────────────────────────────────────
# Error taxonomy
# ─────────────────────────────────────────────────────────────────────────────

_AFC_TAGS = ("AFC", "max_remote_calls", "Automatic function calling")
_TRANSIENT_TAGS = ("429", "RESOURCE_EXHAUSTED", "rate limit", "quota",
                   "503", "UNAVAILABLE", "overloaded")

# Marker emitted by consensus_gate._loop_exhaustion_stub. A patient whose
# diagnostician or auditor output carries this prefix means CrewAI's ReAct loop
# exhausted max_iter — the patient still completes (as ESCALATED with κ≈0) so
# the fold continues, but accumulation of these is a systemic signal that we
# halt on (same N=3 strikes policy as AFC/transient).
_LOOP_EXHAUSTION_MARKER = "[AGENT LOOP EXHAUSTION"


def classify_exception(exc: BaseException) -> str:
    """Return one of: 'afc', 'transient', 'unknown'.

    'afc' and 'transient' are recorded against the patient but halt the run
    (per user policy: pause-on-error). 'unknown' also halts but with a full
    traceback so root cause can be investigated.
    """
    s = str(exc)
    if any(t in s for t in _AFC_TAGS):
        return "afc"
    if any(t in s for t in _TRANSIENT_TAGS):
        return "transient"
    return "unknown"


def _has_loop_exhaustion_stub(*outputs) -> bool:
    """True if any agent output is a recovery stub from consensus_gate."""
    for o in outputs:
        if o is None:
            continue
        hyp = getattr(o, "diagnosis_hypothesis", "") or ""
        if _LOOP_EXHAUSTION_MARKER in hyp:
            return True
    return False


class HaltRun(Exception):
    """Raised to halt the entire run. Carries the offending patient + reason."""

    def __init__(self, patient_id: str, reason: str, category: str, original: BaseException | None = None):
        self.patient_id = patient_id
        self.reason = reason
        self.category = category
        self.original = original
        super().__init__(f"HALT at patient {patient_id}: [{category}] {reason}")


# ─────────────────────────────────────────────────────────────────────────────
# Discovery + splitting
# ─────────────────────────────────────────────────────────────────────────────

def discover_patient_ids(min_codes: int = 3) -> list[str]:
    from collections import defaultdict
    patient_codes: dict[str, set] = defaultdict(set)
    with gzip.open(MIMIC_DIAGNOSES, "rt") as f:
        for row in csv.DictReader(f):
            patient_codes[row["subject_id"]].add(row["icd_code"])
    return sorted(pid for pid, codes in patient_codes.items() if len(codes) >= min_codes)


def kfold_split(items: list, k: int) -> list[list]:
    return [items[i::k] for i in range(k)]


def grounding_accuracy(grounding_table: list[dict], total_codes: int) -> float | None:
    if not total_codes:
        return None
    verified = sum(1 for e in grounding_table if e["stage"] is not None)
    return verified / total_codes


# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight
# ─────────────────────────────────────────────────────────────────────────────

def preflight() -> None:
    """Validate environment before spending money. Raises HaltRun on any failure."""
    problems: list[str] = []

    if not os.path.exists(MIMIC_DIAGNOSES):
        problems.append(f"MIMIC diagnoses file missing: {MIMIC_DIAGNOSES}")

    if os.getenv("USE_LOCAL_LLM", "false").lower() != "true":
        if not os.getenv("GEMINI_API_KEY"):
            problems.append("GEMINI_API_KEY not set (and USE_LOCAL_LLM != true)")

    for name, obj in (("diagnostician", diagnostician), ("auditor", auditor), ("manager", manager)):
        if obj is None:
            problems.append(f"Agent {name} failed to import")

    if problems:
        raise HaltRun(
            patient_id="<preflight>",
            reason="; ".join(problems),
            category="preflight",
        )

    logger.info("[preflight] All checks passed")


# ─────────────────────────────────────────────────────────────────────────────
# Per-patient runner
# ─────────────────────────────────────────────────────────────────────────────

def run_patient(patient_id: str, scanner: EHRPatternScanner) -> dict:
    """Run one patient. Returns a result dict.

    Per the policy in feedback_kfold_errors.md:
      • Success → normal result dict.
      • ConsensusFailure → result with escalated=True.
      • AFC cap / transient retries exhausted → result with error + error_category
        ('afc' or 'transient'). Caller increments a per-fold counter; ≥3 halts.
      • Unknown exception / data error → raise HaltRun (halt the whole run).
    """
    run_start = datetime.now(timezone.utc)

    raw = scanner.run(patient_id)
    try:
        traj = json.loads(raw)
    except Exception as exc:
        raise HaltRun(patient_id, f"Scanner returned non-JSON: {exc}", "data") from exc

    if "error" in traj:
        # Data-level issue (patient missing, file missing). Halt — these should
        # have been caught in preflight or by the patient-list filter.
        raise HaltRun(patient_id, f"Scanner error: {traj['error']}", "data")

    total_codes = len(traj.get("icd_codes", []))

    loop_exhaustion = False
    try:
        d_out, a_out, kappa, iterations = run_consensus_gate(
            patient_id=patient_id,
            trajectory_json=raw,
            diagnostician_agent=diagnostician,
            auditor_agent=auditor,
            manager_agent=manager,
        )
        escalated = False
        loop_exhaustion = _has_loop_exhaustion_stub(d_out, a_out)
        grounding_table = read_grounding_results_for_patient(patient_id, since=run_start)
        log_consensus(
            patient_id=patient_id,
            kappa_score=kappa,
            iterations=iterations,
            diagnostician_output=d_out,
            auditor_output=a_out,
            grounding_table=grounding_table,
        )
    except ConsensusFailure as cf:
        kappa = cf.final_kappa
        iterations = cf.iterations
        escalated = True
        loop_exhaustion = _has_loop_exhaustion_stub(cf.diagnostician_output, cf.auditor_output)
        log_disagreement(
            patient_id=cf.patient_id,
            final_kappa=cf.final_kappa,
            iterations=cf.iterations,
            contradiction_points=[f"κ={cf.final_kappa:.3f} after {cf.iterations} iterations"],
            diagnostician_final=cf.diagnostician_output,
            auditor_final=cf.auditor_output,
        )
    except Exception as exc:
        category = classify_exception(exc)
        if category == "unknown":
            raise HaltRun(
                patient_id,
                f"{type(exc).__name__}: {str(exc)[:200]}",
                category,
                original=exc,
            ) from exc
        # afc / transient — record and continue. Per-fold 3-strikes is enforced by caller.
        logger.warning(
            "[run_patient] patient %s failed (%s): %s",
            patient_id, category, str(exc)[:200],
        )
        return {
            "patient_id": patient_id,
            "error": str(exc)[:500],
            "error_category": category,
            "error_type": type(exc).__name__,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

    if escalated:
        # Escalated patients didn't reach the grounding read inside the try block.
        grounding_table = read_grounding_results_for_patient(patient_id, since=run_start)
    ga = grounding_accuracy(grounding_table, total_codes)
    grounded = sum(1 for e in grounding_table if e["stage"] is not None)
    stage_counts = {str(s): 0 for s in (0, 1, 2, 3)}
    stage_counts["failed"] = 0
    for e in grounding_table:
        key = str(e["stage"]) if e["stage"] is not None else "failed"
        stage_counts[key] = stage_counts.get(key, 0) + 1

    return {
        "patient_id": patient_id,
        "kappa": kappa,
        "iterations": iterations,
        "escalated": escalated,
        "loop_exhaustion": loop_exhaustion,
        "grounding_accuracy": ga,
        "total_trajectory_codes": total_codes,
        "grounded_codes": grounded,
        "stage_breakdown": stage_counts,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint / resume
# ─────────────────────────────────────────────────────────────────────────────

def checkpoint_path(run_id: str) -> Path:
    return RESULTS_DIR / f"kfold_{run_id}.checkpoint.jsonl"


def paused_sentinel_path(run_id: str) -> Path:
    return RESULTS_DIR / f"kfold_{run_id}.PAUSED"


def manifest_path(run_id: str) -> Path:
    return RESULTS_DIR / f"kfold_{run_id}.manifest.json"


def load_completed(run_id: str) -> dict[str, dict]:
    """Read the checkpoint file into {patient_id: result}."""
    path = checkpoint_path(run_id)
    if not path.exists():
        return {}
    done: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            done[rec["patient_id"]] = rec
    return done


def append_checkpoint(run_id: str, record: dict) -> None:
    path = checkpoint_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def write_pause_sentinel(run_id: str, halt: HaltRun, fold_idx: int, patient_idx: int) -> Path:
    path = paused_sentinel_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "paused_at": datetime.now(timezone.utc).isoformat(),
        "halted_patient": halt.patient_id,
        "category": halt.category,
        "reason": halt.reason,
        "fold_idx": fold_idx,
        "patient_idx_in_fold": patient_idx,
        "traceback": traceback.format_exception(halt.original) if halt.original else None,
        "next_step": (
            f"Investigate the error above. When ready, resume with:\n"
            f"  python eval/kfold_runner.py --resume {run_id}"
        ),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def write_manifest(run_id: str, args, patient_ids: list[str], folds: list[list[str]]) -> None:
    path = manifest_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "k": args.k,
            "seed": args.seed,
            "min_codes": args.min_codes,
            "patients": patient_ids,
            "folds": folds,
        }, f, indent=2)


def load_manifest(run_id: str) -> dict:
    path = manifest_path(run_id)
    if not path.exists():
        raise HaltRun("<resume>", f"Manifest not found for run_id={run_id}", "resume")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def summarise_fold(fold_idx: int, patient_results: list[dict]) -> dict:
    valid = [r for r in patient_results if "error" not in r]
    kappas = [r["kappa"] for r in valid]
    gas = [r["grounding_accuracy"] for r in valid if r["grounding_accuracy"] is not None]
    n = len(valid)

    mean_k = sum(kappas) / n if n else None
    std_k = (sum((k - mean_k) ** 2 for k in kappas) / n) ** 0.5 if n > 1 else 0.0
    mean_ga = sum(gas) / len(gas) if gas else None

    return {
        "fold": fold_idx,
        "n_patients": len(patient_results),
        "n_valid": n,
        "n_errors": len(patient_results) - n,
        "mean_kappa": mean_k,
        "std_kappa": std_k,
        "min_kappa": min(kappas) if kappas else None,
        "max_kappa": max(kappas) if kappas else None,
        "kappa_above_threshold": sum(1 for k in kappas if k > 0.80) / n if n else None,
        "escalation_rate": sum(1 for r in valid if r["escalated"]) / n if n else None,
        "mean_iterations": sum(r["iterations"] for r in valid) / n if n else None,
        "mean_grounding_accuracy": mean_ga,
        "patients": patient_results,
    }


def print_fold_line(summary: dict, k: int) -> None:
    ga_str = f"{summary['mean_grounding_accuracy']:.2%}" if summary["mean_grounding_accuracy"] is not None else "n/a"
    mean_k = summary["mean_kappa"]
    mean_k_str = f"{mean_k:.3f}" if mean_k is not None else "n/a"
    kappa_above = summary["kappa_above_threshold"]
    kappa_above_str = f"{kappa_above:.0%}" if kappa_above is not None else "n/a"
    esc = summary["escalation_rate"]
    esc_str = f"{esc:.0%}" if esc is not None else "n/a"
    print(
        f"  Fold {summary['fold'] + 1}/{k}: "
        f"mean_κ={mean_k_str}  std={summary['std_kappa']:.3f}  "
        f"κ>0.80={kappa_above_str}  escalated={esc_str}  GA={ga_str}  "
        f"({summary['n_valid']}/{summary['n_patients']} valid)"
    )


def loud_halt_banner(halt: HaltRun, sentinel: Path) -> None:
    bar = "━" * 78
    print(f"\n{bar}", file=sys.stderr)
    print(f"  RUN HALTED — category: {halt.category.upper()}", file=sys.stderr)
    print(f"  patient: {halt.patient_id}", file=sys.stderr)
    print(f"  reason:  {halt.reason}", file=sys.stderr)
    print(f"  paused sentinel: {sentinel}", file=sys.stderr)
    print(f"  resume with:  python eval/kfold_runner.py --resume <run_id from sentinel>",
          file=sys.stderr)
    print(f"{bar}\n", file=sys.stderr)
    if halt.original is not None:
        print("Original traceback:", file=sys.stderr)
        traceback.print_exception(halt.original, file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="K-fold evaluation for the Multi-Agent Clinical Auditor")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-codes", type=int, default=3)
    parser.add_argument("--patients", help="text file with one subject_id per line")
    parser.add_argument("--delay", type=float, default=0.0, help="seconds between patients")
    parser.add_argument("--smoke", action="store_true",
                        help="run preflight + one smallest-trajectory patient, then exit")
    parser.add_argument("--resume", help="run_id of a previously paused run to continue")
    parser.add_argument("--no-auto-resume", action="store_true",
                        help="disable auto-detection of a single .PAUSED sentinel")
    parser.add_argument("--afc-strikes", type=int, default=3,
                        help="halt the fold after N AFC/transient failures in a single fold (default: 3)")
    args = parser.parse_args()

    try:
        preflight()
    except HaltRun as h:
        loud_halt_banner(h, RESULTS_DIR / "preflight.FAILED")
        return 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Auto-resume: detect .PAUSED sentinels if --resume not given ─────
    if not args.resume and not args.no_auto_resume and not args.smoke:
        sentinels = sorted(RESULTS_DIR.glob("kfold_*.PAUSED"))
        if len(sentinels) == 1:
            run_id_from_sentinel = sentinels[0].name.replace("kfold_", "").replace(".PAUSED", "")
            print(f"[auto-resume] Found paused run: {run_id_from_sentinel}")
            print(f"             ({sentinels[0]})")
            args.resume = run_id_from_sentinel
        elif len(sentinels) > 1:
            print("[auto-resume] Multiple paused runs found — pass --resume <run_id> explicitly:",
                  file=sys.stderr)
            for s in sentinels:
                print(f"  {s.name}", file=sys.stderr)
            return 1

    # ── Determine run_id, patient list, folds (resume vs new) ─────────────
    if args.resume:
        run_id = args.resume
        manifest = load_manifest(run_id)
        patient_ids = manifest["patients"]
        folds = manifest["folds"]
        k = manifest["k"]
        seed = manifest["seed"]
        sentinel = paused_sentinel_path(run_id)
        if sentinel.exists():
            sentinel.unlink()
            logger.info("[resume] Cleared PAUSED sentinel for %s", run_id)
        completed = load_completed(run_id)
        logger.info("[resume] %s — %d/%d patients already completed",
                    run_id, len(completed), len(patient_ids))
    else:
        if args.patients:
            with open(args.patients) as f:
                patient_ids = [line.strip() for line in f if line.strip()]
        else:
            logger.info("Auto-discovering patient IDs (min_codes=%d)...", args.min_codes)
            patient_ids = discover_patient_ids(min_codes=args.min_codes)
        logger.info("Loaded %d patient IDs", len(patient_ids))

        random.seed(args.seed)
        shuffled = list(patient_ids)
        random.shuffle(shuffled)
        patient_ids = shuffled

        if args.smoke:
            # Pick the patient with the FEWEST codes for cheap, fast preflight.
            from collections import defaultdict
            counts: dict[str, int] = defaultdict(int)
            with gzip.open(MIMIC_DIAGNOSES, "rt") as f:
                for row in csv.DictReader(f):
                    counts[row["subject_id"]] += 1
            patient_ids = [min(patient_ids, key=lambda p: counts.get(p, 0))]
            folds = [patient_ids]
            k = 1
        else:
            folds = kfold_split(patient_ids, args.k)
            k = args.k

        seed = args.seed
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        # store an args-shim object for manifest writing
        class _A: pass
        a = _A(); a.k = k; a.seed = seed; a.min_codes = args.min_codes
        write_manifest(run_id, a, patient_ids, folds)
        completed = {}

    print(f"\nK-fold evaluation  run_id={run_id}  k={k}  n={len(patient_ids)}  seed={seed}")
    print(f"Checkpoint:        {checkpoint_path(run_id)}")
    if completed:
        print(f"Resuming from:     {len(completed)} already-completed patients")
    print()

    scanner = EHRPatternScanner()
    eval_start = datetime.now(timezone.utc).isoformat()

    # ── Main loop: halt on unknown errors, 3-strikes on afc/transient per fold ──
    try:
        for fold_idx, fold_patients in enumerate(folds):
            logger.info("[Fold %d/%d] %d patients", fold_idx + 1, k, len(fold_patients))
            fold_afc_strikes = 0          # AFC + transient failures (raised as exceptions)
            fold_loop_strikes = 0         # CrewAI max_iter exhaustion (recovered via stub)

            for i, pid in enumerate(fold_patients):
                if pid in completed:
                    logger.info("  [%d/%d] patient %s — SKIP (checkpointed)",
                                i + 1, len(fold_patients), pid)
                    continue

                logger.info("  [%d/%d] patient %s", i + 1, len(fold_patients), pid)
                try:
                    result = run_patient(pid, scanner)
                except HaltRun as halt:
                    sentinel = write_pause_sentinel(run_id, halt, fold_idx, i)
                    loud_halt_banner(halt, sentinel)
                    return 1

                append_checkpoint(run_id, result)
                completed[pid] = result

                if "error" not in result:
                    try:
                        refresh_patients_json()
                    except Exception as exc:
                        logger.warning("[dashboard] refresh_patients_json failed: %s", exc)

                if "error" in result:
                    fold_afc_strikes += 1
                    logger.warning(
                        "    FAILED (%s) — fold strike %d/%d",
                        result.get("error_category", "?"),
                        fold_afc_strikes, args.afc_strikes,
                    )
                    if fold_afc_strikes >= args.afc_strikes:
                        halt = HaltRun(
                            pid,
                            f"{fold_afc_strikes} afc/transient failures in fold {fold_idx + 1} — systemic issue",
                            "systemic",
                        )
                        sentinel = write_pause_sentinel(run_id, halt, fold_idx, i)
                        loud_halt_banner(halt, sentinel)
                        return 1
                else:
                    status = "ESCALATED" if result["escalated"] else f"κ={result['kappa']:.3f}"
                    ga_str = f"{result['grounding_accuracy']:.2%}" if result["grounding_accuracy"] is not None else "n/a"
                    logger.info("    %s  iter=%d  GA=%s", status, result["iterations"], ga_str)

                    if result.get("loop_exhaustion"):
                        fold_loop_strikes += 1
                        logger.warning(
                            "    LOOP EXHAUSTION recovered via stub — fold strike %d/%d",
                            fold_loop_strikes, args.afc_strikes,
                        )
                        if fold_loop_strikes >= args.afc_strikes:
                            halt = HaltRun(
                                pid,
                                f"{fold_loop_strikes} CrewAI max_iter loop-exhaustion stubs in "
                                f"fold {fold_idx + 1} — systemic LLM-response issue (was previously "
                                f"a hard crash; consensus_gate now recovers, but accumulation here "
                                f"means the model is broken upstream)",
                                "systemic-loop",
                            )
                            sentinel = write_pause_sentinel(run_id, halt, fold_idx, i)
                            loud_halt_banner(halt, sentinel)
                            return 1

                if args.delay > 0 and i < len(fold_patients) - 1:
                    time.sleep(args.delay)

    except KeyboardInterrupt:
        halt = HaltRun("<user>", "KeyboardInterrupt — user halted run", "user")
        sentinel = write_pause_sentinel(run_id, halt, -1, -1)
        loud_halt_banner(halt, sentinel)
        return 130

    # ── Aggregation (only reached if every patient succeeded) ──────────
    all_fold_summaries = []
    for fold_idx, fold_patients in enumerate(folds):
        fold_results = [completed[pid] for pid in fold_patients if pid in completed]
        summary = summarise_fold(fold_idx, fold_results)
        all_fold_summaries.append(summary)
        print_fold_line(summary, k)

    all_valid = [p for s in all_fold_summaries for p in s["patients"] if "error" not in p]
    all_errors = [p for s in all_fold_summaries for p in s["patients"] if "error" in p]
    all_kappas = [r["kappa"] for r in all_valid]
    all_gas = [r["grounding_accuracy"] for r in all_valid if r["grounding_accuracy"] is not None]
    n_total = len(all_valid)
    mean_k_overall = sum(all_kappas) / n_total if n_total else None
    std_k_overall = (sum((kp - mean_k_overall) ** 2 for kp in all_kappas) / n_total) ** 0.5 if n_total > 1 else 0.0

    overall = {
        "run_id": run_id,
        "eval_start": eval_start,
        "eval_end": datetime.now(timezone.utc).isoformat(),
        "k": k,
        "seed": seed,
        "n_patients_total": len(patient_ids),
        "n_patients_valid": n_total,
        "n_patients_errored": len(all_errors),
        "errored_patients": [
            {"patient_id": p["patient_id"], "category": p.get("error_category"), "error": p["error"]}
            for p in all_errors
        ],
        "mean_kappa": mean_k_overall,
        "std_kappa": std_k_overall,
        "min_kappa": min(all_kappas) if all_kappas else None,
        "max_kappa": max(all_kappas) if all_kappas else None,
        "kappa_above_threshold": sum(1 for kp in all_kappas if kp > 0.80) / n_total if n_total else None,
        "escalation_rate": sum(1 for r in all_valid if r["escalated"]) / n_total if n_total else None,
        "mean_iterations": sum(r["iterations"] for r in all_valid) / n_total if n_total else None,
        "mean_grounding_accuracy": sum(all_gas) / len(all_gas) if all_gas else None,
        "folds": all_fold_summaries,
    }

    out_path = RESULTS_DIR / f"kfold_{run_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    ga_str = f"{overall['mean_grounding_accuracy']:.2%}" if overall["mean_grounding_accuracy"] is not None else "n/a"
    print(f"\n{'=' * 60}")
    print(f"OVERALL  {k}-fold  n={n_total} patients")
    if mean_k_overall is not None:
        print(f"  Mean κ:           {overall['mean_kappa']:.4f} ± {overall['std_kappa']:.4f}")
        print(f"  Min / Max κ:      {overall['min_kappa']:.3f} / {overall['max_kappa']:.3f}")
        print(f"  κ > 0.80 rate:    {overall['kappa_above_threshold']:.1%}")
        print(f"  Escalation rate:  {overall['escalation_rate']:.1%}")
        print(f"  Mean iterations:  {overall['mean_iterations']:.2f}")
    print(f"  Mean GA:          {ga_str}")
    print(f"\nResults → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
