"""Retry errored patients from a completed k-fold run and patch results in-place.

Usage:
  python eval/retry_errored.py                          # auto-detects the latest .json run
  python eval/retry_errored.py --run 20260527_140636    # target a specific run
"""
from __future__ import annotations

if __name__ == "__main__":
    import os, sys, subprocess
    from pathlib import Path
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    venv_dir = Path(__file__).parent.parent.resolve() / "venv"
    if venv_dir.exists():
        try:
            is_in_venv = Path(sys.executable).resolve().is_relative_to(venv_dir)
        except AttributeError:
            try: Path(sys.executable).resolve().relative_to(venv_dir); is_in_venv = True
            except ValueError: is_in_venv = False
        if not is_in_venv:
            venv_python = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            if venv_python.exists():
                sys.exit(subprocess.call([str(venv_python)] + sys.argv))

import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path

os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

if os.name == "nt":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from src.agents import diagnostician, auditor, manager
from src.consensus_gate import run_consensus_gate, ConsensusFailure
from src.grounding_logger import read_grounding_results_for_patient
from src.logger import get_logger
from src.reasoning_log import log_consensus, log_disagreement
from src.tools import EHRPatternScanner

logger = get_logger(__name__)

RESULTS_DIR = Path("eval/results")


def find_latest_run() -> str:
    jsons = sorted(p for p in RESULTS_DIR.glob("kfold_*.json") if not p.name.endswith(".manifest.json"))
    if not jsons:
        raise SystemExit("No completed run JSON found in eval/results/")
    return jsons[-1].stem.replace("kfold_", "")


def grounding_accuracy(grounding_table, total_codes):
    if not total_codes:
        return None
    verified = sum(1 for e in grounding_table if e["stage"] is not None)
    return verified / total_codes


def run_one_patient(patient_id: str, scanner: EHRPatternScanner) -> dict:
    run_start = datetime.now(timezone.utc)
    raw = scanner.run(patient_id)
    traj = json.loads(raw)
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
        # check for recovery stubs
        for o in (d_out, a_out):
            if o and "[AGENT LOOP EXHAUSTION" in (getattr(o, "diagnosis_hypothesis", "") or ""):
                loop_exhaustion = True
        grounding_table = read_grounding_results_for_patient(patient_id, since=run_start)
        log_consensus(patient_id=patient_id, kappa_score=kappa, iterations=iterations,
                      diagnostician_output=d_out, auditor_output=a_out, grounding_table=grounding_table)
    except ConsensusFailure as cf:
        kappa = cf.final_kappa
        iterations = cf.iterations
        escalated = True
        for o in (cf.diagnostician_output, cf.auditor_output):
            if o and "[AGENT LOOP EXHAUSTION" in (getattr(o, "diagnosis_hypothesis", "") or ""):
                loop_exhaustion = True
        grounding_table = read_grounding_results_for_patient(patient_id, since=run_start)
        log_disagreement(patient_id=cf.patient_id, final_kappa=cf.final_kappa,
                         iterations=cf.iterations,
                         contradiction_points=[f"k={cf.final_kappa:.3f} after {cf.iterations} iterations"],
                         diagnostician_final=cf.diagnostician_output, auditor_final=cf.auditor_output)

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
        "retried_at": datetime.now(timezone.utc).isoformat(),
    }


def patch_results(run_id: str, new_results: dict[str, dict]) -> None:
    """Replace error records in the checkpoint and final JSON with successful retries."""
    # 1. Patch checkpoint
    ckpt = RESULTS_DIR / f"kfold_{run_id}.checkpoint.jsonl"
    lines = []
    patched_in_ckpt = set()
    with open(ckpt, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pid = rec["patient_id"]
            if pid in new_results:
                lines.append(json.dumps(new_results[pid]))
                patched_in_ckpt.add(pid)
            else:
                lines.append(line)
    with open(ckpt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Patched checkpoint: %s patients updated", len(patched_in_ckpt))

    # 2. Regenerate final JSON
    final_path = RESULTS_DIR / f"kfold_{run_id}.json"
    with open(final_path, encoding="utf-8") as f:
        final = json.load(f)

    # replace patient records in each fold
    for fold in final["folds"]:
        for i, p in enumerate(fold["patients"]):
            if p["patient_id"] in new_results:
                fold["patients"][i] = new_results[p["patient_id"]]

    # recompute fold summaries
    for fold in final["folds"]:
        valid = [p for p in fold["patients"] if "error" not in p]
        kappas = [p["kappa"] for p in valid]
        gas = [p["grounding_accuracy"] for p in valid if p.get("grounding_accuracy") is not None]
        n = len(valid)
        fold["n_valid"] = n
        fold["n_errors"] = fold["n_patients"] - n
        fold["mean_kappa"] = sum(kappas) / n if n else None
        fold["std_kappa"] = (sum((k - fold["mean_kappa"]) ** 2 for k in kappas) / n) ** 0.5 if n > 1 else 0.0
        fold["min_kappa"] = min(kappas) if kappas else None
        fold["max_kappa"] = max(kappas) if kappas else None
        fold["kappa_above_threshold"] = sum(1 for k in kappas if k > 0.80) / n if n else None
        fold["escalation_rate"] = sum(1 for p in valid if p["escalated"]) / n if n else None
        fold["mean_iterations"] = sum(p["iterations"] for p in valid) / n if n else None
        fold["mean_grounding_accuracy"] = sum(gas) / len(gas) if gas else None

    # recompute overall
    all_valid = [p for fold in final["folds"] for p in fold["patients"] if "error" not in p]
    all_errors = [p for fold in final["folds"] for p in fold["patients"] if "error" in p]
    all_kappas = [p["kappa"] for p in all_valid]
    all_gas = [p["grounding_accuracy"] for p in all_valid if p.get("grounding_accuracy") is not None]
    n_total = len(all_valid)
    mk = sum(all_kappas) / n_total if n_total else None

    final["n_patients_valid"] = n_total
    final["n_patients_errored"] = len(all_errors)
    final["errored_patients"] = [
        {"patient_id": p["patient_id"], "category": p.get("error_category"), "error": p["error"]}
        for p in all_errors
    ]
    final["mean_kappa"] = mk
    final["std_kappa"] = (sum((k - mk) ** 2 for k in all_kappas) / n_total) ** 0.5 if n_total > 1 else 0.0
    final["min_kappa"] = min(all_kappas) if all_kappas else None
    final["max_kappa"] = max(all_kappas) if all_kappas else None
    final["kappa_above_threshold"] = sum(1 for k in all_kappas if k > 0.80) / n_total if n_total else None
    final["escalation_rate"] = sum(1 for p in all_valid if p["escalated"]) / n_total if n_total else None
    final["mean_iterations"] = sum(p["iterations"] for p in all_valid) / n_total if n_total else None
    final["mean_grounding_accuracy"] = sum(all_gas) / len(all_gas) if all_gas else None
    final["patched_at"] = datetime.now(timezone.utc).isoformat()

    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    logger.info("Patched final JSON: %s", final_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="run_id to patch (default: latest)")
    args = parser.parse_args()

    run_id = args.run or find_latest_run()
    final_path = RESULTS_DIR / f"kfold_{run_id}.json"
    with open(final_path, encoding="utf-8") as f:
        d = json.load(f)

    errored = d.get("errored_patients", [])
    if not errored:
        print("No errored patients found in run — nothing to do.")
        return 0

    print(f"Run {run_id}: {len(errored)} errored patient(s) to retry:")
    for p in errored:
        print(f"  {p['patient_id']}  [{p['category']}]  {p['error'][:80]}")
    print()

    scanner = EHRPatternScanner()
    new_results: dict[str, dict] = {}

    for i, ep in enumerate(errored):
        pid = ep["patient_id"]
        print(f"[{i+1}/{len(errored)}] Retrying patient {pid} ...")
        try:
            result = run_one_patient(pid, scanner)
            status = "ESCALATED" if result["escalated"] else f"kappa={result['kappa']:.3f}"
            if result.get("loop_exhaustion"):
                status += " (loop-exhaustion stub)"
            print(f"  -> {status}  iter={result['iterations']}")
            new_results[pid] = result
        except Exception as exc:
            print(f"  -> FAILED again: {exc}")
            print("     Patient will remain as errored in the results.")

    if not new_results:
        print("\nNo patients recovered — results unchanged.")
        return 1

    print(f"\nPatching {len(new_results)} patient(s) into run {run_id} ...")
    patch_results(run_id, new_results)
    print("Done. Re-export if needed:")
    print(f"  python eval/retry_errored.py  (already patched in kfold_{run_id}.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
