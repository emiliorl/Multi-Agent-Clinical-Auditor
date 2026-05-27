"""K-fold cross-validation runner for the Multi-Agent Clinical Auditor.

Splits all eligible MIMIC-IV patients into k folds, runs each patient
through the full consensus pipeline, and reports:
  - Per-patient: kappa, iterations, escalated, grounding_accuracy
  - Per-fold + overall: mean/std kappa, kappa>0.80 rate, escalation rate,
    mean iterations, mean grounding accuracy (GA)

Results are written to eval/results/kfold_<timestamp>.json.

Usage:
  python eval/kfold_runner.py                  # k=5, all 100 patients
  python eval/kfold_runner.py --k 10
  python eval/kfold_runner.py --patients eval/patient_ids.txt
  python eval/kfold_runner.py --delay 5        # seconds between patients
"""
from __future__ import annotations

if __name__ == "__main__":
    import os
    import sys
    import subprocess
    from pathlib import Path

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
from datetime import datetime, timezone
from pathlib import Path

os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

# Make project root importable regardless of cwd
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from src.agents import diagnostician, auditor, manager
from src.consensus_gate import run_consensus_gate, ConsensusFailure
from src.grounding_logger import read_grounding_results_for_patient
from src.logger import get_logger
from src.tools import EHRPatternScanner

logger = get_logger(__name__)

MIMIC_DIAGNOSES = "data/mimic-iv-clinical-database-demo-2.2/hosp/diagnoses_icd.csv.gz"
RESULTS_DIR = Path("eval/results")


def discover_patient_ids(min_codes: int = 3) -> list[str]:
    from collections import defaultdict
    patient_codes: dict[str, set] = defaultdict(set)
    with gzip.open(MIMIC_DIAGNOSES, "rt") as f:
        for row in csv.DictReader(f):
            patient_codes[row["subject_id"]].add(row["icd_code"])
    return sorted(pid for pid, codes in patient_codes.items() if len(codes) >= min_codes)


def kfold_split(items: list, k: int) -> list[list]:
    """Round-robin fold assignment — each fold gets roughly len(items)/k patients."""
    return [items[i::k] for i in range(k)]


def grounding_accuracy(grounding_table: list[dict], total_codes: int) -> float | None:
    if not total_codes:
        return None
    verified = sum(1 for e in grounding_table if e["stage"] is not None)
    return verified / total_codes


def run_patient(patient_id: str, scanner: EHRPatternScanner) -> dict:
    run_start = datetime.now(timezone.utc)

    raw = scanner.run(patient_id)
    try:
        traj = json.loads(raw)
        if "error" in traj:
            return {"patient_id": patient_id, "error": traj["error"]}
    except Exception as exc:
        return {"patient_id": patient_id, "error": str(exc)}

    total_codes = len(traj.get("icd_codes", []))

    try:
        d_out, a_out, kappa, iterations = run_consensus_gate(
            patient_id=patient_id,
            trajectory_json=raw,
            diagnostician_agent=diagnostician,
            auditor_agent=auditor,
            manager_agent=manager,
        )
        escalated = False
    except ConsensusFailure as cf:
        kappa = cf.final_kappa
        iterations = cf.iterations
        escalated = True

    grounding_table = read_grounding_results_for_patient(patient_id, since=run_start)
    ga = grounding_accuracy(grounding_table, total_codes)
    grounded = sum(1 for e in grounding_table if e["stage"] is not None)
    # Break down by stage for transparency
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
        "grounding_accuracy": ga,
        "total_trajectory_codes": total_codes,
        "grounded_codes": grounded,
        "stage_breakdown": stage_counts,
    }


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
    print(
        f"  Fold {summary['fold'] + 1}/{k}: "
        f"mean_κ={summary['mean_kappa']:.3f}  "
        f"std={summary['std_kappa']:.3f}  "
        f"κ>0.80={summary['kappa_above_threshold']:.0%}  "
        f"escalated={summary['escalation_rate']:.0%}  "
        f"GA={ga_str}  "
        f"({summary['n_valid']}/{summary['n_patients']} valid)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="K-fold evaluation for the Multi-Agent Clinical Auditor")
    parser.add_argument("--k", type=int, default=5, help="Number of folds (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for fold assignment (default: 42)")
    parser.add_argument("--min-codes", type=int, default=3, help="Min ICD codes required for a patient to be included (default: 3)")
    parser.add_argument("--patients", help="Path to a text file with one subject_id per line (optional; auto-discovers if omitted)")
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds to wait between patients — use to avoid rate limits (default: 0)")
    args = parser.parse_args()

    if args.patients:
        with open(args.patients) as f:
            patient_ids = [line.strip() for line in f if line.strip()]
        logger.info("Loaded %d patient IDs from %s", len(patient_ids), args.patients)
    else:
        logger.info("Auto-discovering patient IDs from MIMIC data (min_codes=%d)...", args.min_codes)
        patient_ids = discover_patient_ids(min_codes=args.min_codes)
        logger.info("Found %d eligible patients", len(patient_ids))

    random.seed(args.seed)
    shuffled = list(patient_ids)
    random.shuffle(shuffled)
    folds = kfold_split(shuffled, args.k)

    print(f"\nK-fold evaluation  k={args.k}  n={len(shuffled)}  seed={args.seed}")
    print(f"Patients per fold: ~{len(folds[0])}")
    print()

    scanner = EHRPatternScanner()
    eval_start = datetime.now(timezone.utc).isoformat()
    all_fold_summaries: list[dict] = []

    for fold_idx, fold_patients in enumerate(folds):
        logger.info("[Fold %d/%d] Starting — %d patients", fold_idx + 1, args.k, len(fold_patients))
        patient_results: list[dict] = []

        for i, pid in enumerate(fold_patients):
            logger.info("  [%d/%d] patient %s", i + 1, len(fold_patients), pid)
            result = run_patient(pid, scanner)
            patient_results.append(result)

            if "error" in result:
                logger.warning("    ERROR: %s", result["error"])
            else:
                status = "ESCALATED" if result["escalated"] else f"κ={result['kappa']:.3f}"
                ga_str = f"{result['grounding_accuracy']:.2%}" if result["grounding_accuracy"] is not None else "n/a"
                logger.info("    %s  iter=%d  GA=%s", status, result["iterations"], ga_str)

            if args.delay > 0 and i < len(fold_patients) - 1:
                time.sleep(args.delay)

        summary = summarise_fold(fold_idx, patient_results)
        all_fold_summaries.append(summary)
        print_fold_line(summary, args.k)

    # Overall stats across all patients
    all_valid = [p for fold in all_fold_summaries for p in fold["patients"] if "error" not in p]
    all_kappas = [r["kappa"] for r in all_valid]
    all_gas = [r["grounding_accuracy"] for r in all_valid if r["grounding_accuracy"] is not None]
    n_total = len(all_valid)

    mean_k_overall = sum(all_kappas) / n_total if n_total else None
    std_k_overall = (sum((k - mean_k_overall) ** 2 for k in all_kappas) / n_total) ** 0.5 if n_total > 1 else 0.0

    overall = {
        "eval_start": eval_start,
        "eval_end": datetime.now(timezone.utc).isoformat(),
        "k": args.k,
        "seed": args.seed,
        "n_patients_total": len(shuffled),
        "n_patients_valid": n_total,
        "n_patients_errored": len(shuffled) - n_total,
        "mean_kappa": mean_k_overall,
        "std_kappa": std_k_overall,
        "min_kappa": min(all_kappas) if all_kappas else None,
        "max_kappa": max(all_kappas) if all_kappas else None,
        "kappa_above_threshold": sum(1 for k in all_kappas if k > 0.80) / n_total if n_total else None,
        "escalation_rate": sum(1 for r in all_valid if r["escalated"]) / n_total if n_total else None,
        "mean_iterations": sum(r["iterations"] for r in all_valid) / n_total if n_total else None,
        "mean_grounding_accuracy": sum(all_gas) / len(all_gas) if all_gas else None,
        "folds": all_fold_summaries,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"kfold_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    ga_str = f"{overall['mean_grounding_accuracy']:.2%}" if overall["mean_grounding_accuracy"] is not None else "n/a"
    print(f"\n{'=' * 60}")
    print(f"OVERALL  {args.k}-fold  n={n_total} patients")
    print(f"  Mean κ:           {overall['mean_kappa']:.4f} ± {overall['std_kappa']:.4f}")
    print(f"  Min / Max κ:      {overall['min_kappa']:.3f} / {overall['max_kappa']:.3f}")
    print(f"  κ > 0.80 rate:    {overall['kappa_above_threshold']:.1%}")
    print(f"  Escalation rate:  {overall['escalation_rate']:.1%}")
    print(f"  Mean iterations:  {overall['mean_iterations']:.2f}")
    print(f"  Mean GA:          {ga_str}")
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
