import argparse
import json
import time
from datetime import datetime, timezone
from crewai import Crew, Process
from src.agents import diagnostician, auditor
from src.tasks import get_critique_from_outputs_task
from src.tools import EHRPatternScanner
from src.logger import get_logger
from src.consensus_gate import run_consensus_gate, ConsensusFailure
from src.grounding_logger import read_grounding_results_for_patient
from src.reasoning_log import log_consensus, log_disagreement

logger = get_logger(__name__)

_scanner = EHRPatternScanner()


def _get_trajectory_json(patient_id: str) -> str:
    raw = _scanner.run(patient_id)
    try:
        parsed = json.loads(raw)
        if "error" in parsed:
            raise ValueError(parsed["error"])
    except (json.JSONDecodeError, TypeError):
        raise ValueError(f"EHRPatternScanner returned non-JSON output for patient {patient_id}")
    return raw


def _run_with_retry(fn, max_retries: int = 3, retry_delay: int = 25):
    # High-level safety net only. Granular transient-error retries are handled closer
    # to the source: _kickoff_with_retry in consensus_gate.py (per agent step) and
    # _stage3_generate in tools.py (Stage 3 LLM grounding), both via tenacity with
    # exponential backoff. This wrapper catches anything that escapes those layers.
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except ConsensusFailure:
            raise
        except Exception as e:
            err_str = str(e)
            is_transient = any(
                tag in err_str
                for tag in ["429", "RESOURCE_EXHAUSTED", "rate limit", "quota",
                            "503", "UNAVAILABLE", "overloaded"]
            )
            if is_transient and attempt < max_retries:
                logger.warning(
                    "[API] Transient error (attempt %d/%d) — waiting %ds: %s",
                    attempt, max_retries, retry_delay, err_str[:120],
                )
                time.sleep(retry_delay)
            else:
                raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Agent Clinical Auditor")
    parser.add_argument("--patient", required=True, help="MIMIC-IV subject_id")
    args = parser.parse_args()

    patient_id: str = args.patient

    # ── Tier 1: EHR Data Mining ─────────────────────────────────────────────
    logger.info("[Tier 1] Scanning EHR for patient %s", patient_id)
    trajectory_json = _get_trajectory_json(patient_id)
    logger.info("[Tier 1] Trajectory acquired (%d chars)", len(trajectory_json))

    # ── Tiers 2–4: Grounding → Adversarial Agents → Consensus Gate ──────────
    run_start_time = datetime.now(timezone.utc)
    try:
        d_output, a_output, final_kappa, iterations = _run_with_retry(
            lambda: run_consensus_gate(
                patient_id=patient_id,
                trajectory_json=trajectory_json,
                diagnostician_agent=diagnostician,
                auditor_agent=auditor,
            )
        )

        grounding_table = read_grounding_results_for_patient(patient_id, since=run_start_time)
        run_id = log_consensus(
            patient_id=patient_id,
            kappa_score=final_kappa,
            iterations=iterations,
            diagnostician_output=d_output,
            auditor_output=a_output,
            grounding_table=grounding_table,
        )
        logger.info(
            "[ReasoningLog] Consensus logged — run_id=%s, κ=%.3f, iterations=%d",
            run_id, final_kappa, iterations,
        )

    except ConsensusFailure as cf:
        run_id = log_disagreement(
            patient_id=cf.patient_id,
            final_kappa=cf.final_kappa,
            iterations=cf.iterations,
            contradiction_points=[
                f"κ={cf.final_kappa:.3f} after {cf.iterations} iterations"
            ],
            diagnostician_final=cf.diagnostician_output,
            auditor_final=cf.auditor_output,
        )
        logger.warning(
            "[DisagreementLog] Physician escalation logged — run_id=%s, κ=%.3f",
            run_id, cf.final_kappa,
        )
        print(f"\n[ESCALATION] Patient {cf.patient_id} requires physician review. "
              f"κ={cf.final_kappa:.3f} after {cf.iterations} iterations. "
              f"Logged as run_id={run_id}")
        return

    # ── Adversarial critique (post-consensus, isolated single-task crew) ─────
    # Run the critique as a single task with pre-computed outputs embedded in the
    # description. Running [t1, t2, t3] as a sequential crew would cause CrewAI
    # to implicitly pass t1's output to t2, breaking adversarial isolation.
    logger.info("[Critique] Running adversarial critique for patient %s", patient_id)
    critique_task = get_critique_from_outputs_task(auditor, d_output, a_output)
    critique_crew = Crew(
        agents=[auditor],
        tasks=[critique_task],
        process=Process.sequential,
        verbose=True,
    )
    result = _run_with_retry(critique_crew.kickoff)
    logger.info("[Critique] Complete for patient %s (κ=%.3f)", patient_id, final_kappa)
    print(f"\n[RESULT] Patient {patient_id} — κ={final_kappa:.3f}")
    print(result)


if __name__ == "__main__":
    main()
