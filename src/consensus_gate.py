"""Consensus Gate: Reflection Loop and ConsensusFailure exception."""

from __future__ import annotations

import json
from crewai import Crew, Process

from src.consensus import compute_kappa
from src.models import AgentDiagnosis
from src.logger import get_logger

logger = get_logger(__name__)

MAX_ITER = 3
KAPPA_THRESHOLD = 0.80


class ConsensusFailure(Exception):
    def __init__(
        self,
        patient_id: str,
        final_kappa: float,
        iterations: int,
        diagnostician_output=None,
        auditor_output=None,
    ):
        self.patient_id = patient_id
        self.final_kappa = final_kappa
        self.iterations = iterations
        self.diagnostician_output = diagnostician_output
        self.auditor_output = auditor_output
        super().__init__(
            f"Consensus not reached for {patient_id} after {iterations} iterations "
            f"(κ={final_kappa:.3f})"
        )


def _build_contradiction_report(
    d_output: AgentDiagnosis,
    a_output: AgentDiagnosis,
) -> str:
    d_only = set(d_output.icd_codes_cited) - set(a_output.icd_codes_cited)
    a_only = set(a_output.icd_codes_cited) - set(d_output.icd_codes_cited)
    return json.dumps(
        {
            "diagnostician_hypothesis": d_output.diagnosis_hypothesis,
            "auditor_hypothesis": a_output.diagnosis_hypothesis,
            "codes_only_in_diagnostician": sorted(d_only),
            "codes_only_in_auditor": sorted(a_only),
            "diagnostician_confidence": d_output.confidence_score,
            "auditor_confidence": a_output.confidence_score,
        },
        indent=2,
    )


def _run_single_task(agent, task_fn, *args, **kwargs) -> AgentDiagnosis:
    """Run one agent + one task as an isolated single-task crew."""
    task = task_fn(agent, *args, **kwargs)
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )
    crew.kickoff()
    return task.output.pydantic


def run_consensus_gate(
    patient_id: str,
    trajectory_json: str,
    diagnostician_agent,
    auditor_agent,
) -> tuple[AgentDiagnosis, AgentDiagnosis, float, int]:
    """
    Runs the Reflection Loop up to MAX_ITER times.
    Returns (diagnostician_output, auditor_output, final_kappa, iterations_used).
    Raises ConsensusFailure if kappa stays <= KAPPA_THRESHOLD after MAX_ITER.
    """
    from src.tasks import get_mining_task, get_audit_task, get_reflection_task

    logger.info("[ConsensusGate] Running initial diagnostician pass for patient %s", patient_id)
    d_output = _run_single_task(diagnostician_agent, get_mining_task, trajectory_json)

    logger.info("[ConsensusGate] Running initial auditor pass for patient %s", patient_id)
    a_output = _run_single_task(auditor_agent, get_audit_task, trajectory_json)

    for iteration in range(1, MAX_ITER + 1):
        kappa = compute_kappa(d_output.icd_codes_cited, a_output.icd_codes_cited)
        logger.info(
            "[ConsensusGate] Iteration %d/%d — κ=%.3f (threshold=%.2f)",
            iteration, MAX_ITER, kappa, KAPPA_THRESHOLD,
        )

        if kappa > KAPPA_THRESHOLD:
            logger.info("[ConsensusGate] Consensus reached at iteration %d (κ=%.3f)", iteration, kappa)
            return d_output, a_output, kappa, iteration

        if iteration < MAX_ITER:
            contradiction = _build_contradiction_report(d_output, a_output)
            logger.info("[ConsensusGate] κ below threshold — triggering reflection (iteration %d)", iteration)

            d_output = _run_single_task(
                diagnostician_agent, get_reflection_task, trajectory_json, contradiction
            )
            a_output = _run_single_task(
                auditor_agent, get_reflection_task, trajectory_json, contradiction
            )

    final_kappa = compute_kappa(d_output.icd_codes_cited, a_output.icd_codes_cited)
    if final_kappa <= KAPPA_THRESHOLD:
        raise ConsensusFailure(patient_id, final_kappa, MAX_ITER, d_output, a_output)
    return d_output, a_output, final_kappa, MAX_ITER
