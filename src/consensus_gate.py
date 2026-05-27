"""Consensus Gate: Reflection Loop and ConsensusFailure exception."""

from __future__ import annotations

import json
import re
from crewai import Crew, Process
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from crewai import Agent
from src.consensus import compute_kappa
from src.models import AgentDiagnosis
from src.logger import get_logger

logger = get_logger(__name__)


def _strip_tools(agent: Agent) -> Agent:
    """Return a tool-free copy of an agent for use in reflection tasks.

    Passing tools=[] on a Task doesn't stop CrewAI from entering
    _invoke_loop_native_tools — that path is chosen by the Agent's tool
    configuration. Creating a new Agent with tools=[] is the only reliable
    way to keep Gemini out of AFC mode during reflection.
    """
    return Agent(
        role=agent.role,
        goal=agent.goal,
        backstory=agent.backstory,
        llm=agent.llm,
        tools=[],
        verbose=agent.verbose,
        allow_delegation=False,
    )

MAX_ITER = 3          # Must match the proposal (Section III, Engineering Constraint)
KAPPA_THRESHOLD = 0.80

_TRANSIENT_TAGS = ("429", "RESOURCE_EXHAUSTED", "rate limit", "quota",
                   "503", "UNAVAILABLE", "overloaded",
                   "Invalid response from LLM call", "None or empty")


def _is_transient(exc: BaseException) -> bool:
    return any(tag in str(exc) for tag in _TRANSIENT_TAGS)


@retry(
    retry=retry_if_exception(_is_transient),
    wait=wait_exponential(multiplier=2, min=30, max=120),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _kickoff_with_retry(crew: Crew) -> None:
    crew.kickoff()


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


def _clean_fences(text: str) -> str:
    return re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()


def _try_parse(text: str) -> AgentDiagnosis | None:
    """Try to parse AgentDiagnosis from text, returning None on any failure."""
    text = _clean_fences(text)
    # find the outermost {...} block
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
        # coerce common type mismatches from local models
        if isinstance(data.get("icd_codes_cited"), str):
            data["icd_codes_cited"] = [c.strip() for c in data["icd_codes_cited"].split(",") if c.strip()]
        if isinstance(data.get("evidence_chain"), str):
            data["evidence_chain"] = [data["evidence_chain"]]
        if isinstance(data.get("unverified_codes"), str):
            data["unverified_codes"] = [c.strip() for c in data["unverified_codes"].split(",") if c.strip()]
        data.setdefault("unverified_codes", [])
        return AgentDiagnosis(**data)
    except Exception:
        return None


def _parse_from_raw(raw: str) -> AgentDiagnosis:
    """
    Multi-strategy extractor for AgentDiagnosis from raw LLM output.
    Local models (Gemma, Qwen) produce ReAct Thought/Action chains before
    the final JSON — each strategy peels back one layer of wrapping.
    """
    # Strategy 1 — raw as-is (works when model outputs clean JSON directly)
    result = _try_parse(raw)
    if result:
        return result

    # Strategy 2 — everything after "Final Answer:" (standard ReAct terminator)
    fa_match = re.search(r'(?:Final Answer:|FINAL ANSWER:)\s*([\s\S]*?)$', raw, re.IGNORECASE)
    if fa_match:
        result = _try_parse(fa_match.group(1))
        if result:
            return result

    # Strategy 3 — strip Thought/Action/Observation lines, retry
    cleaned = re.sub(
        r'^(?:Thought|Action|Observation|Action Input)\s*(?:\d+)?\s*:\s*.*$',
        '', raw, flags=re.MULTILINE | re.IGNORECASE
    )
    result = _try_parse(cleaned)
    if result:
        return result

    # Strategy 4 — try every {...} block from largest to smallest
    for m in sorted(re.finditer(r'\{[^<>]{10,}\}', raw, re.DOTALL), key=lambda x: -len(x.group())):
        result = _try_parse(m.group())
        if result:
            return result

    # Strategy 5 — recovery: build a minimal valid object from the text so the
    # pipeline continues rather than crashing. Confidence is set very low to signal
    # that this output is unreliable and should be treated as ESCALATED.
    logger.error(
        "[_parse_from_raw] All strategies failed — using recovery stub. raw (first 400): %s",
        raw[:400],
    )
    icd_pattern = re.compile(r'\b([A-Z]\d{2}(?:\.\d{1,4})?|\d{3,5}(?:\.\d{1,2})?)\b')
    found_codes = list(dict.fromkeys(icd_pattern.findall(raw)))[:10]
    pid_match = re.search(r'"patient_id"\s*:\s*"([^"]+)"', raw)
    patient_id = pid_match.group(1) if pid_match else "unknown"
    return AgentDiagnosis(
        patient_id=patient_id,
        diagnosis_hypothesis="[PARSE FAILURE] Model did not return structured output.",
        icd_codes_cited=found_codes,
        evidence_chain=["Recovery mode: structured output parsing failed after all strategies."],
        confidence_score=0.0,
        unverified_codes=[],
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
    _kickoff_with_retry(crew)

    pydantic_out = task.output.pydantic
    if pydantic_out is None:
        logger.warning(
            "[_run_single_task] output_pydantic is None — falling back to raw extraction. "
            "raw (first 300): %s", str(task.output.raw)[:300]
        )
        pydantic_out = _parse_from_raw(task.output.raw)

    logger.debug("[_run_single_task] icd_codes_cited=%s", pydantic_out.icd_codes_cited)
    return pydantic_out


def _run_manager_task(manager_agent, task_fn, *args, **kwargs) -> str:
    """Run the Manager Agent's contradiction task and return its raw prose output.

    The Manager Agent produces plain text (not AgentDiagnosis JSON), so we capture
    task.output.raw directly.  Any failure falls back to an empty string — the caller
    then falls back to the Python-level JSON diff so the pipeline never blocks.
    """
    task = task_fn(manager_agent, *args, **kwargs)
    crew = Crew(
        agents=[manager_agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )
    try:
        _kickoff_with_retry(crew)
        narrative = (task.output.raw or "").strip()
        logger.info(
            "[Manager] Contradiction narrative produced (%d chars)", len(narrative)
        )
        return narrative
    except Exception as exc:
        logger.warning(
            "[Manager] Manager Agent task failed — will fall back to Python diff. Error: %s", exc
        )
        return ""


def run_consensus_gate(
    patient_id: str,
    trajectory_json: str,
    diagnostician_agent,
    auditor_agent,
    manager_agent=None,
) -> tuple[AgentDiagnosis, AgentDiagnosis, float, int]:
    """
    Runs the Reflection Loop up to MAX_ITER times.
    Returns (diagnostician_output, auditor_output, final_kappa, iterations_used).
    Raises ConsensusFailure if kappa stays <= KAPPA_THRESHOLD after MAX_ITER.

    When manager_agent is provided, it is called on every failed κ check to produce a
    clinically-reasoned contradiction narrative.  That narrative replaces the bare Python
    JSON diff as the contradiction_report injected into the reflection task prompt.
    If the Manager Agent call fails for any reason, the pipeline falls back to the Python
    diff so correctness is never compromised.  κ itself is always computed in Python.
    """
    from src.tasks import get_mining_task, get_audit_task, get_reflection_task, get_manager_contradiction_task

    # Full trajectory code list is the kappa denominator — includes codes neither agent cited,
    # giving both label vectors genuine 0s and keeping Cohen's kappa well-defined.
    traj = json.loads(trajectory_json)
    all_codes = [entry["code"] for entry in traj.get("icd_codes", [])]
    logger.info("[ConsensusGate] Trajectory contains %d unique ICD codes for patient %s", len(all_codes), patient_id)

    logger.info("[ConsensusGate] Running initial diagnostician pass for patient %s", patient_id)
    d_output = _run_single_task(diagnostician_agent, get_mining_task, trajectory_json)

    logger.info("[ConsensusGate] Running initial auditor pass for patient %s", patient_id)
    a_output = _run_single_task(auditor_agent, get_audit_task, trajectory_json)

    for iteration in range(1, MAX_ITER + 1):
        kappa = compute_kappa(d_output.icd_codes_cited, a_output.icd_codes_cited, all_codes)
        logger.info(
            "[ConsensusGate] Iteration %d/%d — κ=%.3f (threshold=%.2f) | diag_codes=%s | audit_codes=%s",
            iteration, MAX_ITER, kappa, KAPPA_THRESHOLD,
            d_output.icd_codes_cited, a_output.icd_codes_cited,
        )

        if kappa > KAPPA_THRESHOLD:
            logger.info("[ConsensusGate] Consensus reached at iteration %d (κ=%.3f)", iteration, kappa)
            return d_output, a_output, kappa, iteration

        if iteration < MAX_ITER:
            logger.info("[ConsensusGate] κ below threshold — triggering reflection (iteration %d)", iteration)

            # Always build the Python diff first — it is the reliable structural baseline.
            python_diff = _build_contradiction_report(d_output, a_output)

            # If a Manager Agent is available, ask it to produce a richer clinical narrative.
            # The narrative is used as the contradiction report fed into the reflection prompts.
            # Fall back to the Python diff if the Manager Agent produces no meaningful output.
            if manager_agent is not None:
                narrative = _run_manager_task(
                    manager_agent, get_manager_contradiction_task, d_output, a_output, kappa
                )
                contradiction = narrative if len(narrative) > 100 else python_diff
                if contradiction is python_diff:
                    logger.warning(
                        "[ConsensusGate] Manager narrative was too short or empty — using Python diff"
                    )
            else:
                contradiction = python_diff

            d_output = _run_single_task(
                _strip_tools(diagnostician_agent), get_reflection_task, trajectory_json, contradiction
            )
            a_output = _run_single_task(
                _strip_tools(auditor_agent), get_reflection_task, trajectory_json, contradiction
            )

    final_kappa = compute_kappa(d_output.icd_codes_cited, a_output.icd_codes_cited, all_codes)
    if final_kappa <= KAPPA_THRESHOLD:
        raise ConsensusFailure(patient_id, final_kappa, MAX_ITER, d_output, a_output)
    return d_output, a_output, final_kappa, MAX_ITER
