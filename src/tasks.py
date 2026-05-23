import os
from dotenv import load_dotenv
from crewai import Task
from src.models import AgentDiagnosis

load_dotenv()

_JSON_SCHEMA_HINT = (
    "\n\nOUTPUT FORMAT — return ONLY a valid JSON object, no markdown fences, no extra text:\n"
    '{"patient_id": "<str>", "diagnosis_hypothesis": "<str>", '
    '"icd_codes_cited": ["<code>", ...], "evidence_chain": ["<str>", ...], '
    '"confidence_score": <float 0.0-1.0>, "unverified_codes": ["<code>", ...]}'
)


def _pydantic_output():
    """Return AgentDiagnosis for cloud (instructor works); None for local LM Studio (tool_choice incompatible)."""
    if os.getenv("USE_LOCAL_LLM", "false").lower() == "true":
        return None
    return AgentDiagnosis


def get_mining_task(agent, trajectory_json: str) -> Task:
    return Task(
        description=(
            "You are the Lead Clinical Data Miner. You have been given the complete patient "
            "trajectory below. Do NOT call any scanning tools — the data is already provided.\n\n"
            f"PATIENT TRAJECTORY JSON:\n{trajectory_json}\n\n"
            "Your job:\n"
            "1. Review the admission history and form a PRIMARY diagnosis hypothesis based on the "
            "clinical pattern across admissions (e.g., sepsis, chronic liver disease, COPD exacerbation).\n"
            "2. Select ONLY the ICD codes that directly support your primary hypothesis. "
            "Do NOT cite every code present — only the ones with clear clinical evidence in the trajectory. "
            "A code that appears in an admission but does not support your primary hypothesis should be excluded.\n"
            "3. Build an evidence chain: for each cited code, state the specific admission (hadm_id) "
            "and clinical data point that justifies its inclusion.\n"
            "4. Place codes you are uncertain about in unverified_codes — do not cite them in icd_codes_cited.\n"
            "5. Assign a confidence_score (0.0–1.0) reflecting your certainty in the primary hypothesis.\n\n"
            "Return a single JSON object matching the AgentDiagnosis schema exactly."
            + _JSON_SCHEMA_HINT
        ),
        expected_output=(
            "A JSON object with fields: patient_id, diagnosis_hypothesis, icd_codes_cited (list of str — "
            "ONLY codes with direct clinical evidence for the primary hypothesis), "
            "evidence_chain (list of str), confidence_score (float 0-1), unverified_codes (list of str)."
        ),
        output_pydantic=_pydantic_output(),
        agent=agent,
    )


def get_audit_task(agent, trajectory_json: str) -> Task:
    import json as _json
    try:
        traj = _json.loads(trajectory_json)
        patient_id = traj.get("patient_id", "unknown")
    except Exception:
        patient_id = "unknown"

    return Task(
        description=(
            "You are the Clinical Audit Specialist operating under CCoT rules. "
            "You have been given the complete patient trajectory below — DO NOT call EHRPatternScanner.\n\n"
            f"PATIENT TRAJECTORY JSON:\n{trajectory_json}\n\n"
            "Follow these steps exactly:\n\n"
            f"STEP 1 — Call batch_medical_knowledge_lookup EXACTLY ONCE with patient_id={patient_id!r} "
            "and all codes from the trajectory's icd_codes list.\n\n"
            "STEP 2 — Form your PRIMARY diagnosis hypothesis independently. "
            "Only include codes that clinically support your hypothesis — VERIFIED status alone is not enough.\n\n"
            "STEP 3 — icd_codes_cited: VERIFIED codes directly relevant to your primary hypothesis only. "
            "Place incidental or administrative codes in unverified_codes.\n\n"
            "STEP 4 — evidence_chain: for each cited code, state the hadm_id and data point.\n\n"
            "STEP 5 — Assign confidence_score (0.0–1.0).\n\n"
            "Return a single JSON object matching the AgentDiagnosis schema."
            + _JSON_SCHEMA_HINT
        ),
        expected_output=(
            "A JSON object with fields: patient_id, diagnosis_hypothesis, icd_codes_cited (list of str — "
            "ONLY verified codes directly relevant to the primary hypothesis, not all verified codes), "
            "evidence_chain (list of str), confidence_score (float 0-1), unverified_codes (list of str)."
        ),
        output_pydantic=_pydantic_output(),
        agent=agent,
    )


def get_reflection_task(agent, trajectory_json: str, contradiction_report: str) -> Task:
    """Task injected during the Reflection Loop when kappa < threshold."""
    import json as _json
    try:
        traj = _json.loads(trajectory_json)
        patient_id = traj.get("patient_id", "unknown")
    except Exception:
        patient_id = "unknown"

    return Task(
        description=(
            "A previous round did NOT reach consensus (κ ≤ 0.80). "
            "Revise your assessment to resolve the disagreements listed below.\n\n"
            f"CONTRADICTION REPORT:\n{contradiction_report}\n\n"
            f"Patient ID: {patient_id}\n\n"
            "Instructions:\n"
            "1. Review each disagreement. For codes only you cited, decide if the evidence is strong enough to keep them.\n"
            "2. For codes the other agent cited that you did not, decide if you should add them.\n"
            "3. Aim to converge — only keep a code in icd_codes_cited if you have clear clinical evidence.\n"
            "4. Do NOT re-call any tools.\n\n"
            "Return ONLY the JSON object below — no extra text, no markdown fences."
            + _JSON_SCHEMA_HINT
        ),
        expected_output=(
            "A JSON object with fields: patient_id, diagnosis_hypothesis, icd_codes_cited (list of str), "
            "evidence_chain (list of str), confidence_score (float 0-1), unverified_codes (list of str)."
        ),
        output_pydantic=_pydantic_output(),
        agent=agent,
    )


def get_critique_from_outputs_task(agent, diagnostician_output, auditor_output) -> Task:
    """Critique task that embeds pre-computed outputs — avoids re-running agents in a sequential crew."""
    import json as _json

    def _fmt(output) -> str:
        try:
            return _json.dumps(output.model_dump(), indent=2)
        except Exception:
            return str(output)

    return Task(
        description=(
            "You are the Clinical Audit Specialist. Below are TWO independent assessments "
            "of the same patient produced by separate agents that did NOT share outputs.\n\n"
            f"DIAGNOSTICIAN OUTPUT:\n{_fmt(diagnostician_output)}\n\n"
            f"AUDITOR OUTPUT:\n{_fmt(auditor_output)}\n\n"
            "Your job is to produce an adversarial critique:\n\n"
            "1. List every ICD code cited by the Diagnostician but NOT by you (Auditor).\n"
            "2. List every ICD code cited by you but NOT by the Diagnostician.\n"
            "3. For each disagreement, state the clinical reason why one agent included it and the other did not.\n"
            "4. Identify any codes where grounding stages differ (e.g., Diagnostician accepted UNVERIFIED, Auditor did not).\n"
            "5. Produce a CONTRADICTION REPORT listing all substantive disagreements.\n"
            "6. Conclude with a CONSENSUS ASSESSMENT: do both agents fundamentally agree on the primary diagnosis? "
            "State yes/no and the key codes driving agreement or disagreement.\n\n"
            "Be adversarial — your job is to find every point of disagreement, not to be polite."
        ),
        expected_output=(
            "A structured critique containing: (1) codes cited by Diagnostician only, "
            "(2) codes cited by Auditor only, (3) per-disagreement clinical reasoning, "
            "(4) grounding stage conflicts, (5) CONTRADICTION REPORT, (6) CONSENSUS ASSESSMENT."
        ),
        agent=agent,
    )


def get_critique_task(agent, mining_task: Task, audit_task: Task) -> Task:
    return Task(
        description=(
            "You are the Clinical Audit Specialist. You now have access to TWO independent assessments "
            "of the same patient — one from the Lead Clinical Data Miner and one from your own prior audit. "
            "Your job is to produce an adversarial critique:\n\n"
            "1. List every ICD code cited by the Diagnostician but NOT by you (Auditor).\n"
            "2. List every ICD code cited by you but NOT by the Diagnostician.\n"
            "3. For each disagreement, state the clinical reason why one agent included it and the other did not.\n"
            "4. Identify any codes where grounding stages differ (e.g., Diagnostician accepted UNVERIFIED, Auditor did not).\n"
            "5. Produce a CONTRADICTION REPORT listing all substantive disagreements.\n"
            "6. Conclude with a CONSENSUS ASSESSMENT: do both agents fundamentally agree on the primary diagnosis? "
            "State yes/no and the key codes driving agreement or disagreement.\n\n"
            "Be adversarial — your job is to find every point of disagreement, not to be polite."
        ),
        expected_output=(
            "A structured critique containing: (1) codes cited by Diagnostician only, "
            "(2) codes cited by Auditor only, (3) per-disagreement clinical reasoning, "
            "(4) grounding stage conflicts, (5) CONTRADICTION REPORT, (6) CONSENSUS ASSESSMENT."
        ),
        context=[mining_task, audit_task],
        agent=agent,
    )
