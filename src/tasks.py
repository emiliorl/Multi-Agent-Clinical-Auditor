import json as _json
import os
from dotenv import load_dotenv
from crewai import Task
from src.models import AgentDiagnosis

load_dotenv()


def _slim_trajectory_for_prompt(trajectory_json: str) -> str:
    """Drop fields the LLM doesn't need (flat icd_codes list, timeline strings).
    The full trajectory is preserved for the consensus gate; only the prompt copy is slimmed.
    """
    try:
        traj = _json.loads(trajectory_json)
    except Exception:
        return trajectory_json
    traj.pop("icd_codes", None)
    traj.pop("timeline", None)
    return _json.dumps(traj)

_EVIDENCE_FORMAT = (
    "hadm=<HADM_ID> admit=<YYYY-MM-DD> <HH:MM> type=<ADMIT_TYPE> has diagnosis <CODE> (<Description>)"
)

_JSON_SCHEMA_HINT = (
    "\n\nOUTPUT FORMAT — return ONLY a valid JSON object, no markdown fences, no extra text:\n"
    '{"patient_id": "<str>", "diagnosis_hypothesis": "<str>", '
    '"icd_codes_cited": ["<code>", ...], "evidence_chain": ["<str>", ...], '
    '"confidence_score": <float 0.0-1.0>, "unverified_codes": ["<code>", ...]}\n\n'
    "EVIDENCE_CHAIN FORMAT — each entry MUST follow this exact template:\n"
    f'  "{_EVIDENCE_FORMAT}"\n'
    "Example: \"hadm=123456 admit=2180-03-15 14:22 type=URGENT has diagnosis A419 (Sepsis, unspecified organism)\"\n"
    "Use the hadm_id, admit_time, and admission_type values from the trajectory JSON directly. "
    "Do NOT write free-text justifications — the dashboard will not render them as structured cards."
)


def _pydantic_output():
    """Return AgentDiagnosis for cloud (instructor works); None for local LM Studio (tool_choice incompatible)."""
    if os.getenv("USE_LOCAL_LLM", "false").lower() == "true":
        return None
    return AgentDiagnosis


def get_mining_task(agent, trajectory_json: str) -> Task:
    slim = _slim_trajectory_for_prompt(trajectory_json)
    return Task(
        description=(
            "You are the Lead Clinical Data Miner applying Clinical Chain-of-Thought (CCoT) reasoning: "
            "Perceive the full admission history, Think by identifying the dominant clinical pattern, "
            "Act by selecting only the codes that directly support that pattern, Reflect by discarding "
            "any code you cannot justify with a specific admission event.\n\n"
            "You have been given the complete patient trajectory below. "
            "Do NOT call any scanning tools — the data is already provided.\n\n"
            f"PATIENT TRAJECTORY JSON:\n{slim}\n\n"
            "Your job:\n"
            "1. Review the admission history and form a PRIMARY diagnosis hypothesis based on the "
            "clinical pattern across admissions (e.g., sepsis, chronic liver disease, COPD exacerbation).\n"
            "2. Select ONLY the ICD codes that directly support your primary hypothesis — MAXIMUM 12 codes. "
            "Do NOT cite every code present — only the ones with clear clinical evidence in the trajectory. "
            "A code that appears in an admission but does not support your primary hypothesis should be excluded. "
            "Aim for precision: a small, focused set of well-evidenced codes is better than a long list.\n"
            "3. Build an evidence chain: ONE entry per cited code (maximum 12 entries total) using EXACTLY the "
            "template from EVIDENCE_CHAIN FORMAT below (hadm_id, admit_time, admit_type, code, description "
            "taken directly from the trajectory JSON). No free-text — structured tags only.\n"
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
    try:
        traj = _json.loads(trajectory_json)
        patient_id = traj.get("patient_id", "unknown")
    except Exception:
        patient_id = "unknown"
    slim = _slim_trajectory_for_prompt(trajectory_json)

    return Task(
        description=(
            "You are the Clinical Audit Specialist operating under Clinical Chain-of-Thought (CCoT) "
            "reasoning rules: Perceive the data, Think through the grounding results, Act by selecting "
            "only evidence-grounded codes, Reflect by checking every cited code has a Verification Token.\n\n"
            "You have been given the complete patient trajectory below — DO NOT call EHRPatternScanner.\n\n"
            f"PATIENT TRAJECTORY JSON:\n{slim}\n\n"
            "Follow these steps exactly:\n\n"
            f"STEP 1 — Call batch_medical_knowledge_lookup EXACTLY ONCE with patient_id={patient_id!r}. "
            "If the trajectory has more than 15 codes, send ONLY the top ~15 most clinically salient codes "
            "(prioritize acute/primary diagnoses; deprioritize chronic backdrop, administrative Z-codes, "
            "and incidental findings). The tool hard-caps at 20 to protect the model's response budget — "
            "exceeding that means your prioritization wasn't tight enough.\n\n"
            "STEP 2 — Form your PRIMARY diagnosis hypothesis independently (do NOT look at any other "
            "agent's output). Only include codes that clinically support your hypothesis — "
            "VERIFIED grounding status alone is not sufficient justification.\n\n"
            "STEP 3 — icd_codes_cited: VERIFIED codes directly relevant to your primary hypothesis only — "
            "MAXIMUM 12 codes. Place incidental, administrative, or UNVERIFIED codes in unverified_codes. "
            "A code with status=UNVERIFIED must NEVER appear in icd_codes_cited.\n\n"
            "STEP 4 — evidence_chain: ONE entry per cited code (maximum 12 entries total) using EXACTLY the "
            "template from EVIDENCE_CHAIN FORMAT below (hadm_id, admit_time, admit_type, code, description "
            "taken directly from the trajectory JSON). No free-text — structured tags only.\n\n"
            "STEP 5 — Assign confidence_score (0.0–1.0).\n\n"
            "Return a single JSON object matching the AgentDiagnosis schema."
            + _JSON_SCHEMA_HINT
        ),
        expected_output=(
            "A JSON object with fields: patient_id, diagnosis_hypothesis, icd_codes_cited (list of str — "
            "ONLY verified codes directly relevant to the primary hypothesis, not all verified codes), "
            "evidence_chain (list of str), confidence_score (float 0-1), unverified_codes (list of str)."
        ),
        output_pydantic=None,
        agent=agent,
    )


def get_reflection_task(agent, trajectory_json: str, contradiction_report: str) -> Task:
    """Task injected during the Reflection Loop when kappa < threshold."""
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
            "Apply Clinical Chain-of-Thought (CCoT) reasoning — Perceive the contradiction, "
            "Think through each disputed code, Act by revising your code list, Reflect on whether "
            "your revised output would reach κ > 0.80 with the other agent's revised output.\n\n"
            "Instructions:\n"
            "1. Review each disagreement. For codes only you cited, decide if the evidence is strong enough to keep them.\n"
            "2. For codes the other agent cited that you did not, decide if you should add them.\n"
            "3. Aim to converge — only keep a code in icd_codes_cited if you have clear clinical evidence. "
            "The system requires Cohen's Kappa κ > 0.80 inter-agent agreement; unnecessary divergence will "
            "trigger another reflection round or physician escalation.\n"
            "4. Re-emit evidence_chain entries using EXACTLY the EVIDENCE_CHAIN FORMAT from the original task "
            "(hadm=... admit=... type=... has diagnosis CODE (Desc)). No free-text.\n"
            "5. No tools are available for this task — produce the revised JSON directly.\n\n"
            "Return ONLY the JSON object below — no extra text, no markdown fences."
            + _JSON_SCHEMA_HINT
        ),
        expected_output=(
            "A JSON object with fields: patient_id, diagnosis_hypothesis, icd_codes_cited (list of str), "
            "evidence_chain (list of str), confidence_score (float 0-1), unverified_codes (list of str)."
        ),
        output_pydantic=_pydantic_output(),
        # Override agent's tools — reflection should never call tools, and exposing them
        # to Gemini causes it to loop on tool signals instead of emitting the final JSON
        # (manifesting as "Invalid response from LLM call - None or empty" via max_iter).
        tools=[],
        agent=agent,
    )


def get_manager_contradiction_task(
    agent,
    d_output,
    a_output,
    kappa: float,
) -> Task:
    """Task for the Manager Agent: produce a clinically-reasoned contradiction narrative.

    Python already computed κ and decided the threshold was not met.  The Manager Agent's
    job is NOT to recompute κ — it is to explain *why* the two assessments diverge so that
    the reflection task prompt is clinically meaningful rather than a bare code diff.
    """

    def _fmt(output) -> str:
        try:
            # Drop evidence_chain — the Manager only needs hypothesis + codes + confidence
            # to write the contradiction narrative. evidence_chain entries are long per-code
            # strings that bloat the prompt without informing the reasoning.
            slim = output.model_dump()
            slim.pop("evidence_chain", None)
            return _json.dumps(slim, indent=2)
        except Exception:
            return str(output)

    d_only = sorted(set(d_output.icd_codes_cited) - set(a_output.icd_codes_cited))
    a_only = sorted(set(a_output.icd_codes_cited) - set(d_output.icd_codes_cited))
    shared = sorted(set(d_output.icd_codes_cited) & set(a_output.icd_codes_cited))

    return Task(
        description=(
            f"Inter-agent agreement is κ={kappa:.3f}, which is below the required threshold of κ > 0.80. "
            "Your role as Clinical Governance Manager is NOT to diagnose. "
            "It is to explain the clinical reasons behind the disagreement so that both agents "
            "can revise their assessments in the next reflection round.\n\n"
            f"DIAGNOSTICIAN OUTPUT:\n{_fmt(d_output)}\n\n"
            f"AUDITOR OUTPUT:\n{_fmt(a_output)}\n\n"
            "CODE AGREEMENT SUMMARY:\n"
            f"  Shared codes (both agents agree): {shared}\n"
            f"  Diagnostician only: {d_only}\n"
            f"  Auditor only: {a_only}\n\n"
            "Produce a CONTRADICTION REPORT with the following four sections:\n\n"
            "SECTION 1 — HYPOTHESIS COMPARISON\n"
            "Compare the two primary diagnosis hypotheses. Are they clinically compatible or "
            "contradictory? Explain why, in one short paragraph.\n\n"
            "SECTION 2 — CODE-LEVEL DISAGREEMENTS\n"
            "For each code cited by only one agent, write one sentence explaining the clinical "
            "rationale for why it may or may not belong in the primary hypothesis. "
            "Do not just list the codes -- explain them.\n\n"
            "SECTION 3 -- GROUNDING DISCIPLINE DIFFERENCES\n"
            "Note any codes where one agent placed the code in icd_codes_cited while the other "
            "placed it in unverified_codes. This is a grounding rule issue, not a clinical one -- "
            "flag it explicitly so agents apply the rule consistently.\n\n"
            "SECTION 4 -- CONVERGENCE GUIDANCE\n"
            "Give each agent one or two specific, actionable instructions for revising their code "
            "selection to reach kappa > 0.80. Name the specific codes each agent should reconsider "
            "and state a concrete clinical reason for each recommendation.\n\n"
            "Write in plain prose. Do NOT output JSON. Do NOT diagnose the patient. "
            "Do NOT recompute kappa yourself."
        ),
        expected_output=(
            "A plain-text CONTRADICTION REPORT with four labelled sections: "
            "(1) Hypothesis Comparison, (2) Code-Level Disagreements, "
            "(3) Grounding Discipline Differences, (4) Convergence Guidance."
        ),
        agent=agent,
    )


def get_critique_from_outputs_task(agent, diagnostician_output, auditor_output) -> Task:
    """Critique task that embeds pre-computed outputs -- avoids re-running agents in a sequential crew."""

    def _fmt(output) -> str:
        try:
            slim = output.model_dump()
            slim.pop("evidence_chain", None)
            return _json.dumps(slim, indent=2)
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
            "Be adversarial -- your job is to find every point of disagreement, not to be polite."
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
            "of the same patient -- one from the Lead Clinical Data Miner and one from your own prior audit. "
            "Your job is to produce an adversarial critique:\n\n"
            "1. List every ICD code cited by the Diagnostician but NOT by you (Auditor).\n"
            "2. List every ICD code cited by you but NOT by the Diagnostician.\n"
            "3. For each disagreement, state the clinical reason why one agent included it and the other did not.\n"
            "4. Identify any codes where grounding stages differ (e.g., Diagnostician accepted UNVERIFIED, Auditor did not).\n"
            "5. Produce a CONTRADICTION REPORT listing all substantive disagreements.\n"
            "6. Conclude with a CONSENSUS ASSESSMENT: do both agents fundamentally agree on the primary diagnosis? "
            "State yes/no and the key codes driving agreement or disagreement.\n\n"
            "Be adversarial -- your job is to find every point of disagreement, not to be polite."
        ),
        expected_output=(
            "A structured critique containing: (1) codes cited by Diagnostician only, "
            "(2) codes cited by Auditor only, (3) per-disagreement clinical reasoning, "
            "(4) grounding stage conflicts, (5) CONTRADICTION REPORT, (6) CONSENSUS ASSESSMENT."
        ),
        context=[mining_task, audit_task],
        agent=agent,
    )
