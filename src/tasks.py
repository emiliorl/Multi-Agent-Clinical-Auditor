from crewai import Task
from src.models import AgentDiagnosis


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
        ),
        expected_output=(
            "A JSON object with fields: patient_id, diagnosis_hypothesis, icd_codes_cited (list of str — "
            "ONLY codes with direct clinical evidence for the primary hypothesis), "
            "evidence_chain (list of str), confidence_score (float 0-1), unverified_codes (list of str)."
        ),
        output_pydantic=AgentDiagnosis,
        agent=agent,
    )


def get_audit_task(agent, trajectory_json: str) -> Task:
    import json as _json
    try:
        traj = _json.loads(trajectory_json)
        patient_id = traj.get("patient_id", "unknown")
        codes_list = [
            {"icd_code": e["code"], "icd_version": e["version"]}
            for e in traj.get("icd_codes", [])
        ]
        batch_payload = _json.dumps({"patient_id": patient_id, "codes": codes_list})
    except Exception:
        patient_id = "unknown"
        batch_payload = '{"patient_id": "unknown", "codes": []}'

    return Task(
        description=(
            "You are the Clinical Audit Specialist operating under Clinical Chain-of-Thought (CCoT) rules. "
            "You have been given the complete patient trajectory below — DO NOT call EHRPatternScanner. "
            "You must form an independent assessment without seeing any other agent's output.\n\n"
            f"PATIENT TRAJECTORY JSON:\n{trajectory_json}\n\n"
            "Your job — follow these steps exactly:\n\n"
            "STEP 1 — Call batch_medical_knowledge_lookup EXACTLY ONCE with this payload:\n"
            f"{batch_payload}\n\n"
            "STEP 2 — Form your PRIMARY diagnosis hypothesis independently from the trajectory. "
            "Do NOT default to citing all verified codes — only include codes that clinically support "
            "your hypothesis. A code being VERIFIED does not mean it is relevant to the primary diagnosis.\n\n"
            "STEP 3 — Select icd_codes_cited: codes from the grounding_table where status=VERIFIED "
            "AND the code is directly relevant to your primary hypothesis. Exclude verified codes "
            "that are incidental findings, prior history unrelated to the current presentation, or "
            "administrative codes. Place excluded verified codes in unverified_codes with a note.\n\n"
            "STEP 4 — Write diagnosis_hypothesis that reflects your clinical reasoning.\n\n"
            "STEP 5 — Build evidence_chain: for each cited code, state the specific admission (hadm_id) "
            "and data point that justifies its inclusion in the primary diagnosis.\n\n"
            "STEP 6 — Assign confidence_score (0.0–1.0).\n\n"
            "Return a single JSON object matching the AgentDiagnosis schema."
        ),
        expected_output=(
            "A JSON object with fields: patient_id, diagnosis_hypothesis, icd_codes_cited (list of str — "
            "ONLY verified codes directly relevant to the primary hypothesis, not all verified codes), "
            "evidence_chain (list of str), confidence_score (float 0-1), unverified_codes (list of str)."
        ),
        output_pydantic=AgentDiagnosis,
        agent=agent,
    )


def get_reflection_task(agent, trajectory_json: str, contradiction_report: str) -> Task:
    """Task injected during the Reflection Loop when kappa < threshold."""
    import json as _json
    try:
        traj = _json.loads(trajectory_json)
        patient_id = traj.get("patient_id", "unknown")
        codes_list = [
            {"icd_code": e["code"], "icd_version": e["version"]}
            for e in traj.get("icd_codes", [])
        ]
        batch_payload = _json.dumps({"patient_id": patient_id, "codes": codes_list})
    except Exception:
        patient_id = "unknown"
        batch_payload = '{"patient_id": "unknown", "codes": []}'

    return Task(
        description=(
            "A previous round of assessment did NOT reach consensus (κ ≤ 0.80). "
            "You are asked to revise your clinical assessment in light of the contradiction report below.\n\n"
            f"CONTRADICTION REPORT:\n{contradiction_report}\n\n"
            f"PATIENT TRAJECTORY JSON:\n{trajectory_json}\n\n"
            "Re-examine the trajectory carefully. For each code where you disagreed with the other agent, "
            "reconsider whether the evidence in the trajectory supports including or excluding it.\n\n"
            "If you are the Auditor, call batch_medical_knowledge_lookup EXACTLY ONCE with this payload:\n"
            f"{batch_payload}\n\n"
            "After reflection, return a single JSON object matching the AgentDiagnosis schema. "
            "Your revised output should resolve as many disagreements as the evidence allows."
        ),
        expected_output=(
            "A JSON object with fields: patient_id, diagnosis_hypothesis, icd_codes_cited (list of str), "
            "evidence_chain (list of str), confidence_score (float 0-1), unverified_codes (list of str). "
            "The output should reflect careful reconsideration of the contradiction points."
        ),
        output_pydantic=AgentDiagnosis,
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
