from crewai import Task
from src.models import AgentDiagnosis


def get_mining_task(agent, trajectory_json: str) -> Task:
    return Task(
        description=(
            "You are the Lead Clinical Data Miner. You have been given the complete patient "
            "trajectory below. Do NOT call any scanning tools — the data is already provided.\n\n"
            f"PATIENT TRAJECTORY JSON:\n{trajectory_json}\n\n"
            "Your job:\n"
            "1. Parse the trajectory to identify all hospital admissions (hadm_id) and their ICD codes.\n"
            "2. Analyze the trajectory for sepsis triggers and patterns across admissions.\n"
            "3. Build an evidence chain: cite specific data rows from the trajectory that support your hypothesis.\n"
            "4. List every ICD code you are citing in icd_codes_cited (strings only, e.g. '99591').\n"
            "5. Any ICD code you cannot interpret or find evidence for must go in unverified_codes.\n"
            "6. Assign a confidence_score between 0.0 and 1.0 for your overall diagnosis hypothesis.\n\n"
            "Return a single JSON object matching the AgentDiagnosis schema exactly."
        ),
        expected_output=(
            "A JSON object with fields: patient_id, diagnosis_hypothesis, icd_codes_cited (list of str), "
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
            "STEP 2 — From the returned grounding_table, build your assessment:\n"
            "  - icd_codes_cited: codes where status=VERIFIED (use the icd_code string, e.g. '496')\n"
            "  - unverified_codes: codes where status=UNVERIFIED\n"
            "  - YOU ARE FORBIDDEN from citing any code with status=UNVERIFIED\n\n"
            "STEP 3 — Write diagnosis_hypothesis based only on verified codes.\n\n"
            "STEP 4 — Build evidence_chain: each item is one piece of evidence from the trajectory.\n\n"
            "STEP 5 — Assign confidence_score (0.0–1.0).\n\n"
            "Return a single JSON object matching the AgentDiagnosis schema."
        ),
        expected_output=(
            "A JSON object with fields: patient_id, diagnosis_hypothesis, icd_codes_cited (list of str), "
            "evidence_chain (list of str), confidence_score (float 0-1), unverified_codes (list of str). "
            "All cited codes must appear as VERIFIED in the batch_medical_knowledge_lookup response."
        ),
        output_pydantic=AgentDiagnosis,
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
