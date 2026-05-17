from crewai import Task


def get_mining_task(agent, patient_id: str) -> Task:
    return Task(
        description=(
            f"1. Use the EHRPatternScanner tool to extract the history for patient '{patient_id}'.\n"
            "2. Identify every hospital admission (hadm_id) and the unique ICD codes assigned.\n"
            "3. Analyze the trajectory for sepsis triggers (e.g., patterns across admissions).\n"
            "4. Output a structured 'Clinical Reasoning Chain' citing specific data from the trajectory."
        ),
        expected_output=(
            "A longitudinal trajectory of the patient's data, including all discovered ICD codes "
            "with their version (ICD-9 or ICD-10) and admission context."
        ),
        agent=agent,
    )


def get_audit_task(agent, patient_id: str) -> Task:
    return Task(
        description=(
            f"Perform a Clinical Chain-of-Thought (CCoT) audit for patient '{patient_id}'.\n\n"
            f"1. Use EHRPatternScanner to retrieve the trajectory for patient '{patient_id}'.\n"
            "2. Parse the returned JSON to extract every unique ICD code and its icd_version.\n"
            "3. For EVERY code, call medical_knowledge_lookup with this exact JSON payload:\n"
            '   {"icd_code": "<code>", "icd_version": "<9 or 10>", "patient_id": "<id>"}\n'
            "4. After each lookup, explicitly state:\n"
            "   'Stage X matched at similarity Y — Verification Token: <token>'\n"
            "5. Build a Grounding Table:\n"
            "   | ICD Code | Version | Stage | Similarity | Verification Token | Status |\n"
            "6. Codes with status=UNVERIFIED must appear in a separate UNVERIFIED section.\n"
            "7. YOU ARE FORBIDDEN from reasoning about or citing any unverified code.\n"
            "8. YOU ARE FORBIDDEN from using your own training knowledge to define codes."
        ),
        expected_output=(
            "A CCoT audit report containing: (1) a Grounding Table mapping every ICD code to its "
            "stage, similarity score, and Verification Token; (2) an UNVERIFIED section listing "
            "any codes that could not be grounded; (3) a summary of grounding coverage."
        ),
        agent=agent,
    )
