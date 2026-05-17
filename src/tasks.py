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
            f"1. Use the EHRPatternScanner tool to retrieve the trajectory for patient '{patient_id}'.\n"
            "2. For EVERY unique ICD code found, you MUST call the medical_knowledge_lookup tool.\n"
            "3. YOU ARE FORBIDDEN from using your own internal knowledge to define codes.\n"
            "4. Create a 'Grounding Table': | ICD Code | Condition | KG Verification Token |\n"
            "5. Compare observations against the 'audit_protocol' returned by the tool.\n"
            "6. If a 'Verification Token' is missing for any code, flag the audit as 'UNVERIFIED'."
        ),
        expected_output=(
            "A verified audit report where every clinical claim is linked to a unique "
            "KG Verification Token."
        ),
        agent=agent,
    )
