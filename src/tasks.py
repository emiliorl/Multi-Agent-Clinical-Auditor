from crewai import Task

# Task 1: Data Mining
def get_mining_task(agent, patient_id):
    return Task(
        description=(
            f"1. Use the EHRPatternScanner tool to extract the history for patient {patient_id}.\n"
            "2. Identify every hospital admission (hadm_id) and the unique ICD codes assigned.\n"
            "3. Analyze vitals for 'Sepsis Triggers' (e.g., rising HR, falling SpO2).\n"
            "4. Output a structured 'Clinical Reasoning Chain' citing specific raw data rows."
        ),
        expected_output="A longitudinal trajectory of the patient's data, including all discovered ICD codes.",
        agent=agent
    )

# Task 2: Clinical Audit 
def get_audit_task(agent, mining_task):
    return Task(
        description=(
            "1. Review the clinical trajectory provided by the Lead Clinical Data Miner.\n"
            "2. For EVERY unique ICD code found, you MUST call the MedicalKnowledgeLookup tool.\n"
            "3. YOU ARE FORBIDDEN from using your own internal knowledge to define codes.\n"
            "4. Create a 'Grounding Table' with three columns: [ICD Code | Condition | KG Verification Token].\n"
            "5. Compare the Miner's observations against the 'audit_protocol' returned by the tool.\n"
            "6. If a 'Verification Token' is missing for a code, flag the entire audit as 'UNVERIFIED'."
        ),
        expected_output="A verified audit report where every clinical claim is linked to a unique KG Verification Token.",
        agent=agent,
        context=[mining_task] 
    )