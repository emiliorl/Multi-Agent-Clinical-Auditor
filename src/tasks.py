from crewai import Task

def get_mining_task(agent, patient_id):
    return Task(
        description=(
            f"Use the EHR_Trajectory_Miner tool to extract the history for patient {patient_id}. "
            "Analyze vitals for 'Sepsis Triggers' (e.g., rising HR, falling SpO2). "
            "You MUST output a structured 'Clinical Reasoning Chain' with timestamps."
        ),
        expected_output="A longitudinal analysis of the patient's condition grounded in raw EHR data.",
        agent=agent
    )