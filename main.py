from crewai import Crew, Task, Process
from src.agents import diagnostician, auditor

# Task 1: Mining
mining_task = Task(
    description="Scan the history for patient '10000032'. Extract all ICD codes and describe the trajectory.",
    expected_output="A structured list of diagnosis codes and clinical observations.",
    agent=diagnostician
)

# Task 2: Auditing
audit_task = Task(
    description=(
        "1. Analyze the Miner's trajectory for Patient 10000032.\n"
        "2. For EVERY unique ICD code, you MUST execute 'medical_knowledge_lookup'.\n"
        "3. YOU ARE FORBIDDEN from using your own training data to define medical terms.\n"
        "4. Your report MUST contain a 'Grounding Table' formatted as: \n"
        "   | ICD Code | Verification Token | Sepsis Pathway |\n"
        "5. If the Verification Token is missing, the audit is considered legally invalid."
    ),
    expected_output="A verified clinical audit report where every code is mapped to a KG Verification Token.",
    agent=auditor,
    context=[mining_task]
)

# Assemble the Crew
clinical_auditor_crew = Crew(
    agents=[diagnostician, auditor],
    tasks=[mining_task, audit_task],
    process=Process.sequential, 
    verbose=True
)

result = clinical_auditor_crew.kickoff()
print(result)