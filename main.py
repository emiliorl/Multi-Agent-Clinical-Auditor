from crewai import Crew, Task
from src.agents import diagnostician

# 1. Define the Task
mining_task = Task(
    description="Scan the history for patient '10000032' (use actual ID from your data). Find all diagnosis codes and timestamps.",
    expected_output="A structured summary of the patient's clinical trajectory.",
    agent=diagnostician
)

# 2. Assemble the Crew
clinical_crew = Crew(
    agents=[diagnostician],
    tasks=[mining_task],
    verbose=True
)

# 3. Kickoff
result = clinical_crew.kickoff()
print("\n\n########################")
print("## MINING RESULT ##")
print("########################\n")
print(result)