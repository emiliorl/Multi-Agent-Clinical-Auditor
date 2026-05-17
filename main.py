import argparse
from crewai import Crew, Process
from src.agents import diagnostician, auditor
from src.tasks import get_mining_task, get_audit_task
from src.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Agent Clinical Auditor")
    parser.add_argument("--patient", required=True, help="MIMIC-IV subject_id")
    args = parser.parse_args()

    patient_id: str = args.patient

    mining_task = get_mining_task(diagnostician, patient_id)
    audit_task = get_audit_task(auditor, patient_id)

    crew = Crew(
        agents=[diagnostician, auditor],
        tasks=[mining_task, audit_task],
        process=Process.sequential,
        verbose=True,
    )

    logger.info("Starting clinical audit for patient %s", patient_id)
    result = crew.kickoff()
    logger.info("Audit complete for patient %s", patient_id)
    print(result)


if __name__ == "__main__":
    main()
