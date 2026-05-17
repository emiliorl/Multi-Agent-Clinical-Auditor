import argparse
import time
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
    
    max_retries = 3
    retry_delay = 25  # seconds
    result = None

    for attempt in range(1, max_retries + 1):
        try:
            result = crew.kickoff()
            logger.info("Audit complete for patient %s", patient_id)
            print(result)
            break
        except Exception as e:
            err_str = str(e)
            is_rate_limit = (
                "429" in err_str or 
                "RESOURCE_EXHAUSTED" in err_str or 
                "rate limit" in err_str.lower() or
                "quota" in err_str.lower()
            )
            is_unavailable = (
                "503" in err_str or 
                "UNAVAILABLE" in err_str or 
                "overloaded" in err_str.lower()
            )
            
            if is_rate_limit:
                logger.warning(
                    "\n⚠️  [GEMINI API RATE LIMIT (429)] You have exceeded the free tier quota limits.\n"
                    "Waiting %d seconds for the API cooldown period before retrying (Attempt %d/%d)...",
                    retry_delay, attempt, max_retries
                )
                if attempt < max_retries:
                    time.sleep(retry_delay)
                else:
                    logger.error("\n❌ Max retries reached. The audit has been blocked by API rate limits.")
                    raise e
            elif is_unavailable:
                logger.warning(
                    "\n⚠️  [GEMINI API SERVICE UNAVAILABLE (503)] The Gemini service is currently overloaded or unavailable.\n"
                    "Waiting %d seconds before retrying (Attempt %d/%d)...",
                    retry_delay, attempt, max_retries
                )
                if attempt < max_retries:
                    time.sleep(retry_delay)
                else:
                    logger.error("\n❌ Max retries reached. The audit has been blocked by API service downtime.")
                    raise e
            else:
                logger.error("\n❌ An unexpected error occurred during the clinical audit: %s", e)
                raise e


if __name__ == "__main__":
    main()
