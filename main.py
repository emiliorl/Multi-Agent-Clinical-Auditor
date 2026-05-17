import argparse
import json
import time
from crewai import Crew, Process
from src.agents import diagnostician, auditor
from src.tasks import get_mining_task, get_audit_task, get_critique_task
from src.tools import EHRPatternScanner
from src.logger import get_logger

logger = get_logger(__name__)

_scanner = EHRPatternScanner()


def _get_trajectory_json(patient_id: str) -> str:
    raw = _scanner.run(patient_id)
    try:
        parsed = json.loads(raw)
        if "error" in parsed:
            raise ValueError(parsed["error"])
    except (json.JSONDecodeError, TypeError):
        raise ValueError(f"EHRPatternScanner returned non-JSON output for patient {patient_id}")
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Agent Clinical Auditor")
    parser.add_argument("--patient", required=True, help="MIMIC-IV subject_id")
    args = parser.parse_args()

    patient_id: str = args.patient

    logger.info("Scanning EHR for patient %s", patient_id)
    trajectory_json = _get_trajectory_json(patient_id)
    logger.info("Trajectory acquired (%d chars)", len(trajectory_json))

    t1 = get_mining_task(diagnostician, trajectory_json)
    t2 = get_audit_task(auditor, trajectory_json)
    t3 = get_critique_task(auditor, t1, t2)

    crew = Crew(
        agents=[diagnostician, auditor],
        tasks=[t1, t2, t3],
        process=Process.sequential,
        verbose=True,
    )

    logger.info("Starting adversarial clinical audit for patient %s", patient_id)

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
                "429" in err_str
                or "RESOURCE_EXHAUSTED" in err_str
                or "rate limit" in err_str.lower()
                or "quota" in err_str.lower()
            )
            is_unavailable = (
                "503" in err_str
                or "UNAVAILABLE" in err_str
                or "overloaded" in err_str.lower()
            )

            if is_rate_limit:
                logger.warning(
                    "[GEMINI 429] Rate limit hit. Waiting %ds before retry (%d/%d).",
                    retry_delay, attempt, max_retries,
                )
                if attempt < max_retries:
                    time.sleep(retry_delay)
                else:
                    logger.error("Max retries reached — blocked by API rate limits.")
                    raise
            elif is_unavailable:
                logger.warning(
                    "[GEMINI 503] Service unavailable. Waiting %ds before retry (%d/%d).",
                    retry_delay, attempt, max_retries,
                )
                if attempt < max_retries:
                    time.sleep(retry_delay)
                else:
                    logger.error("Max retries reached — blocked by API service downtime.")
                    raise
            else:
                logger.error("Unexpected error during clinical audit: %s", e)
                raise


if __name__ == "__main__":
    main()
