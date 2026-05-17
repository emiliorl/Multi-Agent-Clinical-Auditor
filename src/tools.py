import json
import os
import pandas as pd
from crewai.tools import BaseTool
from src.models import Admission, IcdEntry, PatientTrajectory
from src.logger import get_logger

logger = get_logger(__name__)

_DIAGNOSES_FILENAME = "diagnoses_icd.csv.gz"
_ADMISSIONS_FILENAME = "admissions.csv.gz"


def _data_dir() -> str:
    return os.path.dirname(
        os.getenv(
            "CLINICAL_DATA_PATH",
            f"./data/mimic-iv-clinical-database-demo-2.2/hosp/{_DIAGNOSES_FILENAME}",
        )
    )


class EHRPatternScanner(BaseTool):
    name: str = "EHRPatternScanner"
    description: str = (
        "Searches gzipped MIMIC-IV files for a specific patient's longitudinal trajectory. "
        "Useful for identifying sepsis triggers and vital sign patterns."
    )

    def _run(self, patient_id: str) -> str:
        # --- input validation (1.4) ---
        try:
            pid_int = int(patient_id)
        except (ValueError, TypeError):
            return json.dumps({
                "error": f"patient_id must be a valid integer, got: {patient_id!r}",
                "patient_id": patient_id,
                "code": "INVALID_PATIENT_ID",
            })

        data_dir = _data_dir()
        diag_path = os.path.join(data_dir, _DIAGNOSES_FILENAME)
        adm_path = os.path.join(data_dir, _ADMISSIONS_FILENAME)

        for label, path in (("diagnoses", diag_path), ("admissions", adm_path)):
            if not os.path.exists(path):
                return json.dumps({
                    "error": f"{label} file not found: {path}",
                    "patient_id": patient_id,
                    "code": "FILE_NOT_FOUND",
                })

        try:
            # --- sequence mining: chunked read for memory efficiency ---
            diag_rows = []
            for chunk in pd.read_csv(diag_path, compression="gzip", chunksize=10_000):
                match = chunk[chunk["subject_id"] == pid_int]
                if not match.empty:
                    diag_rows.append(match)

            if not diag_rows:
                logger.info("No records found for patient %s", patient_id)
                return json.dumps({
                    "error": f"No records found for patient_id {patient_id}",
                    "patient_id": patient_id,
                    "code": "PATIENT_NOT_FOUND",
                })

            diag_df = pd.concat(diag_rows).reset_index(drop=True)

            # --- load admissions for this patient ---
            adm_rows = []
            for chunk in pd.read_csv(adm_path, compression="gzip", chunksize=10_000):
                match = chunk[chunk["subject_id"] == pid_int]
                if not match.empty:
                    adm_rows.append(match)

            adm_df = (
                pd.concat(adm_rows).reset_index(drop=True)
                if adm_rows
                else pd.DataFrame(columns=["hadm_id", "admittime", "admission_type"])
            )
            adm_map = {
                row["hadm_id"]: row
                for _, row in adm_df.iterrows()
            }

            # --- build per-admission diagnosis lists ---
            admissions: list[Admission] = []
            for hadm_id, group in diag_df.groupby("hadm_id"):
                adm_row = adm_map.get(hadm_id, {})
                diagnoses = [
                    IcdEntry(
                        code=str(r["icd_code"]).strip(),
                        version=str(int(r["icd_version"])),
                        description=None,
                    )
                    for _, r in group.iterrows()
                ]
                admissions.append(
                    Admission(
                        hadm_id=int(hadm_id),
                        admit_time=str(adm_row.get("admittime", "unknown")),
                        admit_type=str(adm_row.get("admission_type", "unknown")),
                        diagnoses=diagnoses,
                    )
                )

            # sort admissions chronologically
            admissions.sort(key=lambda a: a.admit_time)

            # --- flat deduplicated ICD code list ---
            seen: set[tuple[str, str]] = set()
            flat_codes: list[IcdEntry] = []
            for adm in admissions:
                for entry in adm.diagnoses:
                    key = (entry.code, entry.version)
                    if key not in seen:
                        seen.add(key)
                        flat_codes.append(entry)

            # --- timeline: one label per admission ---
            timeline = [
                f"hadm={adm.hadm_id} admit={adm.admit_time} type={adm.admit_type}"
                for adm in admissions
            ]

            trajectory = PatientTrajectory(
                patient_id=str(pid_int),
                admissions=admissions,
                icd_codes=flat_codes,
                timeline=timeline,
            )

            logger.info(
                "Built trajectory for patient %s: %d admissions, %d unique ICD codes",
                patient_id,
                len(admissions),
                len(flat_codes),
            )
            return trajectory.model_dump_json()

        except Exception as exc:
            logger.error("EHRPatternScanner failed for patient %s: %s", patient_id, exc)
            return json.dumps({
                "error": str(exc),
                "patient_id": patient_id,
                "code": "SCANNER_ERROR",
            })


class MedicalKnowledgeLookup(BaseTool):
    name: str = "medical_knowledge_lookup"
    description: str = "MANDATORY: Use this to fetch the Verification Token for an ICD code."

    def _run(self, icd_code: str) -> str:
        clean_code = str(icd_code).strip().replace("'", "").replace('"', "")

        if not clean_code.startswith("ICD_9_"):
            key = f"ICD_9_{clean_code}"
        else:
            key = clean_code

        kb_path = "data/kb_sepsis.json"
        try:
            with open(kb_path, "r") as f:
                kb = json.load(f)

            node = kb["nodes"].get(key)
            if node:
                return json.dumps({
                    "condition": node["label"],
                    "token": node["verification_token"],
                    "protocol": node["relationships"],
                })
            return f"CRITICAL ERROR: Code {clean_code} NOT found in KG. Audit blocked."
        except Exception as exc:
            logger.error("MedicalKnowledgeLookup failed: %s", exc)
            return f"SYSTEM_FAILURE: {exc}"
