import json
import os
import pandas as pd
from crewai.tools import BaseTool
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.models import Admission, IcdEntry, PatientTrajectory
from src.icd_client import ICDApiClient
from src.grounding_logger import log_grounding_attempt
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


_SIMILARITY_THRESHOLD = 0.85
_icd_client = ICDApiClient()
_embedder: "SentenceTransformer | None" = None


def _get_embedder() -> "SentenceTransformer":
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _cosine(a: list[float], b: list[float]) -> float:
    import numpy as np
    va = np.array(a).reshape(1, -1)
    vb = np.array(b).reshape(1, -1)
    return float(cosine_similarity(va, vb)[0][0])


def _ground_single_code(icd_code: str, icd_version: str, patient_id: str) -> dict:
    """Run the 3-stage grounding pipeline for one ICD code. Returns a result dict."""
    logger.info("Grounding ICD-%s code %s for patient %s", icd_version, icd_code, patient_id)

    # ════════════════════════════════════════════════════════════════════
    # Stage 1 — Exact API match
    # ════════════════════════════════════════════════════════════════════
    exact = _icd_client.lookup(icd_code, icd_version)
    if exact:
        log_grounding_attempt(
            patient_id=patient_id, icd_code=icd_code, icd_version=icd_version,
            stage_matched=1, similarity_score=None,
            verification_token=exact["token"],
        )
        return {
            "stage": 1, "icd_code": icd_code, "icd_version": icd_version,
            "title": exact.get("title"), "similarity_score": None,
            "verification_token": exact["token"], "status": "VERIFIED",
        }

    # ════════════════════════════════════════════════════════════════════
    # Stage 2 — Similarity match (τ > 0.85) via sentence-transformers
    # ════════════════════════════════════════════════════════════════════
    candidates = _icd_client.search_concept(icd_code, icd_version, top_k=5)
    if candidates:
        embedder = _get_embedder()
        query_vec = embedder.encode(icd_code).tolist()
        best_score = 0.0
        best_candidate = None
        for cand in candidates:
            title = cand.get("title") or cand.get("code", "")
            cand_vec = embedder.encode(title).tolist()
            score = _cosine(query_vec, cand_vec)
            if score > best_score:
                best_score = score
                best_candidate = cand

        if best_score > _SIMILARITY_THRESHOLD and best_candidate:
            log_grounding_attempt(
                patient_id=patient_id, icd_code=icd_code, icd_version=icd_version,
                stage_matched=2, similarity_score=best_score,
                verification_token=best_candidate["token"],
            )
            return {
                "stage": 2, "icd_code": icd_code, "icd_version": icd_version,
                "title": best_candidate.get("title"),
                "matched_code": best_candidate.get("code"),
                "similarity_score": round(best_score, 4),
                "verification_token": best_candidate["token"], "status": "VERIFIED",
            }

    # ════════════════════════════════════════════════════════════════════
    # Stage 3 — LLM fallback, then verify against API
    # ════════════════════════════════════════════════════════════════════
    try:
        from google import genai as google_genai
        client = google_genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
        prompt = (
            f"You are a clinical coding expert. Provide the standard medical concept name "
            f"for ICD-{icd_version} code '{icd_code}'. Reply with only the concept name, "
            f"nothing else."
        )
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite", contents=prompt
        )
        concept_name = response.text.strip()

        if concept_name:
            verified = _icd_client.search_concept(concept_name, icd_version, top_k=1)
            if verified and verified[0].get("token"):
                cand = verified[0]
                log_grounding_attempt(
                    patient_id=patient_id, icd_code=icd_code, icd_version=icd_version,
                    stage_matched=3, similarity_score=None,
                    verification_token=cand["token"],
                )
                return {
                    "stage": 3, "icd_code": icd_code, "icd_version": icd_version,
                    "title": cand.get("title"), "llm_concept": concept_name,
                    "similarity_score": None,
                    "verification_token": cand["token"], "status": "VERIFIED",
                }
    except Exception as exc:
        logger.error("Stage 3 LLM fallback failed for %s: %s", icd_code, exc)

    # ════════════════════════════════════════════════════════════════════
    # All stages failed
    # ════════════════════════════════════════════════════════════════════
    log_grounding_attempt(
        patient_id=patient_id, icd_code=icd_code, icd_version=icd_version,
        stage_matched=None, similarity_score=None, verification_token=None,
    )
    return {
        "stage": "failed", "icd_code": icd_code, "icd_version": icd_version,
        "similarity_score": None, "verification_token": None, "status": "UNVERIFIED",
    }


class MedicalKnowledgeLookup(BaseTool):
    name: str = "medical_knowledge_lookup"
    description: str = (
        "Ground a single ICD code against the authoritative knowledge base. "
        "Input format: JSON string with fields 'icd_code', 'icd_version' (9 or 10), "
        "and 'patient_id'. Returns a Verification Token for the matched code."
    )

    def _run(self, input_str: str) -> str:
        try:
            params = json.loads(input_str)
            icd_code = str(params["icd_code"]).strip()
            icd_version = str(params["icd_version"]).strip()
            patient_id = str(params.get("patient_id", "unknown"))
        except Exception:
            icd_code = str(input_str).strip().replace("'", "").replace('"', "")
            icd_version = "10"
            patient_id = "unknown"

        return json.dumps(_ground_single_code(icd_code, icd_version, patient_id))


class BatchMedicalKnowledgeLookup(BaseTool):
    name: str = "batch_medical_knowledge_lookup"
    description: str = (
        "Ground ALL ICD codes for a patient in a single call. "
        "Input: JSON with 'patient_id' (str) and 'codes' (list of objects, each with "
        "'icd_code' and 'icd_version'). "
        "Returns a grounding table: one result per code with stage, verification_token, and status. "
        "Use this instead of calling medical_knowledge_lookup repeatedly."
    )

    def _run(self, input_str: str) -> str:
        try:
            params = json.loads(input_str)
            patient_id = str(params.get("patient_id", "unknown"))
            codes: list[dict] = params["codes"]
        except Exception as exc:
            return json.dumps({"error": f"Invalid input: {exc}"})

        results = []
        for entry in codes:
            icd_code = str(entry.get("icd_code", "")).strip()
            icd_version = str(entry.get("icd_version", "10")).strip()
            if not icd_code:
                continue
            results.append(_ground_single_code(icd_code, icd_version, patient_id))

        verified = [r for r in results if r["status"] == "VERIFIED"]
        unverified = [r for r in results if r["status"] == "UNVERIFIED"]
        logger.info(
            "Batch grounding complete for patient %s: %d verified, %d unverified",
            patient_id, len(verified), len(unverified),
        )
        return json.dumps({
            "patient_id": patient_id,
            "total": len(results),
            "verified_count": len(verified),
            "unverified_count": len(unverified),
            "grounding_table": results,
        })
