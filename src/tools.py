import json
import os
from threading import Lock
import pandas as pd
from crewai.tools import BaseTool
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
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

_LOCAL_DICT_FILENAME = "d_icd_diagnoses.csv.gz"
_local_icd_dict: "dict[tuple[str,str], str] | None" = None
_local_dict_lock = Lock()


def _get_local_dict() -> "dict[tuple[str,str], str]":
    """Lazily load {(icd_code, icd_version): long_title} from the MIMIC ICD dictionary."""
    global _local_icd_dict
    if _local_icd_dict is None:
        with _local_dict_lock:
            if _local_icd_dict is None:
                dict_path = os.path.join(_data_dir(), _LOCAL_DICT_FILENAME)
                if os.path.exists(dict_path):
                    try:
                        df = pd.read_csv(dict_path, compression="gzip", dtype=str)
                        d: dict[tuple[str, str], str] = {}
                        for _, row in df.iterrows():
                            code = str(row["icd_code"]).strip()
                            try:
                                version = str(int(float(str(row["icd_version"]))))
                            except (ValueError, TypeError):
                                version = str(row["icd_version"]).strip()
                            title = str(row.get("long_title", "")).strip()
                            d[(code, version)] = title
                        _local_icd_dict = d
                        logger.info("Loaded local ICD dictionary: %d entries", len(d))
                    except Exception as exc:
                        logger.warning("Could not load local ICD dictionary: %s", exc)
                        _local_icd_dict = {}
                else:
                    logger.warning("Local ICD dictionary not found at %s", dict_path)
                    _local_icd_dict = {}
    return _local_icd_dict

_TRANSIENT_TAGS = ("429", "RESOURCE_EXHAUSTED", "rate limit", "quota",
                   "503", "UNAVAILABLE", "overloaded",
                   "Invalid response from LLM call", "None or empty")


def _is_transient(exc: BaseException) -> bool:
    return any(tag in str(exc) for tag in _TRANSIENT_TAGS)


def _build_stage3_kwargs() -> dict:
    """Return litellm.completion kwargs for the Stage 3 grounding call."""
    if os.getenv("USE_LOCAL_LLM", "false").lower() == "true":
        return {
            "model": f"openai/{os.getenv('LM_STUDIO_MODEL', 'google/gemma-4-e4b')}",
            "base_url": os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
            "api_key": "lm-studio",
        }
    grounding_model = os.getenv("GROUNDING_MODEL", "gemini-2.5-flash-lite")
    return {
        "model": f"gemini/{grounding_model}",
        "api_key": os.getenv("GEMINI_API_KEY", ""),
    }


@retry(
    retry=retry_if_exception(_is_transient),
    wait=wait_exponential(multiplier=2, min=30, max=120),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _stage3_generate(prompt: str) -> str:
    import litellm
    kwargs = _build_stage3_kwargs()
    response = litellm.completion(
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    return response.choices[0].message.content.strip()
import concurrent.futures

_embedder: "SentenceTransformer | None" = None
_embedder_lock = Lock()


def _get_embedder() -> "SentenceTransformer":
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _cosine(a: list[float], b: list[float]) -> float:
    import numpy as np
    va = np.array(a).reshape(1, -1)
    vb = np.array(b).reshape(1, -1)
    return float(cosine_similarity(va, vb)[0][0])


def normalize_code(c: str) -> str:
    return str(c).strip().replace(".", "").lower()


def _ground_single_code(icd_code: str, icd_version: str, patient_id: str) -> dict:
    """Run the 3-stage grounding pipeline for one ICD code. Returns a result dict."""
    logger.info("Grounding ICD-%s code %s for patient %s", icd_version, icd_code, patient_id)

    # ════════════════════════════════════════════════════════════════════
    # Stage 0 — Local MIMIC ICD dictionary (zero network calls)
    # ════════════════════════════════════════════════════════════════════
    local_dict = _get_local_dict()
    local_title = local_dict.get((icd_code, icd_version))
    if local_title is not None:
        token = f"LOCAL_SIG_{icd_code.replace('.', '_')}"
        log_grounding_attempt(
            patient_id=patient_id, icd_code=icd_code, icd_version=icd_version,
            stage_matched=0, similarity_score=None,
            verification_token=token,
        )
        return {
            "stage": 0, "icd_code": icd_code, "icd_version": icd_version,
            "title": local_title, "similarity_score": None,
            "verification_token": token, "status": "VERIFIED",
        }

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
        # First, try to find an exact normalized code match among search candidates.
        # This is extremely fast, avoids SentenceTransformer overhead, and avoids Stage 3 LLM fallbacks.
        normalized_query = normalize_code(icd_code)
        for cand in candidates:
            cand_code = cand.get("code", "")
            if cand_code and normalize_code(cand_code) == normalized_query:
                log_grounding_attempt(
                    patient_id=patient_id, icd_code=icd_code, icd_version=icd_version,
                    stage_matched=2, similarity_score=1.0,
                    verification_token=cand["token"],
                )
                return {
                    "stage": 2, "icd_code": icd_code, "icd_version": icd_version,
                    "title": cand.get("title"),
                    "matched_code": cand.get("code"),
                    "similarity_score": 1.0,
                    "verification_token": cand["token"], "status": "VERIFIED",
                }

        # Fallback to SentenceTransformer semantic similarity if exact code match fails.
        try:
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
        except Exception as exc:
            logger.error("Stage 2 semantic similarity failed for %s: %s", icd_code, exc)

    # ════════════════════════════════════════════════════════════════════
    # Stage 3 — LLM fallback, then verify against API
    # ════════════════════════════════════════════════════════════════════
    try:
        prompt = (
            f"You are a clinical coding expert. Provide the standard medical concept name "
            f"for ICD-{icd_version} code '{icd_code}'. Reply with only the concept name, "
            f"nothing else."
        )
        concept_name = _stage3_generate(prompt)

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
        "Action Input MUST use this exact format: "
        "{\"input_str\": \"{\\\"patient_id\\\": \\\"PATIENT_ID\\\", \\\"codes\\\": "
        "[{\\\"icd_code\\\": \\\"CODE\\\", \\\"icd_version\\\": \\\"9\\\"},...]}\"} "
        "Returns a grounding table with stage, verification_token, and status per code. "
        "Call EXACTLY ONCE — do not repeat for individual codes."
    )

    def _run(self, input_str: str) -> str:
        try:
            params = json.loads(input_str)
            # Some models wrap the payload: {"input_str": "{...}"} — unwrap it
            if "input_str" in params and "codes" not in params:
                params = json.loads(params["input_str"])
            patient_id = str(params.get("patient_id", "unknown"))
            codes: list[dict] = params["codes"]
        except Exception as exc:
            return json.dumps({"error": f"Invalid input: {exc}"})

        # Deduplicate code entries in the batch to avoid redundant grounding tasks
        seen = set()
        unique_codes = []
        for entry in codes:
            icd_code = str(entry.get("icd_code", "")).strip()
            icd_version = str(entry.get("icd_version", "10")).strip()
            if not icd_code:
                continue
            key = (icd_code, icd_version)
            if key not in seen:
                seen.add(key)
                unique_codes.append((icd_code, icd_version))

        results = []
        # Since _ground_single_code is heavily I/O-bound (NLM/WHO API calls & LLM fallback),
        # running grounding in parallel via a ThreadPoolExecutor significantly reduces overall latency.
        max_workers = min(10, len(unique_codes)) if unique_codes else 1
        if max_workers > 0:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_code = {
                    executor.submit(_ground_single_code, code, ver, patient_id): (code, ver)
                    for code, ver in unique_codes
                }
                for future in concurrent.futures.as_completed(future_to_code):
                    code, ver = future_to_code[future]
                    try:
                        res = future.result()
                        results.append(res)
                    except Exception as exc:
                        logger.error("Failed to ground code %s (v%s) in parallel: %s", code, ver, exc)
                        results.append({
                            "stage": "failed", "icd_code": code, "icd_version": ver,
                            "similarity_score": None, "verification_token": None, "status": "UNVERIFIED",
                            "error": str(exc),
                        })

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
