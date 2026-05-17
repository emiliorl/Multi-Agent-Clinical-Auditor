"""
ICDApiClient — dual-source ICD code lookup.
  ICD-9  : NLM Clinical Tables API (no auth)
  ICD-10 : WHO ICD API (OAuth2, id.who.int)
Includes SQLite response cache, OAuth2 token auto-refresh, and exponential backoff.
"""

import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from threading import Lock

import requests
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from src.logger import get_logger

load_dotenv()  # ensure .env is loaded regardless of import order
logger = get_logger(__name__)

# ── constants ────────────────────────────────────────────────────────────────

_TOKEN_URL = "https://icdaccessmanagement.who.int/connect/token"
_NLM_SEARCH_URL = "https://clinicaltables.nlm.nih.gov/api/icd9cm_dx/v3/search"
_WHO_ICD10_BASE = "{primary}/icd/release/10/{code}"
_WHO_SEARCH_URL = "{primary}/icd/entity/search"
_WHO_HEADERS = {
    "API-Version": "v2",
    "Accept-Language": "en",
}
_CACHE_DB = os.path.join("data", "icd_cache.db")


# ── SQLite cache ─────────────────────────────────────────────────────────────

def _init_cache(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS icd_cache (
            icd_code    TEXT NOT NULL,
            icd_version TEXT NOT NULL,
            lookup_type TEXT NOT NULL,
            term        TEXT,
            response    TEXT NOT NULL,
            cached_at   TEXT NOT NULL,
            PRIMARY KEY (icd_code, icd_version, lookup_type, term)
        )
        """
    )
    conn.commit()
    return conn


# ── client ───────────────────────────────────────────────────────────────────

class ICDApiClient:
    def __init__(self) -> None:
        self._client_id = os.getenv("ICD_CLIENT_ID", "")
        self._client_secret = os.getenv("ICD_CLIENT_SECRET", "")
        self._primary = os.getenv("ICD_API_PRIMARY", "https://id.who.int")
        self._fallback = os.getenv("ICD_API_FALLBACK", "")

        self._token: str | None = None
        self._token_expires_at: float = 0.0
        self._token_lock = Lock()

        self._db = _init_cache(_CACHE_DB)
        self._db_lock = Lock()

    # ── OAuth2 ───────────────────────────────────────────────────────────────

    def get_token(self) -> str:
        with self._token_lock:
            if self._token and time.time() < self._token_expires_at - 60:
                return self._token

            resp = requests.post(
                _TOKEN_URL,
                data={
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "scope": "icdapi_access",
                    "grant_type": "client_credentials",
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            self._token = data["access_token"]
            self._token_expires_at = time.time() + data.get("expires_in", 3600)
            logger.info("WHO ICD token refreshed, expires in %ds", data.get("expires_in", 3600))
            return self._token

    # ── cache helpers ─────────────────────────────────────────────────────────

    def _cache_get(self, code: str, version: str, lookup_type: str, term: str | None) -> dict | None:
        with self._db_lock:
            row = self._db.execute(
                "SELECT response FROM icd_cache WHERE icd_code=? AND icd_version=? AND lookup_type=? AND term IS ?",
                (code, version, lookup_type, term),
            ).fetchone()
        if row:
            return json.loads(row[0])
        return None

    def _cache_set(self, code: str, version: str, lookup_type: str, term: str | None, data: dict) -> None:
        with self._db_lock:
            self._db.execute(
                """
                INSERT OR REPLACE INTO icd_cache
                    (icd_code, icd_version, lookup_type, term, response, cached_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (code, version, lookup_type, term, json.dumps(data),
                 datetime.now(timezone.utc).isoformat()),
            )
            self._db.commit()

    # ── ICD-9 via NLM Clinical Tables ─────────────────────────────────────────

    @retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(4))
    def _nlm_request(self, params: dict) -> list:
        resp = requests.get(_NLM_SEARCH_URL, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def lookup_icd9(self, code: str) -> dict | None:
        cached = self._cache_get(code, "9", "exact", None)
        if cached:
            return cached

        try:
            # Search without sf=code restriction — NLM sometimes returns empty
            # names for top-level codes when restricting to code field only.
            # Use terms={code} with df=code,name and pick the exact code match.
            data = self._nlm_request({"terms": code, "maxList": 10})
            results = data[3] if data and len(data) > 3 else []

            # Find the entry whose code matches exactly (strip dots for comparison)
            matched = None
            for entry in results:
                if entry[0].replace(".", "") == code.replace(".", ""):
                    matched = entry
                    break

            if not matched:
                return None

            result = {
                "code": matched[0],
                "title": matched[1],
                "source": "NLM",
                "token": f"NLM_SIG_{matched[0].replace('.', '_')}",
            }
            self._cache_set(code, "9", "exact", None, result)
            return result

        except Exception as exc:
            logger.error("NLM ICD-9 lookup failed for %s: %s", code, exc)
            return None

    def search_icd9(self, term: str, top_k: int = 5) -> list[dict]:
        cached = self._cache_get(term, "9", "search", term)
        if cached:
            return cached.get("results", [])

        try:
            data = self._nlm_request({"terms": term, "maxList": top_k})
            results_raw = data[3] if data and len(data) > 3 else []
            results = [
                {
                    "code": r[0],
                    "title": r[1],
                    "source": "NLM",
                    "token": f"NLM_SIG_{r[0].replace('.', '_')}",
                }
                for r in results_raw
            ]
            self._cache_set(term, "9", "search", term, {"results": results})
            return results

        except Exception as exc:
            logger.error("NLM ICD-9 search failed for %r: %s", term, exc)
            return []

    # ── ICD-10 via WHO ICD API ────────────────────────────────────────────────

    def _who_get(self, url: str) -> requests.Response:
        """Single WHO API GET with auth. Wrapped by retry at call sites."""
        headers = {**_WHO_HEADERS, "Authorization": f"Bearer {self.get_token()}"}
        resp = requests.get(url, headers=headers, timeout=15)
        if not resp.ok and self._fallback:
            fallback_url = url.replace(self._primary, self._fallback, 1)
            logger.warning("Primary WHO API returned %s, trying fallback", resp.status_code)
            resp = requests.get(fallback_url, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp

    @retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(4))
    def _who_get_with_retry(self, url: str) -> requests.Response:
        return self._who_get(url)

    def lookup_icd10(self, code: str) -> dict | None:
        cached = self._cache_get(code, "10", "exact", None)
        if cached:
            return cached

        url = _WHO_ICD10_BASE.format(primary=self._primary, code=code)
        try:
            resp = self._who_get_with_retry(url)
            data = resp.json()

            # @id is e.g. "http://id.who.int/icd/release/10/J44.1"
            # Use the code itself as the entity segment for ICD-10 (not numeric entity IDs)
            raw_id = data.get("@id", "")
            entity_segment = raw_id.rstrip("/").split("/")[-1] or code
            token = f"ENTITY_SIG_{entity_segment}"

            title_field = data.get("title", "")
            title = title_field.get("@value", "") if isinstance(title_field, dict) else str(title_field)

            result = {
                "code": code,
                "title": title,
                "source": "WHO_ICD10",
                "token": token,
                "entity_uri": raw_id,
            }
            self._cache_set(code, "10", "exact", None, result)
            return result

        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return None
            logger.error("WHO ICD-10 lookup failed for %s: %s", code, exc)
            return None
        except Exception as exc:
            logger.error("WHO ICD-10 lookup failed for %s: %s", code, exc)
            return None

    def search_icd10(self, term: str, top_k: int = 5) -> list[dict]:
        cached = self._cache_get(term, "10", "search", term)
        if cached:
            return cached.get("results", [])

        url = _WHO_SEARCH_URL.format(primary=self._primary)
        try:
            query = requests.utils.quote(term)
            resp = self._who_get_with_retry(
                f"{url}?q={query}&subtreeFilterUsage=includedChildrenOnly"
            )
            data = resp.json()
            destinations = data.get("destinationEntities", [])[:top_k]

            results = []
            for ent in destinations:
                raw_id = ent.get("id", "")
                entity_segment = raw_id.rstrip("/").split("/")[-1]
                token = f"ENTITY_SIG_{entity_segment}" if entity_segment else None
                results.append({
                    "code": ent.get("theCode", ""),
                    "title": ent.get("title", ""),
                    "source": "WHO_ICD10_SEARCH",
                    "token": token,
                    "entity_uri": raw_id,
                    "score": ent.get("score", 0.0),
                })

            self._cache_set(term, "10", "search", term, {"results": results})
            return results

        except Exception as exc:
            logger.error("WHO ICD-10 search failed for %r: %s", term, exc)
            return []

    # ── unified dispatch ──────────────────────────────────────────────────────

    def lookup(self, code: str, version: str) -> dict | None:
        if version == "10":
            return self.lookup_icd10(code)
        return self.lookup_icd9(code)

    def search_concept(self, term: str, version: str, top_k: int = 5) -> list[dict]:
        if version == "10":
            return self.search_icd10(term, top_k)
        return self.search_icd9(term, top_k)
