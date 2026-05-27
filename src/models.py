from typing import Literal
from pydantic import BaseModel, Field, field_validator


class IcdEntry(BaseModel):
    code: str
    version: Literal["9", "10"]
    description: str | None = None


class Admission(BaseModel):
    hadm_id: int
    admit_time: str
    admit_type: str
    diagnoses: list[IcdEntry]


class PatientTrajectory(BaseModel):
    patient_id: str
    admissions: list[Admission]
    icd_codes: list[IcdEntry]   # Flat deduplicated list for grounding
    timeline: list[str]         # Chronologically ordered event labels


_MAX_CODES = 15


class AgentDiagnosis(BaseModel):
    patient_id: str
    diagnosis_hypothesis: str
    icd_codes_cited: list[str] = Field(default_factory=list)
    evidence_chain: list[str] = Field(
        default_factory=list,
        description="One entry per cited code — exactly the hadm/admit/type/code/desc template",
    )
    confidence_score: float = Field(ge=0.0, le=1.0)
    unverified_codes: list[str] = Field(default_factory=list, description="Codes that came back stage=failed")

    @field_validator("icd_codes_cited", "evidence_chain", mode="before")
    @classmethod
    def _truncate(cls, v: object) -> object:
        if isinstance(v, list) and len(v) > _MAX_CODES:
            return v[:_MAX_CODES]
        return v
