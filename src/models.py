from typing import Literal
from pydantic import BaseModel, Field


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


class AgentDiagnosis(BaseModel):
    patient_id: str
    diagnosis_hypothesis: str
    icd_codes_cited: list[str]
    evidence_chain: list[str] = Field(description="Each item is one cited row from the trajectory")
    confidence_score: float = Field(ge=0.0, le=1.0)
    unverified_codes: list[str] = Field(default_factory=list, description="Codes that came back stage=failed")
