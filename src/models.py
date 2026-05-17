from typing import Literal
from pydantic import BaseModel


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
