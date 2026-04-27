import pandas as pd
from crewai.tools import BaseTool
from pydantic import Field
import json
import os
from crewai.tools import BaseTool

class EHRPatternScanner(BaseTool):
    name: str = "EHRPatternScanner"
    description: str = (
        "Searches gzipped MIMIC-IV files for a specific patient's longitudinal trajectory. "
        "Useful for identifying sepsis triggers and vital sign patterns."
    )

    def _run(self, patient_id: str) -> str:
        # Path to your Drive folder
        path = "/content/drive/MyDrive/clinical_data_storage/mimic-iv-clinical-database-demo-2.2/hosp/diagnoses_icd.csv.gz"
        
        try:
            # Sequence Mining: Chunked reading for memory efficiency
            relevant_rows = []
            for chunk in pd.read_csv(path, compression='gzip', chunksize=10000):
                match = chunk[chunk['subject_id'] == int(patient_id)]
                if not match.empty:
                    relevant_rows.append(match)
            
            if relevant_rows:
                df = pd.concat(relevant_rows)
                return f"Trajectory for Patient {patient_id}:\n{df.to_string()}"
            return f"No records found for ID {patient_id} in {path}."
            
        except Exception as e:
            return f"Tool Error: {str(e)}"

class MedicalKnowledgeLookup(BaseTool):
    name: str = "medical_knowledge_lookup"
    description: str = "MANDATORY: Use this to fetch the Verification Token for an ICD code."

    def _run(self, icd_code: str) -> str:
        # Clean the input (stripping quotes or spaces the agent might add)
        clean_code = str(icd_code).strip().replace("'", "").replace('"', "")
        
        # Construct the key correctly
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
                    "protocol": node["relationships"]
                })
            return f"CRITICAL ERROR: Code {clean_code} NOT found in KG. Audit blocked."
        except Exception as e:
            return f"SYSTEM_FAILURE: {str(e)}"