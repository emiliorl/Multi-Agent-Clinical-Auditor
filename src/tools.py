import pandas as pd
from crewai.tools import BaseTool
from pydantic import Field

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