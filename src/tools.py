import pandas as pd
from langchain.tools import tool

class ClinicalDataTools:
    @tool("EHR_Trajectory_Miner")
    def mine_patient_data(patient_id: str):
        """Extracts the 48-hour clinical trajectory for a specific patient 
        from gzipped EHR files. Essential for detecting Sepsis patterns."""
        
        # 1. Connect to the 'vitals' path from our mapper
        path = "/content/drive/MyDrive/MIMIC_IV_DATA/icu/chartevents.csv.gz"
        
        # 2. Logic: Search for the patient_id (subject_id in MIMIC)
        # We use a chunked reader to save Colab RAM
        relevant_data = []
        for chunk in pd.read_csv(path, compression='gzip', chunksize=10000):
            match = chunk[chunk['subject_id'] == int(patient_id)]
            if not match.empty:
                relevant_data.append(match)
        
        # 3. Format the 'Trajectory' for the Agent
        if relevant_data:
            df = pd.concat(relevant_data)
            summary = df[['charttime', 'valuenum', 'label']].to_string()
            return f"Patient {patient_id} Trajectory:\n{summary}"
        
        return "No trajectory data found for this ID."