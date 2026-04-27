import os
from crewai import Agent, LLM
from src.tools import EHRPatternScanner, MedicalKnowledgeLookup

# 1. Fetch API Key
api_key = os.getenv("GEMINI_API_KEY")

# 2. Configure LLM using native wrapper (Fixes 'BaseLLM' validation error)
clinical_llm = LLM(
    model="gemini/gemini-2.5-flash-lite", 
    api_key=api_key,
    temperature=0.1
)

# Agent A
diagnostician = Agent(
    role='Lead Clinical Data Miner',
    goal='Identify patient trajectories and sepsis triggers from raw EHR data',
    backstory='Expert in pattern discovery and sequence mining from clinical records.',
    tools=[EHRPatternScanner()], # Passing the class instance
    llm=clinical_llm,
    verbose=True,
    allow_delegation=False
)

# Agent B 
auditor = Agent(
    role='Clinical Audit Specialist',
    goal='Ensure 100% grounding of clinical findings against the Knowledge Graph.',
    backstory=(
        "You are a regulatory compliance agent. You do not offer opinions; you only "
        "verify links between raw data and the Knowledge Graph. You are rewarded "
        "for finding cases where Agent A's codes do not match the KG."
    ),
    tools=[MedicalKnowledgeLookup()],
    llm=clinical_llm, 
    max_iter=15,       # Allow enough turns to call the tool for multiple codes
    verbose=True,
    allow_delegation=False # Keep it focused on the audit task
)