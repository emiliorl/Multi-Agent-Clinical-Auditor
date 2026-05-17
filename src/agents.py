import os
from dotenv import load_dotenv
from crewai import Agent, LLM
from src.tools import EHRPatternScanner, MedicalKnowledgeLookup

# Load environment variables from .env
load_dotenv()

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
    max_rpm=15,                 # Pacing requests to avoid 429 rate limits
    verbose=True,
    allow_delegation=False
)

# Agent B
auditor = Agent(
    role='Clinical Audit Specialist',
    goal='Ensure 100% grounding of clinical findings against the authoritative ICD knowledge base.',
    backstory=(
        "You are a regulatory compliance agent operating under Clinical Chain-of-Thought (CCoT) "
        "reasoning rules. For every ICD code you encounter, you MUST call medical_knowledge_lookup "
        "with a JSON payload: {\"icd_code\": \"<code>\", \"icd_version\": \"<9 or 10>\", "
        "\"patient_id\": \"<id>\"}. "
        "After each lookup, you MUST explicitly state: "
        "'Stage X matched at similarity Y — Verification Token: <token>'. "
        "Codes whose lookup returns status=UNVERIFIED MUST be listed in a dedicated UNVERIFIED "
        "section. You are FORBIDDEN from reasoning about or accepting any code that lacks a "
        "Verification Token. You do not offer clinical opinions; you only verify grounding."
    ),
    tools=[MedicalKnowledgeLookup()],
    llm=clinical_llm,
    max_iter=25,
    max_rpm=15,                 # Pacing requests to avoid 429 rate limits
    verbose=True,
    allow_delegation=False,
)