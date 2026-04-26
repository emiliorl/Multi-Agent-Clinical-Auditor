import os
from crewai import Agent, LLM
from src.tools import EHRPatternScanner

# 1. Fetch API Key
api_key = os.getenv("GEMINI_API_KEY")

# 2. Configure LLM using native wrapper (Fixes 'BaseLLM' validation error)
clinical_llm = LLM(
    model="gemini/gemini-2.5-flash-lite", 
    api_key=api_key,
    temperature=0.1
)

# 3. Define the Agent
# We use the instance of our new tool class here
diagnostician = Agent(
    role='Lead Clinical Data Miner',
    goal='Identify patient trajectories and sepsis triggers from raw EHR data',
    backstory='Expert in pattern discovery and sequence mining from clinical records.',
    tools=[EHRPatternScanner()], # Passing the class instance
    llm=clinical_llm,
    verbose=True,
    allow_delegation=False
)