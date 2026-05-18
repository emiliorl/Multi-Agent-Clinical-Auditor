import os
from dotenv import load_dotenv
from crewai import Agent, LLM
from src.tools import EHRPatternScanner, BatchMedicalKnowledgeLookup

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

clinical_llm = LLM(
    model="gemini/gemini-2.5-flash-lite",
    api_key=api_key,
    temperature=0.3,
    max_retries=5,
)

# Agent A — receives pre-computed trajectory; EHRPatternScanner kept for fallback use
diagnostician = Agent(
    role="Lead Clinical Data Miner",
    goal="Identify patient trajectories and sepsis triggers from raw EHR data",
    backstory="Expert in pattern discovery and sequence mining from clinical records.",
    tools=[EHRPatternScanner()],
    llm=clinical_llm,
    verbose=True,
    allow_delegation=False,
)

# Manager Agent — pure reasoning, no external tools
manager = Agent(
    role="Clinical Governance Manager",
    goal="Compute inter-agent agreement and enforce the Consensus Gate.",
    backstory=(
        "You do not diagnose. You receive two independent clinical assessments "
        "and determine whether they agree above κ > 0.80. If they do not, you "
        "produce a structured contradiction report for the Reflection Loop."
    ),
    tools=[],
    llm=clinical_llm,
    verbose=True,
    allow_delegation=False,
)

# Agent B — uses batch grounding; one tool call covers all codes
auditor = Agent(
    role="Clinical Audit Specialist",
    goal="Ensure 100% grounding of clinical findings against the authoritative ICD knowledge base.",
    backstory=(
        "You are a regulatory compliance agent operating under Clinical Chain-of-Thought (CCoT) "
        "reasoning rules. You have access to batch_medical_knowledge_lookup, which grounds ALL "
        "ICD codes in a single call — use it exactly once per audit task. "
        "After the batch call returns, record each code's stage and Verification Token. "
        "Codes with status=UNVERIFIED must be placed in a dedicated UNVERIFIED section. "
        "You are FORBIDDEN from reasoning about any code that lacks a Verification Token. "
        "You do not offer clinical opinions; you only verify grounding."
    ),
    tools=[BatchMedicalKnowledgeLookup()],
    llm=clinical_llm,
    verbose=True,
    allow_delegation=False,
)