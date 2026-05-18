import os
from dotenv import load_dotenv
from crewai import Agent, LLM
from src.tools import EHRPatternScanner, BatchMedicalKnowledgeLookup

load_dotenv()

_use_local = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
_clinical_model = os.getenv("CLINICAL_MODEL", "")

if _use_local or _clinical_model.startswith("openai/"):
    _model_str = _clinical_model or f"openai/{os.getenv('LM_STUDIO_MODEL', 'google/gemma-4-e4b')}"
    _lm_base = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
    # instructor (used by CrewAI for output_pydantic) reads these env vars directly
    os.environ.setdefault("OPENAI_API_KEY", "lm-studio")
    os.environ.setdefault("OPENAI_API_BASE", _lm_base)
    os.environ.setdefault("OPENAI_BASE_URL", _lm_base)
    clinical_llm = LLM(
        model=_model_str,
        base_url=_lm_base,
        api_key="lm-studio",
        temperature=0.3,
        max_retries=1,   # local LLM never rate-limits; retries just flood the server
        timeout=300,     # 5 min — CPU-bound inference needs headroom
    )
else:
    _model_str = _clinical_model or "gemini/gemini-2.5-flash"
    clinical_llm = LLM(
        model=_model_str,
        api_key=os.getenv("GEMINI_API_KEY"),
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