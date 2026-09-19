import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# How many web search results to pull per lead during enrichment
SEARCH_RESULTS_PER_LEAD = 4

# Default score (1-10) a lead must meet to get an outreach draft generated
DEFAULT_SCORE_THRESHOLD = 6

# Pause between LLM calls to stay under free-tier rate limits (seconds)
try:
    SECONDS_BETWEEN_LLM_CALLS = int(os.getenv("SECONDS_BETWEEN_LLM_CALLS", "13"))
except ValueError:
    SECONDS_BETWEEN_LLM_CALLS = 13
