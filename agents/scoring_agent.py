from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from config import GEMINI_API_KEY, GEMINI_MODEL, SECONDS_BETWEEN_LLM_CALLS
from models import EnrichedLead, ScoredLead, ScoringResult
from utils.rate_limiter import throttled_invoke

_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a sales qualification assistant. Score how well a lead fits "
     "the given Ideal Customer Profile (ICP) on a 1-10 scale. "
     "10 = perfect fit worth prioritizing today, 1 = clearly not a fit. "
     "Be a harsh, honest grader — most leads should land in the 3-7 range. "
     "Base the score only on the lead data provided, not assumptions. "
     "If a field like company_size is 'unknown' or an estimate, do not treat "
     "that as a strike against the lead — score based on the fields you do "
     "have (industry, description, pain points) and note the missing data in "
     "your reasoning rather than letting it drag the score down."),
    ("human",
     "Ideal Customer Profile:\n{icp}\n\n"
     "Lead to score:\n"
     "Company: {company_name}\n"
     "Industry: {industry}\n"
     "Size: {company_size}\n"
     "Description: {description}\n"
     "Likely pain points: {likely_pain_points}\n\n"
     "Score this lead's fit against the ICP."),
])


def _get_llm():
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0.1,
    )
    return llm.with_structured_output(ScoringResult)


def score_lead(lead: EnrichedLead, icp: str) -> ScoredLead:
    """
    Scores an enriched lead 1-10 against a free-text ICP description,
    with a short justification the user can sanity-check.
    """
    chain = _PROMPT | _get_llm()
    result: ScoringResult = throttled_invoke(chain, {
        "icp": icp,
        "company_name": lead.company_name,
        "industry": lead.industry,
        "company_size": lead.company_size,
        "description": lead.description,
        "likely_pain_points": lead.likely_pain_points,
    }, SECONDS_BETWEEN_LLM_CALLS)

    return ScoredLead(
        **lead.model_dump(),
        fit_score=result.fit_score,
        score_reasoning=result.reasoning,
    )
