from concurrent.futures import ThreadPoolExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from config import GEMINI_API_KEY, GEMINI_MODEL, SECONDS_BETWEEN_LLM_CALLS
from models import Lead, EnrichedLead, EnrichmentResult
from utils.search import web_search
from utils.cache import get_cached_enrichment, save_enrichment
from utils.rate_limiter import throttled_invoke

_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a B2B research assistant. Given raw web search snippets about a "
     "company, extract structured facts. For company_size specifically: if the "
     "exact headcount isn't stated, make a reasonable estimate from context "
     "clues (funding stage, product maturity, how the company is described) "
     "and label it as an estimate, e.g. 'estimate: 11-50' rather than defaulting "
     "to 'unknown' — a rough estimate is far more useful downstream than no "
     "answer at all. For every other field, don't invent specifics you can't "
     "support from the snippets — say 'unknown' only when there's truly no "
     "basis to infer from. Be concise and factual, no marketing language. "
     "Also assess your own confidence: 'High' if the snippets directly named "
     "concrete facts (a real headcount, a real named person, funding figures), "
     "'Medium' if you mostly inferred from general context, 'Low' if the "
     "snippets were thin or generic and most fields are best-guesses."),
    ("human",
     "Company name: {company_name}\n"
     "Website: {website}\n"
     "User notes: {notes}\n\n"
     "General company search results:\n{general_snippets}\n\n"
     "Size / funding search results:\n{size_snippets}\n\n"
     "Leadership / contact search results:\n{contact_snippets}\n\n"
     "Extract the requested fields from this information."),
])


def _get_llm():
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0.2,
    )
    return llm.with_structured_output(EnrichmentResult)


def _run_searches_in_parallel(lead):
    """
    Runs the three enrichment web searches concurrently instead of one
    after another. They're independent I/O calls, so this cuts the
    wall-clock time for the search step roughly 2-3x with no change in
    data quality or API usage.
    """
    queries = [
        f"{lead.company_name} {lead.website or ''} company about",
        f"{lead.company_name} number of employees funding round size",
        f"{lead.company_name} founder CEO OR head of operations LinkedIn",
    ]
    with ThreadPoolExecutor(max_workers=3) as executor:
        general_snippets, size_snippets, contact_snippets = executor.map(web_search, queries)
    return general_snippets, size_snippets, contact_snippets


def enrich_lead(lead: Lead, use_cache: bool = True) -> EnrichedLead:
    """
    Looks up a company via three targeted searches (general context, size/
    funding, leadership/contact) and asks Gemini to turn the combined
    snippets into structured fields, including a confidence rating.
    Checks the local cache first so re-running the same company across
    sessions doesn't burn API calls and rate-limit budget again.
    """
    if use_cache:
        cached = get_cached_enrichment(lead.company_name, lead.website)
        if cached:
            return EnrichedLead(**{**cached, "from_cache": True})

    general_snippets, size_snippets, contact_snippets = _run_searches_in_parallel(lead)

    chain = _PROMPT | _get_llm()
    result: EnrichmentResult = throttled_invoke(chain, {
        "company_name": lead.company_name,
        "website": lead.website or "unknown",
        "notes": lead.notes or "none",
        "general_snippets": general_snippets,
        "size_snippets": size_snippets,
        "contact_snippets": contact_snippets,
    }, SECONDS_BETWEEN_LLM_CALLS)

    enriched = EnrichedLead(
        company_name=lead.company_name,
        website=lead.website,
        notes=lead.notes,
        industry=result.industry,
        company_size=result.company_size,
        description=result.description,
        likely_pain_points=result.likely_pain_points,
        contact_person=result.contact_person,
        contact_title=result.contact_title,
        confidence=result.confidence,
        source_snippets=f"{general_snippets}\n\n{size_snippets}\n\n{contact_snippets}",
        from_cache=False,
    )

    if use_cache:
        save_enrichment(lead.company_name, lead.website, enriched.model_dump())

    return enriched
