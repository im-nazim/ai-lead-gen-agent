from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from config import GEMINI_API_KEY, GEMINI_MODEL, SECONDS_BETWEEN_LLM_CALLS
from models import ScoredLead, FinalLead, OutreachResult
from utils.rate_limiter import throttled_invoke

_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You write short, personalized cold outreach emails for a B2B seller. "
     "Style rules: no generic flattery ('I noticed your impressive company'), "
     "no buzzwords, no exclamation marks, one specific detail about the lead "
     "worked into the first line, one clear low-friction CTA (e.g. '15-min call?'). "
     "80-130 words. Sound like a person, not a template. "
     "CRITICAL: never invent case studies, client names, project outcomes, "
     "metrics, or results the seller did not provide in their offer "
     "description. Only reference proof points (past results, numbers, "
     "client types) that are explicitly stated in the offer text below. If "
     "the offer text has no concrete proof point to cite, describe the "
     "capability in general terms instead of fabricating a success story — "
     "e.g. 'I build automation workflows that typically save teams several "
     "hours a week' rather than a specific invented client anecdote. "
     "IMPORTANT — respect the enrichment confidence level given below: if "
     "confidence is 'Low', do NOT open with a specific claimed fact about "
     "the company (a named person, a specific product detail, a specific "
     "event) since it may be wrong — instead open with something grounded "
     "in their industry/description, which is safer. Only use a sharp, "
     "specific personal detail in the opening line when confidence is "
     "'High' or 'Medium'."),
    ("human",
     "What we sell / offer:\n{offer}\n\n"
     "Lead:\n"
     "Company: {company_name}\n"
     "Industry: {industry}\n"
     "Description: {description}\n"
     "Likely pain points: {likely_pain_points}\n"
     "Contact: {contact_person} ({contact_title})\n"
     "Enrichment confidence: {confidence}\n"
     "Why they scored well: {score_reasoning}\n\n"
     "Write the outreach email."),
])


def _get_llm():
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0.7,
    )
    return llm.with_structured_output(OutreachResult)


def draft_outreach(lead: ScoredLead, offer: str) -> FinalLead:
    """
    Generates a subject line + email body personalized to this lead,
    grounded in the enrichment + scoring data already gathered. Respects
    the enrichment confidence level to avoid stating a specific fact that
    might be wrong.
    """
    chain = _PROMPT | _get_llm()
    result: OutreachResult = throttled_invoke(chain, {
        "offer": offer,
        "company_name": lead.company_name,
        "industry": lead.industry,
        "description": lead.description,
        "likely_pain_points": lead.likely_pain_points,
        "contact_person": lead.contact_person,
        "contact_title": lead.contact_title,
        "confidence": lead.confidence or "Low",
        "score_reasoning": lead.score_reasoning,
    }, SECONDS_BETWEEN_LLM_CALLS)

    return FinalLead(
        **lead.model_dump(),
        outreach_subject=result.subject,
        outreach_email=result.email_body,
    )
