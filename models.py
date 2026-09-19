from typing import Optional
from pydantic import BaseModel, Field


class Lead(BaseModel):
    """Raw input lead, as provided by the user via CSV."""
    company_name: str
    website: Optional[str] = None
    notes: Optional[str] = None


class EnrichedLead(Lead):
    """Lead after the enrichment agent has filled in company context."""
    industry: Optional[str] = None
    company_size: Optional[str] = None
    description: Optional[str] = None
    likely_pain_points: Optional[str] = None
    contact_person: Optional[str] = None
    contact_title: Optional[str] = None
    source_snippets: Optional[str] = None
    confidence: Optional[str] = None
    from_cache: bool = False


class ScoredLead(EnrichedLead):
    """Lead after the scoring agent has evaluated fit against the ICP."""
    fit_score: int = Field(ge=1, le=10)
    score_reasoning: str


class FinalLead(ScoredLead):
    """Lead after outreach copy has been generated (only for leads above threshold)."""
    outreach_subject: Optional[str] = None
    outreach_email: Optional[str] = None


# --- Structured-output schemas the LLM is asked to fill in directly ---

class EnrichmentResult(BaseModel):
    industry: str = Field(description="Best-guess industry/vertical of the company")
    company_size: str = Field(description="Rough size estimate, e.g. '1-10', '50-200', 'unknown'")
    description: str = Field(description="1-2 sentence plain description of what the company does")
    likely_pain_points: str = Field(description="1-2 likely pain points this company has, relevant to a B2B seller")
    contact_person: str = Field(default="unknown", description="Name of a likely decision-maker if found, else 'unknown'")
    contact_title: str = Field(default="unknown", description="Title of that decision-maker if found, else 'unknown'")
    confidence: str = Field(description="'High' if search results directly named specific facts, 'Medium' if mostly inferred, 'Low' if thin/generic.")


class ScoringResult(BaseModel):
    fit_score: int = Field(ge=1, le=10, description="1=terrible fit, 10=perfect fit for the ICP")
    reasoning: str = Field(description="1-2 sentence justification for the score")


class OutreachResult(BaseModel):
    subject: str = Field(description="Short, non-spammy email subject line")
    email_body: str = Field(description="Personalized outreach email, 80-130 words, no fluff, one clear CTA")
