import io
from typing import Callable, List, Optional

import pandas as pd

from models import Lead, FinalLead, ScoredLead
from agents.enrichment_agent import enrich_lead
from agents.scoring_agent import score_lead
from agents.outreach_agent import draft_outreach
from config import DEFAULT_SCORE_THRESHOLD

RESULT_COLUMNS = [
    "company_name", "website", "industry", "company_size", "confidence", "from_cache",
    "fit_score", "score_reasoning", "contact_person", "contact_title", "description",
    "likely_pain_points", "outreach_subject", "outreach_email",
]


def _optional_text(value) -> Optional[str]:
    """None for missing/NaN/blank cells, else the stripped string."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _is_rate_limit(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(k in text for k in ("429", "resourceexhausted", "quota", "rate limit"))


def leads_from_csv(file) -> List[Lead]:
    """
    Reads an uploaded CSV (needs at least a 'company_name' column).
    Skips rows with no company name, drops duplicate companies (case-insensitive),
    and accepts both UTF-8 (with or without BOM) and Excel's default Windows encoding.
    """
    raw = file.read() if hasattr(file, "read") else open(file, "rb").read()
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    if not raw.strip():
        raise ValueError("The CSV file is empty. It needs a header row with a 'company_name' column.")

    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")  # Excel's default "CSV" export

    try:
        df = pd.read_csv(io.StringIO(text))
    except pd.errors.EmptyDataError:
        raise ValueError("The CSV file has no columns. It needs a header row with a 'company_name' column.")

    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    if "company_name" not in df.columns:
        raise ValueError("CSV must contain a 'company_name' column")

    leads: List[Lead] = []
    seen = set()
    for _, row in df.iterrows():
        name = _optional_text(row.get("company_name"))
        if not name:
            continue  # blank row
        key = name.lower()
        if key in seen:
            continue  # duplicate company: don't pay to enrich it twice
        seen.add(key)
        leads.append(Lead(
            company_name=name,
            website=_optional_text(row.get("website")),
            notes=_optional_text(row.get("notes")),
        ))
    return leads


def run_pipeline(
    leads: List[Lead],
    icp: str,
    offer: str,
    score_threshold: int = DEFAULT_SCORE_THRESHOLD,
    use_cache: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    errors: Optional[list] = None,
) -> List[ScoredLead]:
    """
    Runs each lead through enrich -> score -> (outreach if above threshold).
    Returns a list of ScoredLead / FinalLead objects, sorted by fit_score desc.

    A failing lead no longer throws away the rest of the run: it is skipped and
    recorded in `errors` (a list you pass in; gets (company, message) tuples).
    On a rate-limit/quota error the run stops early (further calls would fail
    too) and returns the leads finished so far. If EVERY lead fails, the last
    exception is raised so the caller can show it.
    progress_callback(done_count, total_count, current_company) lets a UI
    show live progress; it's optional so this also works headless.
    """
    total = len(leads)
    results: List[ScoredLead] = []
    last_exc: Optional[Exception] = None

    for i, lead in enumerate(leads, start=1):
        if progress_callback:
            progress_callback(i - 1, total, lead.company_name)
        try:
            enriched = enrich_lead(lead, use_cache=use_cache)
            scored = score_lead(enriched, icp)
            if scored.fit_score >= score_threshold:
                results.append(draft_outreach(scored, offer))
            else:
                results.append(scored)
        except Exception as exc:
            last_exc = exc
            if errors is not None:
                errors.append((lead.company_name, str(exc)))
            if _is_rate_limit(exc):
                if errors is not None:
                    for rest in leads[i:]:
                        errors.append((rest.company_name, "not processed (rate limit reached)"))
                break

    if not results and last_exc is not None:
        raise last_exc

    if progress_callback:
        progress_callback(total, total, "done")

    results.sort(key=lambda l: l.fit_score, reverse=True)
    return results


def results_to_dataframe(results: List[ScoredLead]) -> pd.DataFrame:
    """Flattens pipeline output into a CSV-friendly dataframe (always has every column)."""
    rows = []
    for r in results:
        row = {
            "company_name": r.company_name,
            "website": r.website,
            "industry": r.industry,
            "company_size": r.company_size,
            "confidence": r.confidence,
            "from_cache": r.from_cache,
            "fit_score": r.fit_score,
            "score_reasoning": r.score_reasoning,
            "contact_person": r.contact_person,
            "contact_title": r.contact_title,
            "description": r.description,
            "likely_pain_points": r.likely_pain_points,
        }
        if isinstance(r, FinalLead):
            row["outreach_subject"] = r.outreach_subject
            row["outreach_email"] = r.outreach_email
        else:
            row["outreach_subject"] = ""
            row["outreach_email"] = ""
        rows.append(row)
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)
