from ddgs import DDGS
from config import SEARCH_RESULTS_PER_LEAD


def web_search(query: str, max_results: int = SEARCH_RESULTS_PER_LEAD) -> str:
    """
    Runs a DuckDuckGo search and returns a flattened text block of
    title + snippet + url per result, ready to hand to an LLM as context.
    Fails soft: returns an empty string instead of raising, so a bad
    lookup never kills the whole pipeline run.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return f"[search failed: {e}]"

    if not results:
        return "[no search results found]"

    chunks = []
    for r in results:
        title = r.get("title", "")
        body = r.get("body", "")
        href = r.get("href", "")
        chunks.append(f"- {title}: {body} ({href})")
    return "\n".join(chunks)
