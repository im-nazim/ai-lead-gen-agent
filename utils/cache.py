import json
import os
from pathlib import Path
from typing import Optional

_CACHE_DIR = Path(__file__).parent.parent / ".cache"
_CACHE_FILE = _CACHE_DIR / "enrichment_cache.json"


def _cache_key(company_name: str, website: Optional[str]) -> str:
    return f"{company_name.strip().lower()}|{(website or '').strip().lower()}"


def _load_cache() -> dict:
    if not _CACHE_FILE.exists():
        return {}
    try:
        with open(_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def get_cached_enrichment(company_name: str, website: Optional[str]) -> Optional[dict]:
    """Returns a cached enrichment dict for this company, or None if not cached."""
    cache = _load_cache()
    return cache.get(_cache_key(company_name, website))


def save_enrichment(company_name: str, website: Optional[str], enrichment_dict: dict) -> None:
    """Saves an enrichment result so future runs skip the web search + LLM call for this company."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    cache = _load_cache()
    cache[_cache_key(company_name, website)] = enrichment_dict
    with open(_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def clear_cache() -> None:
    """Wipes the enrichment cache entirely — useful if a company's info has changed."""
    if _CACHE_FILE.exists():
        os.remove(_CACHE_FILE)
