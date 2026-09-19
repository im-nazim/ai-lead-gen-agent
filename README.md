# AI Lead Generation & Sales Agent

A multi-agent pipeline that takes a CSV of company names, enriches each one
with web research, scores it against your Ideal Customer Profile (ICP), and
drafts a personalized outreach email for every lead that qualifies.

## Pipeline

```
CSV of companies
      │
      ▼
Enrichment Agent   → web search + Gemini → industry, size, pain points, contact
      │
      ▼
Scoring Agent      → Gemini scores 1-10 fit against your ICP
      │
      ▼
Outreach Agent     → drafts subject + email for leads above your score threshold
      │
      ▼
Streamlit UI       → review scores, read drafts, export CSV
```

This is a v1 scoped deliberately around **CSV input** rather than live
scraping (LinkedIn/Apollo scraping is fragile and ToS-risky). It's the safer,
faster-to-ship version of this project — you can swap in a paid data source
(Apollo.io, Hunter.io, Clearbit) later by replacing `leads_from_csv` with an
API call that returns the same `Lead` objects.

## Setup

```bash
cd ai-lead-gen-agent
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env and paste your Gemini API key
# (free key: https://aistudio.google.com/apikey)
```

## Run

```bash
streamlit run app.py
```

Then in the browser:
1. Fill in your ICP and offer in the **Setup** tab.
2. Upload `sample_leads.csv` (included) to test, or your own list.
3. Click **Run pipeline**.
4. Review scored leads, expand qualified ones to read/copy outreach drafts,
   download the full results as CSV.

## File structure

```
ai-lead-gen-agent/
├── app.py                    # Streamlit UI
├── orchestrator.py           # Runs leads through the full pipeline
├── config.py                 # API key / model / threshold settings
├── models.py                 # Pydantic schemas for every stage
├── agents/
│   ├── enrichment_agent.py   # Web search + Gemini → structured company data
│   ├── scoring_agent.py      # Gemini → 1-10 ICP fit score
│   └── outreach_agent.py     # Gemini → personalized email draft
├── utils/
│   └── search.py             # DuckDuckGo search wrapper
├── sample_leads.csv
└── requirements.txt
```

## Notes / next steps

- **Cost control**: each lead makes 1 web search + 3 Gemini calls (enrich,
  score, and outreach if qualified). For large lists, consider batching or
  caching enrichment results by company name.
- **Swapping the LLM**: everything routes through `ChatGoogleGenerativeAI` in
  each agent file — swap for `ChatOpenAI` or `ChatAnthropic` if a client
  wants a different provider.
- **Real lead sourcing**: to go from "enrich a list" to "find the list",
  plug an Apollo.io or Hunter.io API call in front of `leads_from_csv` that
  returns a `list[Lead]` — nothing downstream needs to change.
- **CRM sync**: add a `utils/crm.py` that pushes `FinalLead` rows to
  Airtable/HubSpot/Google Sheets after the pipeline runs.
