"""
QA suite for the AI Lead Gen & Sales Agent.  Run from the project folder:

    python qa_lead_gen.py

Safe: no network, no API key, writes no files. The real agents/ (Gemini + web search) are
replaced IN MEMORY by deterministic stand-ins, so this tests the code AROUND them:
CSV loading, orchestration, models, config and the Streamlit results UI.
"""
import io, os, sys, types, warnings, subprocess
sys.path.insert(0, os.getcwd())
os.environ["GEMINI_API_KEY"] = "test-key"

class _State: FAIL_ON=None; EMAIL="Hi there,\nWe help teams. Book a call?\n\nBest"
state=_State
import models as _m
def _mod(name, **attrs):
    m=types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name]=m; return m
def _enrich(lead, use_cache=True):
    if state.FAIL_ON and lead.company_name==state.FAIL_ON: raise RuntimeError("429 ResourceExhausted: quota exceeded")
    return _m.EnrichedLead(**lead.model_dump(), industry="SaaS", company_size="50-200", description="d", likely_pain_points="p", confidence="Medium")
def _score(e, icp): return _m.ScoredLead(**e.model_dump(), fit_score=(len(e.company_name)%10)+1, score_reasoning="stub")
def _draft(s, offer): return _m.FinalLead(**s.model_dump(), outreach_subject="Quick idea", outreach_email=state.EMAIL)
_pkg=_mod("agents"); _pkg.__path__=[]
_mod("agents.enrichment_agent", enrich_lead=_enrich); _mod("agents.scoring_agent", score_lead=_score); _mod("agents.outreach_agent", draft_outreach=_draft)
import pandas as pd
from pydantic import ValidationError
import models, orchestrator
from orchestrator import leads_from_csv, run_pipeline, results_to_dataframe
from models import Lead, ScoredLead, FinalLead

R=[]
def check(name, ok, detail=""):
    R.append(ok); print(("PASS  " if ok else "FAIL  ")+name+(f"\n        -> {detail}" if detail and not ok else ""))
def csv(text, enc="utf-8"): return io.BytesIO(text.encode(enc) if isinstance(text,str) else text)

print("\n### CSV LOADING (orchestrator.leads_from_csv)")
leads = leads_from_csv(open("sample_leads.csv","rb"))
check("sample_leads.csv loads", len(leads)>=4, f"{len(leads)} leads")
leads2 = leads_from_csv(open("better_test_leads.csv","rb"))
check("better_test_leads.csv loads", len(leads2)>=4, f"{len(leads2)} leads")
check("headers like 'Company Name' are normalised", leads_from_csv(csv("Company Name,Website\nAcme,https://a.com\n"))[0].company_name=="Acme")
try: leads_from_csv(csv("name,website\nAcme,x\n")); check("missing company_name -> clear ValueError", False)
except ValueError as e: check("missing company_name -> clear ValueError", "company_name" in str(e))
check("header-only CSV returns []", leads_from_csv(csv("company_name,website\n"))==[])
l = leads_from_csv(csv("company_name,website\n,https://x.com\nAcme,\n"))
check("blank company_name row is skipped (not turned into the word 'nan')", all(x.company_name.lower() not in ("nan","") for x in l),
      f"got company names: {[x.company_name for x in l]}")
l = leads_from_csv(csv("company_name\nAcme\nAcme\n"))
check("duplicate companies are de-duplicated (avoids paying twice)", len(l)==1, f"{len(l)} leads kept")
try:
    leads_from_csv(csv(""))
    check("empty file -> friendly error", False, "no error raised")
except Exception as e:
    check("empty file -> friendly error", "company_name" in str(e) or "empty" in str(e).lower(), f"{type(e).__name__}: {e}")
try:
    leads_from_csv(csv("company_name\nCaf\xe9 Ltd\n".encode("latin-1")))
    check("non-UTF8 (Excel 'CSV') file -> friendly error or handled", True)
except Exception as e:
    check("non-UTF8 (Excel 'CSV') file -> friendly error or handled", "encoding" in str(e).lower() or "utf-8" in str(e).lower() and "save" in str(e).lower(), f"{type(e).__name__}: {str(e)[:90]}")
check("UTF-8 BOM (Excel 'CSV UTF-8') handled", leads_from_csv(csv("\ufeffcompany_name\nAcme\n"))[0].company_name=="Acme")

print("\n### PIPELINE (orchestrator.run_pipeline, stub agents)")
pl=[Lead(company_name=n) for n in ("Al","Bobby","Carolina","Dx","Enterprise1")]
prog=[]
res=run_pipeline(pl, icp="i", offer="o", score_threshold=6, progress_callback=lambda d,t,c: prog.append((d,t,c)))
scores=[r.fit_score for r in res]
check("results sorted by score desc", scores==sorted(scores, reverse=True), str(scores))
check("outreach only for leads >= threshold", all((isinstance(r,FinalLead))==(r.fit_score>=6) for r in res))
check("progress callback: starts at 0, ends at total", prog[0][0]==0 and prog[-1][:2]==(5,5), str(prog))
for th in (1,10):
    r2=run_pipeline(pl,"i","o",score_threshold=th)
    check(f"threshold={th} boundary respected", all(isinstance(x,FinalLead)==(x.fit_score>=th) for x in r2))
state.FAIL_ON="Carolina"; part=[]
try:
    run_pipeline(pl,"i","o"); ok=False
except Exception as e: ok=True
check("one failing lead must not throw away the other leads' work (partial results kept)", False if ok else True,
      "lead #3 of 5 raised a 429 -> run_pipeline raised, so leads 1-2 (already paid for) are lost, 4-5 never ran")
state.FAIL_ON=None

print("\n### DATAFRAME / MODELS / CONFIG")
df=results_to_dataframe(res)
check("results_to_dataframe has expected columns", {"company_name","fit_score","outreach_email","confidence"}<=set(df.columns))
edf=results_to_dataframe([])
disp=["company_name","fit_score","confidence","industry","company_size","score_reasoning"]
try: edf[disp]; check("empty results -> dataframe still has the columns the UI indexes", True)
except KeyError as e: check("empty results -> dataframe still has the columns the UI indexes", False, f"KeyError {e}: app.py's df[display_cols] would crash the Results tab")
for bad in (0,11):
    try: ScoredLead(company_name="x", fit_score=bad, score_reasoning="r"); check(f"fit_score={bad} rejected by model", False)
    except ValidationError: check(f"fit_score={bad} rejected by model", True)
env=dict(os.environ, SECONDS_BETWEEN_LLM_CALLS="ten")
p=subprocess.run([sys.executable,"-c","import config"],env=env,capture_output=True,text=True,cwd=os.getcwd())
check("non-numeric SECONDS_BETWEEN_LLM_CALLS does not crash import", p.returncode==0, (p.stderr.strip().splitlines() or [""])[-1])
_src = open("app.py", encoding="utf-8").read()
check("app.py uses Styler.map (applymap is deprecated in pandas 2.2 and removed in 3.x)", ".applymap(" not in _src,
      "app.py still calls Styler.applymap")

print("\n### UI (Streamlit AppTest, streamlit==1.38.0)")
from streamlit.testing.v1 import AppTest
def app(): return AppTest.from_file(os.path.join(os.getcwd(),"app.py"), default_timeout=30)
at=app().run()
check("app loads with a key set, no exception", not at.exception, str(at.exception))
check("'Run pipeline' disabled until CSV+ICP+offer given", at.button[0].disabled)
os.environ["GEMINI_API_KEY"]=""
import importlib, config; importlib.reload(config)
at=app().run()
check("missing API key shows a warning", len(at.warning)>0)
os.environ["GEMINI_API_KEY"]="test-key"; importlib.reload(config)

fl=[FinalLead(company_name="Acme", fit_score=9, score_reasoning="great", confidence="Low", industry="SaaS", company_size="50",
              outreach_subject="Hello", outreach_email='Hi <img src=x onerror="alert(1)">\nprice is $1,500 or $2,000'),
    ScoredLead(company_name="Beta", fit_score=3, score_reasoning="poor", confidence="High", industry="X", company_size="5")]
at=app(); at.session_state["results"]=fl; at.session_state["df"]=results_to_dataframe(fl); at.run()
check("Results tab renders with data, no exception", not at.exception, str(at.exception))
mv={m.label:m.value for m in at.metric}
check("metrics correct (2 leads, 1 qualified, avg 6.0)", mv.get("Total leads")=="2" and mv.get("Qualified for outreach")=="1" and mv.get("Average fit score")=="6.0/10", str(mv))
check("low-confidence draft shows a warning", any("Low enrichment confidence" in w.value for w in at.warning))
md=[m.value for m in at.markdown]
check("LLM email text is HTML-escaped before rendering", not any('<img src=x' in v for v in md),
      "raw '<img ... onerror=...>' from model output is passed to st.markdown(unsafe_allow_html=True) unescaped")
check("draft card wrapper opens and closes in the SAME element", not (any(v.strip()=='<div class="draft-card">' for v in md) and any(v.strip()=='</div>' for v in md)),
      "opening <div class=draft-card> and closing </div> are separate st.markdown calls, so Streamlit never wraps the content in the card")
at=app(); at.session_state["results"]=[]; at.session_state["df"]=results_to_dataframe([]); at.run()
check("Results tab survives an empty result set", not at.exception, "; ".join(str(e.value)[:80] for e in at.exception))

print(f"\n{sum(R)}/{len(R)} passed")