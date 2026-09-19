import html

import streamlit as st
 
from config import DEFAULT_SCORE_THRESHOLD, GEMINI_API_KEY
from orchestrator import leads_from_csv, run_pipeline, results_to_dataframe
from models import FinalLead


def _esc(value) -> str:
    """Escape model/web-derived text before it is rendered as HTML or markdown."""
    return html.escape(str(value or "")).replace("$", "&#36;")
 
st.set_page_config(
    page_title="AI Lead Gen & Sales Agent",
    page_icon="🎯",
    layout="wide",
)
 
# ---------------------------------------------------------------------------
# Minimal supporting CSS (colors/theme now come from .streamlit/config.toml)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; max-width: 1150px; }
 
    .app-header {
        display: flex;
        align-items: center;
        gap: 0.9rem;
        margin-bottom: 0.2rem;
    }
    .app-header .icon-badge {
        background: linear-gradient(135deg, #818CF8, #6366F1);
        width: 48px; height: 48px;
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.4rem;
        flex-shrink: 0;
    }
    .app-header h1 { margin: 0; font-size: 1.65rem; font-weight: 700; }
    .app-subtitle { color: #94A3B8; margin: 0.1rem 0 1.6rem 62px; font-size: 0.95rem; }
 
    div[data-testid="stMetric"] {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem 1.2rem;
    }
 
    .draft-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-left: 4px solid #818CF8;
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        margin-top: 0.5rem;
        max-width: 100%;
        overflow-wrap: break-word;
    }
 
    .streamlit-expanderContent {
        overflow-wrap: break-word;
        word-wrap: break-word;
    }
 
    div.stButton > button[kind="primary"] {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.55rem 1.6rem;
    }
</style>
""", unsafe_allow_html=True)
 
# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="app-header">
    <div class="icon-badge">🎯</div>
    <h1>AI Lead Generation &amp; Sales Agent</h1>
</div>
<p class="app-subtitle">Upload a company list → auto-enrich → score against your ICP → get drafted outreach.</p>
""", unsafe_allow_html=True)
 
if not GEMINI_API_KEY:
    st.warning("No GEMINI_API_KEY found. Add it to a .env file (see .env.example) before running the pipeline.")
 
# ---------------------------------------------------------------------------
# Tabs: Setup / Results
# ---------------------------------------------------------------------------
setup_tab, results_tab = st.tabs(["⚙️  Setup", "📊  Results"])
 
with setup_tab:
    col_icp, col_offer = st.columns(2)
    with col_icp:
        st.markdown("**1. Ideal Customer Profile**")
        icp = st.text_area(
            "Describe who you're targeting",
            placeholder="e.g. B2B SaaS companies, 10-200 employees, no in-house data team, "
                        "actively hiring for growth/marketing roles.",
            height=150,
            label_visibility="collapsed",
        )
    with col_offer:
        st.markdown("**2. Your offer**")
        offer = st.text_area(
            "What are you selling / offering?",
            placeholder="e.g. We build custom AI automation workflows that cut manual "
                        "reporting time by 80%, starting at $1,500/mo.",
            height=150,
            label_visibility="collapsed",
        )
 
    st.markdown("**3. Settings**")
    s1, s2 = st.columns([2, 1])
    with s1:
        threshold = st.slider("Minimum fit score to draft outreach", 1, 10, DEFAULT_SCORE_THRESHOLD)
    with s2:
        use_cache = st.checkbox(
            "Use cached enrichment",
            value=True,
            help="Skips web search + LLM call for a company you've already enriched before.",
        )
 
    st.markdown("**4. Upload leads**")
    st.caption("CSV needs a `company_name` column. `website` and `notes` columns are optional but help enrichment.")
    uploaded = st.file_uploader("Lead list (CSV)", type=["csv"], label_visibility="collapsed")
 
    ready = bool(uploaded and icp.strip() and offer.strip() and GEMINI_API_KEY)
    run = st.button("Run pipeline", type="primary", disabled=not ready)
 
    if uploaded and not (icp.strip() and offer.strip()):
        st.info("Add an ICP and an offer above before running.")
 
    if run:
        leads = None
        try:
            leads = leads_from_csv(uploaded)
        except ValueError as e:
            st.error(str(e))

        if leads is not None and not leads:
            st.warning("No usable rows found - every row was blank or had no company_name.")
        elif leads:
            st.write(f"Loaded **{len(leads)}** leads. Running enrichment → scoring → outreach...")
            progress_bar = st.progress(0)
            status = st.empty()

            def _progress(done, total, company):
                progress_bar.progress(done / total if total else 0)
                status.write(f"Processing: **{company}** ({done}/{total})")

            results = None
            failed = []
            try:
                results = run_pipeline(
                    leads, icp=icp, offer=offer, score_threshold=threshold,
                    use_cache=use_cache, progress_callback=_progress, errors=failed,
                )
            except Exception as e:
                status.empty()
                progress_bar.empty()
                error_text = str(e)
                if "429" in error_text or "ResourceExhausted" in error_text or "quota" in error_text.lower():
                    st.error(
                        "The AI service is temporarily rate-limited. Please wait a minute and "
                        "try running the pipeline again."
                    )
                else:
                    st.error(
                        "Something went wrong while processing leads. Please try again, and if "
                        "this keeps happening, check that your API key and settings are correct."
                    )
                with st.expander("Technical details"):
                    st.code(error_text)

            if results is not None:
                if failed:
                    st.warning(
                        f"{len(failed)} lead(s) were skipped because of errors. "
                        "The leads that finished are in the Results tab."
                    )
                    with st.expander("Skipped leads"):
                        for name, err in failed:
                            st.write(f"**{name}** — {err}")
                status.write("✅ Done — see the Results tab.")
                st.session_state["results"] = results
                st.session_state["df"] = results_to_dataframe(results)
 
# ---------------------------------------------------------------------------
# Results tab
# ---------------------------------------------------------------------------
with results_tab:
    if "df" not in st.session_state or not st.session_state.get("results"):
        st.caption("Run the pipeline from the Setup tab to see results here.")
    else:
        df = st.session_state["df"]
        results = st.session_state["results"]
 
        qualified = sum(1 for r in results if isinstance(r, FinalLead))
        avg_score = round(sum(r.fit_score for r in results) / len(results), 1) if results else 0
        cached_count = sum(1 for r in results if r.from_cache)
 
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total leads", len(results))
        c2.metric("Qualified for outreach", qualified)
        c3.metric("Average fit score", f"{avg_score}/10")
        c4.metric("Served from cache", cached_count)
 
        st.markdown("#### Lead breakdown")
 
        def _score_bg(val):
            try:
                score = int(val)
            except (ValueError, TypeError):
                return ""
            if score >= 7:
                return "background-color: #14532D; color: #BBF7D0;"
            if score >= 4:
                return "background-color: #713F12; color: #FDE68A;"
            return "background-color: #7F1D1D; color: #FECACA;"
 
        def _confidence_bg(val):
            v = str(val).lower()
            if v == "high":
                return "background-color: #14532D; color: #BBF7D0;"
            if v == "medium":
                return "background-color: #713F12; color: #FDE68A;"
            if v == "low":
                return "background-color: #7F1D1D; color: #FECACA;"
            return ""
 
        display_cols = ["company_name", "fit_score", "confidence", "industry", "company_size", "score_reasoning"]
        display_df = df[display_cols].rename(columns={
            "company_name": "Company",
            "fit_score": "Score",
            "confidence": "Confidence",
            "industry": "Industry",
            "company_size": "Size",
            "score_reasoning": "Why",
        })
 
        styled = (
            display_df.style
            .map(_score_bg, subset=["Score"])
            .map(_confidence_bg, subset=["Confidence"])
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)
 
        st.download_button(
            "⬇ Download full results as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="lead_gen_results.csv",
            mime="text/csv",
        )
 
        st.markdown("#### ✉️ Outreach drafts")
        qualified_leads = [r for r in results if isinstance(r, FinalLead)]
        if not qualified_leads:
            st.caption("No leads cleared the fit-score threshold this run.")
        for r in qualified_leads:
            with st.expander(f"{r.company_name}  —  {r.fit_score}/10  ·  {r.confidence} confidence"):
                if (r.confidence or "").lower() == "low":
                    st.warning(
                        "Low enrichment confidence — the personal detail in this draft may be "
                        "a guess rather than a verified fact. Double-check before sending."
                    )
                with st.container(border=True):
                    st.markdown(f"**Why it qualified:** {_esc(r.score_reasoning)}")
                    st.markdown(f"**Subject:** {_esc(r.outreach_subject)}")
                    email_html = _esc(r.outreach_email).replace("\n", "<br>")
                    st.markdown(
                        f'<div style="white-space: normal; word-wrap: break-word; '
                        f'overflow-wrap: break-word; line-height: 1.55; text-align: justify;">{email_html}</div>',
                        unsafe_allow_html=True,
                    )