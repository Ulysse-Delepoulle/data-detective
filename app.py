"""Streamlit UI for DataDetective.

This is a thin layer over the agent. It holds no analysis logic: it collects a
question and a dataset, previews the data, builds the chosen backend, calls
run_agent, and renders the structured report plus the full reasoning transcript.

Run it with:  streamlit run app.py

Streamlit reruns this whole script top to bottom on every interaction, so the
last result is kept in st.session_state to survive those reruns.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

# Make the src layout importable when launched with "streamlit run app.py",
# which does not use the pytest pythonpath setting.
_SRC = Path(__file__).parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dotenv import load_dotenv

load_dotenv()  # make ANTHROPIC_API_KEY from .env available for the cloud backend

import pandas as pd
import requests
import streamlit as st

from datadetective.agent.graph import run_agent
from datadetective.llm.cache import CachingBackend
from datadetective.llm.claude_backend import ClaudeBackend
from datadetective.llm.ollama_backend import OllamaBackend

_SAMPLE_DIR = Path(__file__).parent / "data" / "sample_datasets"
_OLLAMA_HOST = "http://localhost:11434"


# --------------------------------------------------------------------------
# Small helpers (no agent logic, just UI support).
# --------------------------------------------------------------------------
def _ollama_is_up() -> bool:
    try:
        requests.get(f"{_OLLAMA_HOST}/api/tags", timeout=0.5)
        return True
    except requests.RequestException:
        return False


def _load_dataframe(path: str) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001 - preview is best effort
        return None


def _suggest_questions(df: pd.DataFrame) -> list[str]:
    """Build a few example questions from the columns, so the user has a
    starting point instead of a blank box."""
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    ideas: list[str] = []
    if categorical and numeric:
        ideas.append(f"What is the total {numeric[0]} by {categorical[0]}?")
        ideas.append(f"Which {categorical[0]} has the highest {numeric[0]}?")
    if len(numeric) >= 2:
        ideas.append(f"Is there a relationship between {numeric[0]} and {numeric[1]}?")
    elif numeric:
        ideas.append(f"What is the average {numeric[0]}?")
    return ideas[:3]


def _zebra(df: pd.DataFrame):
    """Return a Styler with alternating row shading and bright text, for a
    higher-contrast, easier-to-read table."""
    df = df.reset_index(drop=True)

    def shade(row):
        bg = "#1C2430" if row.name % 2 == 0 else "#141A22"
        return [f"background-color: {bg}; color: #F2F5FA"] * len(row)

    return df.style.apply(shade, axis=1)


def _column_overview(df: pd.DataFrame) -> pd.DataFrame:
    """A clean, single-dtype summary table (avoids the mixed-type Arrow
    warning that describe(include='all') produces)."""
    return pd.DataFrame(
        {
            "column": df.columns,
            "type": [str(df[c].dtype) for c in df.columns],
            "non-null": [int(df[c].notna().sum()) for c in df.columns],
            "unique": [int(df[c].nunique()) for c in df.columns],
        }
    )


# --------------------------------------------------------------------------
# Page setup and light styling.
# --------------------------------------------------------------------------
st.set_page_config(page_title="DataDetective", page_icon="🔎", layout="wide")
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
      :root {
        --accent: #4C84D6;
        --accent-soft: rgba(76,132,214,0.12);
        --accent-border: rgba(76,132,214,0.32);
        --card-bg: #151A21;
        --card-border: #232A33;
        --muted: #8A94A3;
      }
      /* Apply Inter via inheritance from the root only. Do NOT target icon
         spans: Streamlit icons are a ligature font, and overriding their
         font-family makes the ligature text (e.g. "upload", "arrow_drop_down")
         render as literal words instead of the icon glyph. */
      html, body, .stApp { font-family: 'Inter', -apple-system, system-ui, sans-serif; }
      h1, h2, h3, h4, h5, p, label, .stMarkdown, .stMarkdown p, .stMarkdown li {
        font-family: 'Inter', -apple-system, system-ui, sans-serif;
      }
      /* Keep Material icon ligatures rendering as icons, not text. */
      [data-testid="stIconMaterial"], .material-icons,
      .material-symbols-rounded, .material-symbols-outlined {
        font-family: 'Material Symbols Rounded', 'Material Symbols Outlined',
                     'Material Icons' !important;
      }
      .block-container { padding-top: 2rem; max-width: 1060px; }
      h1, h2, h3 { letter-spacing: -0.02em; }

      /* Hero banner: flat slate panel with a thin accent top rule */
      .hero {
        padding: 24px 28px; margin-bottom: 24px; border-radius: 14px;
        background: #12171E; border: 1px solid var(--card-border);
        border-top: 2px solid var(--accent);
      }
      .hero-title { font-size: 1.9rem; font-weight: 700; margin: 0; }
      .hero-sub { color: #C4CCD6; font-size: 1rem; margin-top: 6px; max-width: 70ch;
        line-height: 1.5; }
      .hero-badges { margin-top: 16px; display: flex; gap: 8px; flex-wrap: wrap; }
      .pill {
        font-size: 0.76rem; color: var(--muted); background: transparent;
        border: 1px solid var(--card-border); padding: 4px 10px; border-radius: 6px;
      }

      /* Section cards (st.container(border=True)) */
      div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--card-bg);
        border: 1px solid var(--card-border) !important;
        border-radius: 12px;
        padding: 20px 24px !important;
        margin-bottom: 18px;
      }

      /* Numbered section headers */
      .sec-header { display: flex; align-items: center; gap: 11px; margin-bottom: 6px; }
      .sec-num {
        display: inline-flex; align-items: center; justify-content: center;
        width: 26px; height: 26px; border-radius: 7px;
        background: var(--accent-soft); border: 1px solid var(--accent-border);
        color: var(--accent); font-weight: 600; font-size: 0.85rem;
      }
      .sec-title { font-size: 1.12rem; font-weight: 600; }

      /* Metrics: flat slate cards */
      div[data-testid="stMetric"] {
        background: #11161C; border: 1px solid var(--card-border);
        border-radius: 10px; padding: 14px 16px;
      }

      /* Primary button: solid accent, subtle hover */
      div[data-testid="stButton"] > button[kind="primary"] {
        background: var(--accent); border: none; font-weight: 600; border-radius: 8px;
      }
      div[data-testid="stButton"] > button[kind="primary"]:hover {
        background: #5B90DC;
      }

      /* Sidebar */
      section[data-testid="stSidebar"] {
        width: 320px !important;
        background: #11161C; border-right: 1px solid var(--card-border);
      }
      .finding {
        font-size: 1.3rem; font-weight: 600; line-height: 1.4;
        padding: 14px 16px; border-left: 3px solid var(--accent);
        background: var(--accent-soft); border-radius: 8px;
      }

      /* Higher-contrast frame around data tables */
      div[data-testid="stDataFrame"] {
        border: 1px solid var(--card-border); border-radius: 8px;
      }

      /* Compact sidebar labels and status chips */
      .side-label {
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em;
        text-transform: uppercase; color: var(--muted); margin: 2px 0 8px 2px;
      }
      .status {
        display: flex; align-items: center; gap: 8px;
        font-size: 0.8rem; color: #cfd4de; margin: 8px 2px 2px;
      }
      .status .dot {
        width: 8px; height: 8px; border-radius: 50%; flex: 0 0 auto;
        box-shadow: 0 0 8px currentColor;
      }
      .status .sub { color: var(--muted); }

      /* Run metadata row above the transcript */
      .run-meta {
        display: flex; flex-wrap: wrap; gap: 8px; margin: 2px 0 10px;
      }
      .chip-meta {
        font-size: 0.78rem; color: #cfd4de;
        background: rgba(255,255,255,0.05); border: 1px solid var(--card-border);
        padding: 4px 10px; border-radius: 999px;
      }
      .chip-meta.cache { color: var(--accent); background: var(--accent-soft);
        border-color: var(--accent-border); }
    </style>
    """,
    unsafe_allow_html=True,
)


def section_header(num: str, title: str) -> None:
    """A numbered, accent-colored section heading."""
    st.markdown(
        f'<div class="sec-header"><span class="sec-num">{num}</span>'
        f'<span class="sec-title">{title}</span></div>',
        unsafe_allow_html=True,
    )


def side_label(text: str) -> None:
    st.markdown(f'<div class="side-label">{text}</div>', unsafe_allow_html=True)


def status_chip(kind: str, text: str, sub: str = "") -> None:
    """A small colored-dot status line. kind is ok, warn, or err."""
    color = {"ok": "#22c55e", "warn": "#f59e0b", "err": "#ef4444"}[kind]
    sub_html = f'<span class="sub">· {sub}</span>' if sub else ""
    st.markdown(
        f'<div class="status"><span class="dot" style="color:{color};'
        f'background:{color}"></span><span>{text}</span> {sub_html}</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Sidebar: brand, backend picker with live status, run settings.
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🔎 DataDetective")
    st.caption("Autonomous data analyst")
    st.divider()

    side_label("Model backend")
    backend_choice = st.segmented_control(
        "Model backend",
        options=["Local", "Cloud"],
        default="Local",
        format_func=lambda x: f"💻 {x}" if x == "Local" else f"☁️ {x}",
        label_visibility="collapsed",
    )
    backend_choice = backend_choice or "Local"
    use_cloud = backend_choice == "Cloud"

    # Compact live status for the chosen backend.
    if use_cloud:
        if os.environ.get("ANTHROPIC_API_KEY"):
            status_chip("ok", "Claude ready", "API costs apply")
        else:
            status_chip("err", "No API key in .env")
    else:
        if _ollama_is_up():
            status_chip("ok", "Ollama running", "free and private")
        else:
            status_chip("warn", "Ollama offline on :11434")

    st.divider()
    side_label("Run settings")
    max_attempts = st.slider(
        "Max code attempts", 1, 5, 3,
        help="How many times the agent may fix its own code before giving up.",
    )


# --------------------------------------------------------------------------
# Header.
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="hero-title">🔎 DataDetective</div>
      <div class="hero-sub">Ask a question, get a verified report. The agent writes
      and runs its own analysis code in a sandbox, fixes its mistakes, and returns
      a structured answer.</div>
      <div class="hero-badges">
        <span class="pill">Sandboxed execution</span>
        <span class="pill">Self-correcting</span>
        <span class="pill">Local or cloud model</span>
        <span class="pill">Structured report</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Section 1: dataset choice.
# --------------------------------------------------------------------------
with st.container(border=True):
    section_header("1", "Dataset")
    samples = (
        sorted(p.name for p in _SAMPLE_DIR.glob("*.csv"))
        if _SAMPLE_DIR.exists()
        else []
    )
    # Two separate tabs so only one input shows at a time.
    tab_sample, tab_upload = st.tabs(["Bundled sample", "Upload your own"])
    with tab_sample:
        sample_name = st.selectbox(
            "Choose a dataset", ["(none)"] + samples, label_visibility="collapsed"
        )
    with tab_upload:
        uploaded = st.file_uploader(
            "Upload a CSV", type=["csv"], label_visibility="collapsed"
        )

    dataset_path: str | None = None
    if uploaded is not None:
        # The sandbox mounts a file path, so write the upload to a temp file.
        tmp = Path(tempfile.gettempdir()) / f"dd_upload_{uploaded.name}"
        tmp.write_bytes(uploaded.getbuffer())
        dataset_path = str(tmp)
    elif sample_name != "(none)":
        dataset_path = str(_SAMPLE_DIR / sample_name)

    # Preview so the user knows what the data contains before asking.
    preview_df: pd.DataFrame | None = None
    if dataset_path:
        preview_df = _load_dataframe(dataset_path)

    if preview_df is None and dataset_path:
        st.warning("Could not read that file as a CSV.")
    elif preview_df is not None:
        st.write("")
        m1, m2, m3 = st.columns(3)
        m1.metric("Rows", f"{len(preview_df):,}")
        m2.metric("Columns", f"{preview_df.shape[1]}")
        m3.metric("Numeric columns",
                  sum(pd.api.types.is_numeric_dtype(preview_df[c])
                      for c in preview_df.columns))
        st.dataframe(_zebra(preview_df.head(10)), width="stretch", hide_index=True)
        with st.expander("Column types and summary"):
            st.markdown("**Columns**")
            st.dataframe(_zebra(_column_overview(preview_df)),
                         width="stretch", hide_index=True)
            numeric_cols = preview_df.select_dtypes("number")
            if not numeric_cols.empty:
                st.markdown("**Numeric summary**")
                stats = numeric_cols.describe().round(2).transpose().reset_index()
                stats = stats.rename(columns={"index": "column"})
                st.dataframe(_zebra(stats), width="stretch", hide_index=True)


# --------------------------------------------------------------------------
# Section 2: question.
# --------------------------------------------------------------------------
with st.container(border=True):
    section_header("2", "Question")

    # Example questions derived from the dataset. Clicking one fills the box.
    if preview_df is not None:
        ideas = _suggest_questions(preview_df)
        if ideas:
            picked = st.pills("Need ideas? Try one", ideas, default=None)
            if picked and picked != st.session_state.get("_last_pick"):
                st.session_state["question"] = picked
                st.session_state["_last_pick"] = picked

    question = st.text_area(
        "Business question",
        key="question",
        placeholder="e.g. Which product has the highest total revenue?",
        height=90,
    )

    ready = bool(dataset_path) and bool(question.strip())
    run = st.button(
        "▶  Run analysis", type="primary", disabled=not ready,
        width="stretch",
    )
    if not ready:
        st.caption("Pick a dataset and type a question to enable the run button.")


# --------------------------------------------------------------------------
# Run the agent on click, storing the result so it survives reruns.
# --------------------------------------------------------------------------
if run:
    if use_cloud and not os.environ.get("ANTHROPIC_API_KEY"):
        st.error("No ANTHROPIC_API_KEY found. Add it to your .env to use Claude.")
    else:
        inner = ClaudeBackend() if use_cloud else OllamaBackend()
        backend = CachingBackend(inner)
        with st.spinner("Writing code and running it in the sandbox..."):
            try:
                start = time.monotonic()
                st.session_state["final"] = run_agent(
                    question,
                    dataset_path,
                    backend=backend,
                    max_attempts=max_attempts,
                )
                st.session_state["run_meta"] = {
                    "duration": time.monotonic() - start,
                    "hits": backend.hits,
                    "misses": backend.misses,
                    "backend": "Claude" if use_cloud else "Ollama",
                }
                st.toast("Analysis complete", icon="✅")
            except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
                st.session_state["final"] = None
                st.session_state.pop("run_meta", None)
                st.error(f"Run failed: {exc}")


# --------------------------------------------------------------------------
# Section 3: render the stored result.
# --------------------------------------------------------------------------
final = st.session_state.get("final")
if final:
    report = final.get("report")

    with st.container(border=True):
        section_header("3", "Report")
        if report is None:
            st.warning("No structured report was produced.")
            st.text(final.get("answer", ""))
        else:
            st.markdown(f'<div class="finding">{report.finding}</div>',
                        unsafe_allow_html=True)
            st.write("")
            left, right = st.columns(2)
            with left:
                st.markdown("**Supporting numbers**")
                for number in report.supporting_numbers:
                    st.markdown(f"- {number}")
            with right:
                st.markdown("**Caveats**")
                if report.caveats:
                    for caveat in report.caveats:
                        st.markdown(f"- {caveat}")
                else:
                    st.markdown("_None stated._")
            st.divider()
            st.markdown("**Method**")
            st.write(report.method)
            for chart in report.chart_files:
                if Path(chart).exists():
                    st.image(chart)

    # The reasoning transcript: every attempt, in order.
    with st.container(border=True):
        section_header("4", "Reasoning transcript")

        meta = st.session_state.get("run_meta")
        if meta:
            calls = meta["misses"]            # misses are the real model calls
            cached = meta["hits"]             # hits were served from disk
            chips = [
                f'<span class="chip-meta">{final.get("attempts")} attempt(s)</span>',
                f'<span class="chip-meta">{meta["duration"]:.1f}s</span>',
                f'<span class="chip-meta">{meta["backend"]}</span>',
                f'<span class="chip-meta">{calls} model call(s)</span>',
            ]
            if cached:
                chips.append(
                    f'<span class="chip-meta cache">{cached} served from cache</span>'
                )
            st.markdown(f'<div class="run-meta">{"".join(chips)}</div>',
                        unsafe_allow_html=True)
            if cached and not calls:
                st.caption(
                    "This exact run was answered entirely from cache, so it was "
                    "instant and the attempts match the first run."
                )
        for record in final.get("history", []):
            ok = record.exit_code == 0 and not record.timed_out
            icon = "✅" if ok else "❌"
            label = f"{icon}  Attempt {record.attempt}"
            with st.expander(label, expanded=not ok):
                st.code(record.code, language="python")
                if record.stdout.strip():
                    st.text("stdout:\n" + record.stdout)
                if not ok and record.stderr.strip():
                    st.text("stderr:\n" + record.stderr)
