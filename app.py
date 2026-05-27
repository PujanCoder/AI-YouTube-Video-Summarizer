"""
YouTube AI Summarizer - Main Application
=========================================
A production-grade Streamlit app that extracts YouTube transcripts
and summarizes them using LLMs (OpenAI / Groq) via LangChain.

Run: streamlit run app.py
"""

import streamlit as st
import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Local module imports
from .src.transcript import extract_video_id, get_transcript, format_transcript_text
from .src.summarizer import get_llm, chunk_transcript, generate_all_summaries
from .src.chains import build_qa_chain, ask_question
from .src.cache_manager import get_cached_result, save_to_cache, load_history, save_history_entry
from .src.export_utils import export_to_txt, export_to_pdf
from .src.utils import get_youtube_thumbnail, format_seconds, sanitize_filename

# ─────────────────────────────────────────────
# 1. PAGE CONFIG & THEME
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube AI Summarizer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS – modern card UI, consistent colors, dark-mode friendly
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Header banner ── */
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 1.5rem;
}
.main-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
.main-header p  { margin: 0.4rem 0 0; opacity: .85; font-size: 1rem; }

/* ── Cards ── */
.result-card {
    background: var(--background-color, #fff);
    border: 1px solid rgba(102,126,234,.25);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,.05);
}

/* ── Insight bullets ── */
.insight-item {
    display: flex; align-items: flex-start; gap: .6rem;
    padding: .55rem .7rem; margin-bottom: .4rem;
    border-radius: 8px; background: rgba(102,126,234,.07);
    border-left: 3px solid #667eea;
}

/* ── Timestamp chip ── */
.timestamp-chip {
    display: inline-block;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white; font-size: .72rem; font-weight: 600;
    padding: .15rem .55rem; border-radius: 20px; margin-right: .5rem;
}

/* ── TLDR box ── */
.tldr-box {
    background: linear-gradient(135deg, rgba(102,126,234,.12), rgba(118,75,162,.12));
    border: 2px solid rgba(102,126,234,.4);
    border-radius: 12px; padding: 1.2rem 1.5rem;
    font-size: 1.05rem; font-weight: 500; font-style: italic;
}

/* ── Keyword badge ── */
.keyword-badge {
    display: inline-block;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white; border-radius: 20px;
    padding: .25rem .75rem; margin: .2rem;
    font-size: .82rem; font-weight: 500;
}

/* ── Chat bubble ── */
.chat-user     { background:#667eea; color:#fff; border-radius:18px 18px 4px 18px;
                 padding:.7rem 1rem; margin:.4rem 0; max-width:80%; margin-left:auto; }
.chat-assistant{ background:rgba(102,126,234,.1); border-radius:18px 18px 18px 4px;
                 padding:.7rem 1rem; margin:.4rem 0; max-width:85%; }

/* ── Sidebar tweaks ── */
.sidebar-section { padding:.5rem 0; border-bottom:1px solid rgba(128,128,128,.15); margin-bottom:.8rem; }

/* ── Status pills ── */
.status-pill {
    display:inline-flex; align-items:center; gap:.35rem;
    padding:.25rem .75rem; border-radius:20px; font-size:.8rem; font-weight:600;
}
.pill-success { background:#d4edda; color:#155724; }
.pill-error   { background:#f8d7da; color:#721c24; }
.pill-info    { background:#d1ecf1; color:#0c5460; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 2. SESSION STATE INITIALISATION
# ─────────────────────────────────────────────
def init_session():
    defaults = {
        "transcript":      None,
        "transcript_raw":  None,
        "video_id":        None,
        "summaries":       None,
        "qa_chain":        None,
        "chat_history":    [],
        "processing":      False,
        "current_url":     "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()


# ─────────────────────────────────────────────
# 3. SIDEBAR – SETTINGS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    # ── API Provider ──
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    provider = st.selectbox(
        "🤖 LLM Provider",
        ["Groq (Free & Fast)", "OpenAI"],
        help="Choose your AI backend. Groq is free and very fast.",
    )
    provider_key = "groq" if "Groq" in provider else "openai"

    # Model selection based on provider
    if provider_key == "groq":
        model_options = ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma2-9b-it"]
    else:
        model_options = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
    model_name = st.selectbox("🧠 Model", model_options)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── API Key ──
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    if provider_key == "groq":
        api_key = st.text_input(
            "🔑 Groq API Key",
            value=os.getenv("GROQ_API_KEY", ""),
            type="password",
            help="Get a free key at console.groq.com",
        )
    else:
        api_key = st.text_input(
            "🔑 OpenAI API Key",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Summary Settings ──
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**📝 Summary Settings**")
    summary_length = st.select_slider(
        "Summary Length",
        options=["Brief", "Standard", "Detailed"],
        value="Standard",
    )
    temperature = st.slider(
        "🌡️ Creativity (Temperature)",
        min_value=0.0, max_value=1.0, value=0.3, step=0.05,
        help="Lower = more factual, Higher = more creative",
    )
    language = st.selectbox(
        "🌐 Transcript Language",
        ["en", "es", "fr", "de", "pt", "ja", "ko", "zh", "hi", "ar"],
        help="Preferred transcript language code",
    )
    extract_keywords = st.checkbox("🔑 Extract Keywords", value=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── History ──
    st.markdown("**🕘 Recent Summaries**")
    history = load_history()
    if history:
        for entry in reversed(history[-5:]):
            if st.button(f"▶ {entry['title'][:28]}…", key=entry["video_id"]):
                st.session_state.update({
                    "summaries":      entry["summaries"],
                    "video_id":       entry["video_id"],
                    "transcript":     entry.get("transcript", ""),
                    "current_url":    entry.get("url", ""),
                    "chat_history":   [],
                })
    else:
        st.caption("No history yet.")

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit + LangChain")


# ─────────────────────────────────────────────
# 4. MAIN CONTENT AREA
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎬 YouTube AI Summarizer</h1>
    <p>Paste any YouTube URL → get an instant AI-powered summary, insights, Q&A and more.</p>
</div>
""", unsafe_allow_html=True)

# ── URL Input ──
col_url, col_btn = st.columns([5, 1])
with col_url:
    youtube_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed",
    )
with col_btn:
    process_btn = st.button("▶ Summarize", type="primary", use_container_width=True)


# ─────────────────────────────────────────────
# 5. PROCESSING PIPELINE
# ─────────────────────────────────────────────
if process_btn and youtube_url:
    if not api_key:
        st.error("⚠️ Please enter your API key in the sidebar.")
        st.stop()

    with st.spinner(""):
        progress = st.progress(0, text="🔍 Extracting video ID…")

        try:
            # ── Step 1: Extract video ID ──
            video_id = extract_video_id(youtube_url)
            st.session_state["video_id"]    = video_id
            st.session_state["current_url"] = youtube_url
            progress.progress(15, text="📄 Fetching transcript…")

            # ── Step 2: Check cache ──
            cache_key = f"{video_id}_{model_name}_{summary_length}"
            cached    = get_cached_result(cache_key)

            if cached:
                st.session_state["summaries"]  = cached["summaries"]
                st.session_state["transcript"] = cached["transcript"]
                progress.progress(100, text="✅ Loaded from cache!")
                st.toast("⚡ Loaded from cache — no API cost!", icon="⚡")

            else:
                # ── Step 3: Get transcript ──
                transcript_raw, transcript_text = get_transcript(video_id, language)
                st.session_state["transcript_raw"] = transcript_raw
                st.session_state["transcript"]     = transcript_text
                progress.progress(35, text="✂️ Chunking text…")

                # ── Step 4: Initialise LLM ──
                llm  = get_llm(provider_key, model_name, temperature, api_key)
                docs = chunk_transcript(transcript_text)
                progress.progress(50, text="🤖 Generating summaries…")

                # ── Step 5: Generate all outputs ──
                summaries = generate_all_summaries(
                    llm, docs, transcript_raw, transcript_text,
                    summary_length, extract_keywords
                )
                st.session_state["summaries"] = summaries
                progress.progress(85, text="🔗 Building Q&A engine…")

                # ── Step 6: Build Q&A chain ──
                st.session_state["qa_chain"] = build_qa_chain(
                    llm, transcript_text
                )
                progress.progress(95, text="💾 Saving to cache…")

                # ── Step 7: Cache & history ──
                save_to_cache(cache_key, {"summaries": summaries, "transcript": transcript_text})
                save_history_entry({
                    "video_id":  video_id,
                    "url":       youtube_url,
                    "title":     summaries.get("title", video_id),
                    "summaries": summaries,
                    "transcript": transcript_text,
                    "timestamp": datetime.now().isoformat(),
                })
                progress.progress(100, text="✅ Done!")

            st.session_state["chat_history"] = []

        except Exception as e:
            st.error(f"❌ {e}")
            st.stop()


# ─────────────────────────────────────────────
# 6. RESULTS DISPLAY
# ─────────────────────────────────────────────
summaries = st.session_state.get("summaries")
video_id  = st.session_state.get("video_id")

if summaries:
    # ── Thumbnail + meta row ──
    thumb_col, meta_col = st.columns([1, 3])
    with thumb_col:
        thumb = get_youtube_thumbnail(video_id)
        if thumb:
            st.image(thumb, use_container_width=True)
    with meta_col:
        st.markdown(f"### {summaries.get('title', 'Video Summary')}")
        st.markdown(
            f'<span class="status-pill pill-success">✅ Summary Ready</span>'
            f'<span class="status-pill pill-info" style="margin-left:.5rem">🎬 {video_id}</span>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Tabs ──
    tabs = st.tabs(["📋 Summary", "💡 Key Insights", "⏱️ Topics", "⚡ TL;DR", "💬 Q&A", "🔑 Keywords"])

    # ── Tab 1: Summary ──
    with tabs[0]:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("#### 📋 Overall Summary")
        st.write(summaries.get("summary", "No summary generated."))
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Tab 2: Key Insights ──
    with tabs[1]:
        st.markdown("#### 💡 Key Insights")
        insights = summaries.get("insights", [])
        if insights:
            for insight in insights:
                st.markdown(
                    f'<div class="insight-item">• {insight}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No insights available.")

    # ── Tab 3: Topics with Timestamps ──
    with tabs[2]:
        st.markdown("#### ⏱️ Important Topics & Timestamps")
        topics = summaries.get("topics", [])
        if topics:
            for topic in topics:
                ts  = topic.get("timestamp", "")
                lbl = topic.get("topic", "")
                desc = topic.get("description", "")
                st.markdown(
                    f'<span class="timestamp-chip">{ts}</span> **{lbl}** — {desc}',
                    unsafe_allow_html=True,
                )
                st.markdown("")
        else:
            st.info("No topic timestamps extracted.")

    # ── Tab 4: TL;DR ──
    with tabs[3]:
        st.markdown("#### ⚡ TL;DR")
        st.markdown(
            f'<div class="tldr-box">💬 {summaries.get("tldr", "Not available.")}</div>',
            unsafe_allow_html=True,
        )

    # ── Tab 5: Q&A Chat ──
    with tabs[4]:
        st.markdown("#### 💬 Ask Anything About This Video")

        # Re-build QA chain if missing (e.g. loaded from history/cache)
        if st.session_state.get("qa_chain") is None and st.session_state.get("transcript"):
            with st.spinner("Setting up Q&A…"):
                try:
                    llm = get_llm(provider_key, model_name, temperature, api_key)
                    st.session_state["qa_chain"] = build_qa_chain(
                        llm, st.session_state["transcript"]
                    )
                except Exception as e:
                    st.warning(f"Could not build Q&A chain: {e}")

        # Render chat history
        for msg in st.session_state["chat_history"]:
            role_class = "chat-user" if msg["role"] == "user" else "chat-assistant"
            icon = "🧑" if msg["role"] == "user" else "🤖"
            st.markdown(
                f'<div class="{role_class}">{icon} {msg["content"]}</div>',
                unsafe_allow_html=True,
            )

        # Chat input
        user_q = st.chat_input("Ask a question about the video…")
        if user_q:
            st.session_state["chat_history"].append({"role": "user", "content": user_q})
            with st.spinner("Thinking…"):
                try:
                    answer = ask_question(
                        st.session_state["qa_chain"],
                        user_q,
                        st.session_state["chat_history"],
                    )
                except Exception as e:
                    answer = f"Error: {e}"
            st.session_state["chat_history"].append({"role": "assistant", "content": answer})
            st.rerun()

        if st.button("🗑️ Clear Chat", key="clear_chat"):
            st.session_state["chat_history"] = []
            st.rerun()

    # ── Tab 6: Keywords ──
    with tabs[5]:
        st.markdown("#### 🔑 Extracted Keywords")
        keywords = summaries.get("keywords", [])
        if keywords:
            html = "".join(
                f'<span class="keyword-badge">{kw}</span>' for kw in keywords
            )
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.info("Keyword extraction was not enabled or produced no results.")

    # ─────────────────────────────────────────
    # 7. EXPORT OPTIONS
    # ─────────────────────────────────────────
    st.divider()
    st.markdown("#### 📥 Export Summary")
    exp_col1, exp_col2, exp_col3 = st.columns(3)

    with exp_col1:
        txt_bytes = export_to_txt(summaries, video_id)
        st.download_button(
            label="⬇️ Download TXT",
            data=txt_bytes,
            file_name=f"summary_{sanitize_filename(video_id)}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with exp_col2:
        pdf_bytes = export_to_pdf(summaries, video_id)
        st.download_button(
            label="⬇️ Download PDF",
            data=pdf_bytes,
            file_name=f"summary_{sanitize_filename(video_id)}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    with exp_col3:
        json_str = json.dumps(summaries, indent=2, ensure_ascii=False)
        st.download_button(
            label="⬇️ Download JSON",
            data=json_str,
            file_name=f"summary_{sanitize_filename(video_id)}.json",
            mime="application/json",
            use_container_width=True,
        )

    # ── Raw transcript expander ──
    with st.expander("📜 View Raw Transcript"):
        st.text_area(
            "Full Transcript",
            st.session_state.get("transcript", ""),
            height=250,
            label_visibility="collapsed",
        )

else:
    # ── Welcome / empty state ──
    st.markdown("""
    <div class="result-card" style="text-align:center; padding: 3rem;">
        <h2>👆 Paste a YouTube URL above to get started</h2>
        <p style="opacity:.7; font-size:1rem;">
            The app will extract the transcript, summarize it with AI,<br>
            and let you chat with the video content — all in seconds.
        </p>
        <br>
        <div style="display:flex; justify-content:center; gap:2rem; flex-wrap:wrap;">
            <div>📋 <b>Summary</b></div>
            <div>💡 <b>Key Insights</b></div>
            <div>⏱️ <b>Topics</b></div>
            <div>⚡ <b>TL;DR</b></div>
            <div>💬 <b>Q&A Chat</b></div>
            <div>🔑 <b>Keywords</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
