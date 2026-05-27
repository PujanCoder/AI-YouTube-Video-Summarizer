"""
summarizer.py
=============
Core summarization logic using LangChain Expression Language (LCEL).

Features:
- Supports OpenAI and Groq providers
- Token-efficient prompts (low API cost)
- Map-reduce chunking for long videos
- Parallel generation of summary, insights, TL;DR, topics, keywords
"""

import json
import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Provider-specific imports (lazy to avoid import errors if not installed)
def _get_openai_llm(model: str, temperature: float, api_key: str):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model, temperature=temperature, openai_api_key=api_key)

def _get_groq_llm(model: str, temperature: float, api_key: str):
    from langchain_groq import ChatGroq
    return ChatGroq(model=model, temperature=temperature, groq_api_key=api_key)


# ─────────────────────────────────────────────
# LLM Factory
# ─────────────────────────────────────────────
def get_llm(provider: str, model: str, temperature: float, api_key: str) -> Any:
    """
    Instantiate and return the appropriate LangChain LLM.

    Args:
        provider:    'openai' or 'groq'
        model:       Model name string
        temperature: Sampling temperature (0.0–1.0)
        api_key:     API key for the provider

    Returns:
        LangChain chat model instance.

    Raises:
        ValueError: If provider is unrecognised.
        ImportError: If the provider's package isn't installed.
    """
    if not api_key:
        raise ValueError("API key is required. Please add it in the sidebar.")

    if provider == "openai":
        return _get_openai_llm(model, temperature, api_key)
    elif provider == "groq":
        return _get_groq_llm(model, temperature, api_key)
    else:
        raise ValueError(f"Unknown provider: '{provider}'. Choose 'openai' or 'groq'.")


# ─────────────────────────────────────────────
# Text Chunking
# ─────────────────────────────────────────────
def chunk_transcript(text: str, chunk_size: int = 3500, chunk_overlap: int = 200) -> list[Document]:
    """
    Split a long transcript string into overlapping LangChain Documents.

    Uses RecursiveCharacterTextSplitter so splits happen at sentence
    or paragraph boundaries where possible.

    Args:
        text:          Full transcript text.
        chunk_size:    Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of LangChain Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk) for chunk in chunks]


# ─────────────────────────────────────────────
# Token-efficient prompt helpers
# ─────────────────────────────────────────────
_LENGTH_INSTRUCTIONS = {
    "Brief":    "in 3-5 concise sentences",
    "Standard": "in 2-3 short paragraphs",
    "Detailed": "in 4-6 detailed paragraphs with examples",
}


def _map_reduce_summarize(llm: Any, docs: list[Document], length_hint: str) -> str:
    """
    Summarize a list of documents using the map-reduce pattern.

    Map  : Summarize each chunk individually.
    Reduce: Combine chunk summaries into one final summary.

    Args:
        llm:         LangChain LLM instance.
        docs:        List of Document chunks.
        length_hint: Human-readable length instruction.

    Returns:
        Final combined summary string.
    """
    parser = StrOutputParser()

    # ── Map: summarise each chunk ──
    map_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert video content analyst. Be concise."),
        ("human",  "Summarise this transcript excerpt in 2-4 sentences:\n\n{chunk}"),
    ])
    map_chain = map_prompt | llm | parser

    chunk_summaries = [
        map_chain.invoke({"chunk": doc.page_content})
        for doc in docs
    ]
    combined = "\n\n".join(chunk_summaries)

    # ── Reduce: merge chunk summaries ──
    reduce_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert summariser. Produce coherent, informative summaries."),
        ("human",
         f"Combine these partial summaries into one cohesive summary {length_hint}. "
         "Remove duplicates and maintain logical flow.\n\n{combined}"),
    ])
    reduce_chain = reduce_prompt | llm | parser
    return reduce_chain.invoke({"combined": combined})


def _generate_insights(llm: Any, text: str) -> list[str]:
    """
    Extract 5-8 key insights from the transcript as bullet points.

    Args:
        llm:  LangChain LLM instance.
        text: Transcript (or summary for long videos).

    Returns:
        List of insight strings (without bullet markers).
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert content analyst. Extract the most valuable insights."),
        ("human",
         "From the following video transcript, extract 5-8 key insights or takeaways. "
         "Return ONLY a JSON array of strings, no preamble.\n\n"
         "Example: [\"Insight 1\", \"Insight 2\"]\n\n"
         "Transcript:\n{text}"),
    ])
    chain = prompt | llm | StrOutputParser()

    # Use only first 4000 chars to save tokens
    result = chain.invoke({"text": text[:4000]})

    try:
        # Strip markdown fences if present
        cleaned = re.sub(r"```(?:json)?|```", "", result).strip()
        return json.loads(cleaned)
    except Exception:
        # Fallback: split by newline / bullet
        lines = [
            line.lstrip("•-*123456789. ").strip()
            for line in result.split("\n")
            if line.strip()
        ]
        return lines[:8]


def _generate_tldr(llm: Any, text: str) -> str:
    """
    Generate a single-sentence TL;DR for the video.

    Args:
        llm:  LangChain LLM instance.
        text: Transcript text (first 2000 chars used).

    Returns:
        A single punchy sentence summarising the video.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You write ultra-concise, punchy TL;DRs."),
        ("human",
         "Write a single sentence TL;DR (max 30 words) for this video transcript:\n\n{text}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"text": text[:2000]}).strip()


def _generate_topics(llm: Any, transcript_raw: list[dict]) -> list[dict]:
    """
    Identify 4-6 major topic segments from the timestamped transcript.

    Args:
        llm:            LangChain LLM instance.
        transcript_raw: Raw transcript list with 'start' and 'text' keys.

    Returns:
        List of {'timestamp': 'MM:SS', 'topic': str, 'description': str}.
    """
    # Sample every ~2 minutes to keep token count low
    sampled = []
    last_sampled = -120
    for seg in transcript_raw:
        start = seg.get("start", 0)
        if start - last_sampled >= 120:
            ts_str = f"{int(start)//60:02d}:{int(start)%60:02d}"
            sampled.append(f"[{ts_str}] {seg.get('text','')}")
            last_sampled = start
    sampled_text = "\n".join(sampled[:40])  # cap at 40 samples

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You identify key topic segments in video transcripts."),
        ("human",
         "Based on this timestamped transcript sample, identify 4-6 major topic segments. "
         "Return ONLY a JSON array with objects having keys: timestamp, topic, description.\n\n"
         "Example: [{\"timestamp\": \"00:00\", \"topic\": \"Introduction\", \"description\": \"...\"}]\n\n"
         "Transcript sample:\n{text}"),
    ])
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"text": sampled_text})

    try:
        cleaned = re.sub(r"```(?:json)?|```", "", result).strip()
        return json.loads(cleaned)
    except Exception:
        return [{"timestamp": "00:00", "topic": "Full Video", "description": "See summary tab."}]


def _generate_keywords(llm: Any, text: str) -> list[str]:
    """
    Extract 8-12 important keywords or keyphrases from the transcript.

    Args:
        llm:  LangChain LLM instance.
        text: Transcript text (first 2500 chars used).

    Returns:
        List of keyword strings.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You extract keywords and keyphrases from text."),
        ("human",
         "Extract 8-12 important keywords or keyphrases from this transcript. "
         "Return ONLY a JSON array of strings.\n\nTranscript:\n{text}"),
    ])
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"text": text[:2500]})

    try:
        cleaned = re.sub(r"```(?:json)?|```", "", result).strip()
        return json.loads(cleaned)
    except Exception:
        return [kw.strip() for kw in result.split(",") if kw.strip()][:12]


def _extract_title(llm: Any, text: str) -> str:
    """Generate a short descriptive title for the video content."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You write concise titles."),
        ("human",  "Write a short descriptive title (max 10 words) for this video:\n{text}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"text": text[:500]}).strip().strip('"')


# ─────────────────────────────────────────────
# Master orchestrator
# ─────────────────────────────────────────────
def generate_all_summaries(
    llm:              Any,
    docs:             list[Document],
    transcript_raw:   list[dict],
    transcript_text:  str,
    summary_length:   str = "Standard",
    extract_keywords: bool = True,
) -> dict:
    """
    Run the full summarisation pipeline and return all outputs.

    Generates:
        - title
        - summary
        - insights (list)
        - tldr
        - topics (list with timestamps)
        - keywords (list, optional)

    Args:
        llm:              LangChain LLM instance.
        docs:             Chunked transcript Documents.
        transcript_raw:   Raw transcript with timestamps.
        transcript_text:  Plain transcript string.
        summary_length:   'Brief' | 'Standard' | 'Detailed'
        extract_keywords: Whether to run keyword extraction.

    Returns:
        Dictionary with all generated content.
    """
    length_hint = _LENGTH_INSTRUCTIONS.get(summary_length, _LENGTH_INSTRUCTIONS["Standard"])

    # For short transcripts, use a single pass; for long ones, use map-reduce
    if len(docs) == 1:
        # Single-pass summarization
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert video content analyst. Be concise and informative."),
            ("human",  f"Summarise this video transcript {length_hint}:\n\n{{text}}"),
        ])
        chain = prompt | llm | StrOutputParser()
        summary = chain.invoke({"text": transcript_text[:5000]})
    else:
        summary = _map_reduce_summarize(llm, docs, length_hint)

    # Use the summary as context for other tasks (much cheaper than full transcript)
    context = summary if len(transcript_text) > 4000 else transcript_text

    results = {
        "title":    _extract_title(llm, transcript_text),
        "summary":  summary,
        "insights": _generate_insights(llm, context),
        "tldr":     _generate_tldr(llm, context),
        "topics":   _generate_topics(llm, transcript_raw),
        "keywords": _generate_keywords(llm, context) if extract_keywords else [],
    }
    return results
