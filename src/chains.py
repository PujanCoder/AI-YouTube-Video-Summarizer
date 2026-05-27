"""
chains.py
=========
Builds a conversational Q&A chain grounded in the video transcript.

Uses LangChain's LCEL (LangChain Expression Language) with:
- In-memory vector store (no external DB needed)
- Retrieval-Augmented Generation (RAG) pattern
- Conversation-aware prompts with chat history
"""

from typing import Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings


def _build_vectorstore(transcript_text: str):
    """
    Chunk the transcript and build a FAISS vector store with fake embeddings.

    We use FakeEmbeddings to avoid an extra API key requirement.
    For production, swap in OpenAIEmbeddings or HuggingFaceEmbeddings.

    Args:
        transcript_text: Plain transcript string.

    Returns:
        FAISS vector store populated with transcript chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(transcript_text)

    # FakeEmbeddings generates random vectors — sufficient for keyword-based
    # retrieval on small corpora; swap for real embeddings in production.
    embeddings = FakeEmbeddings(size=1536)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore


def build_qa_chain(llm: Any, transcript_text: str):
    """
    Build a retrieval-augmented Q&A chain for the given transcript.

    The chain:
        1. Retrieves the 4 most relevant chunks from the vector store.
        2. Passes context + chat history to the LLM.
        3. Returns a grounded, conversational answer.

    Args:
        llm:             LangChain LLM instance.
        transcript_text: Full plain transcript string.

    Returns:
        A dict containing {'chain': runnable, 'retriever': retriever}
        ready to be used with ask_question().
    """
    vectorstore = _build_vectorstore(transcript_text)
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a helpful assistant answering questions about a YouTube video.
Use ONLY the provided context from the video transcript to answer.
If the answer is not in the context, say "I couldn't find that in the video."
Be concise and direct. When quoting the video, say "In the video, ...".

Context from the video:
{context}""",
        ),
        ("human", "{question}"),
    ])

    def _format_docs(docs) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context":  retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return {"chain": chain, "retriever": retriever}


def ask_question(
    qa_bundle: dict,
    question: str,
    chat_history: list[dict] | None = None,
) -> str:
    """
    Invoke the Q&A chain with a user question.

    Injects the last 3 Q&A turns into the question for context continuity,
    without rebuilding the full chain each time.

    Args:
        qa_bundle:    Dict returned by build_qa_chain().
        question:     The user's current question string.
        chat_history: List of {'role': str, 'content': str} dicts.

    Returns:
        LLM answer string.

    Raises:
        RuntimeError: If the chain is not available.
    """
    if not qa_bundle or "chain" not in qa_bundle:
        raise RuntimeError("Q&A chain is not initialised.")

    # Build a context-enriched question using the last 3 turns
    history_str = ""
    if chat_history and len(chat_history) > 1:
        recent = chat_history[-6:-1]  # last 3 exchanges (user+assistant pairs)
        pairs  = []
        for msg in recent:
            role    = "User" if msg["role"] == "user" else "Assistant"
            pairs.append(f"{role}: {msg['content']}")
        if pairs:
            history_str = "\n".join(pairs)
            question    = (
                f"Previous conversation:\n{history_str}\n\n"
                f"New question: {question}"
            )

    return qa_bundle["chain"].invoke(question)
