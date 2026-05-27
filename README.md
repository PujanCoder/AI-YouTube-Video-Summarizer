# 🎬 YouTube AI Summarizer

A production-grade, AI-powered web app that extracts YouTube transcripts and
generates summaries, insights, timestamps, TL;DRs, and interactive Q&A — all
in a clean Streamlit UI.

---

## ✨ Features

| Feature | Detail |
|---|---|
| 📋 **Smart Summary** | Map-reduce chunking for videos of any length |
| 💡 **Key Insights** | 5–8 extracted bullet-point takeaways |
| ⏱️ **Topics & Timestamps** | Auto-detected chapter markers |
| ⚡ **TL;DR** | One-sentence video summary |
| 💬 **RAG Q&A Chat** | Ask anything, grounded in the transcript |
| 🔑 **Keyword Extraction** | Top keyphrases from the video |
| 📥 **Export** | Download as PDF, TXT, or JSON |
| ⚡ **Caching** | No repeat API calls for the same video |
| 🕘 **History** | Last 50 summaries stored locally |
| 🌐 **Multi-language** | 10+ transcript language options |

---

## 🚀 Quick Start

### 1 · Clone the repo

```bash
git clone https://github.com/your-username/youtube-ai-summarizer.git
cd youtube-ai-summarizer
```

### 2 · Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3 · Install dependencies

```bash
pip install -r requirements.txt
```

### 4 · Add your API key

```bash
cp .env.example .env
# Edit .env and paste your Groq or OpenAI key
```

> **Groq is recommended** — it's free, extremely fast, and supports
> Llama 3 / Mixtral models. Get a free key at <https://console.groq.com/keys>.

### 5 · Run the app

```bash
streamlit run app.py
```

Open <http://localhost:8501> in your browser.

---

## 🗂️ Project Structure

```
youtube_summarizer/
├── app.py                  # Streamlit entry point
├── requirements.txt
├── .env.example
├── .streamlit/
│   └── config.toml         # Theme & server config
├── src/
│   ├── __init__.py
│   ├── transcript.py       # YouTube transcript extraction
│   ├── summarizer.py       # LLM summarization (LangChain LCEL)
│   ├── chains.py           # RAG Q&A chain
│   ├── cache_manager.py    # Disk cache + history log
│   ├── export_utils.py     # PDF / TXT export
│   └── utils.py            # Misc helpers
└── data/                   # Auto-created at runtime
    ├── cache/              # Cached API responses (JSON)
    └── history.jsonl       # Summary history log
```

---

## ⚙️ Configuration

All settings are available in the **sidebar** at runtime:

| Setting | Options |
|---|---|
| LLM Provider | Groq (free) · OpenAI |
| Model | llama3-8b, llama3-70b, mixtral, gpt-4o-mini, etc. |
| Summary Length | Brief · Standard · Detailed |
| Temperature | 0.0 (factual) → 1.0 (creative) |
| Transcript Language | en, es, fr, de, pt, ja, ko, zh, hi, ar |
| Keyword Extraction | On / Off toggle |

---

## ☁️ Deploy to Streamlit Cloud

1. Push your code to a **public** GitHub repo.
2. Go to <https://share.streamlit.io> and click **New app**.
3. Select your repo, set the main file to `app.py`.
4. Under **Secrets**, add:
   ```toml
   GROQ_API_KEY = "your_key_here"
   # or
   OPENAI_API_KEY = "your_key_here"
   ```
5. Click **Deploy** — done!

---

## 🤗 Deploy to Hugging Face Spaces

1. Create a new **Streamlit** Space on <https://huggingface.co/spaces>.
2. Upload all files (or connect your GitHub repo).
3. Add your API key in **Settings → Repository Secrets**.
4. The Space will auto-build and launch.

---

## 💡 Architecture Notes

### Transcript pipeline
```
YouTube URL → extract_video_id() → YouTubeTranscriptAPI
  → language fallback chain → plain text + raw timestamps
```

### Summarization pipeline
```
transcript text → RecursiveCharacterTextSplitter
  → (map) chunk summaries → (reduce) final summary
  → parallel: insights · tldr · topics · keywords
```

### Q&A pipeline
```
transcript → FAISS vector store (FakeEmbeddings)
  → retriever (top-4 chunks) → LLM with chat history
```

### Caching
- Results are hashed by `video_id + model + summary_length` and stored as JSON.
- Cache hit = **zero API calls** and instant response.

### Token efficiency
- Only the first 2 000–4 000 chars are used for TL;DR, keywords, and title.
- Map-reduce limits each chunk to ~3 500 chars.
- Chat Q&A uses only the 4 most relevant transcript chunks.

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---|---|
| "Transcripts are disabled" | The creator disabled captions. Try another video. |
| "Video unavailable" | The video is private, deleted, or age-restricted. |
| Import errors | Run `pip install -r requirements.txt` again. |
| Slow responses | Switch to Groq + `llama3-8b-8192` for fastest inference. |
| PDF download empty | Ensure `reportlab` is installed: `pip install reportlab`. |

---

## 📄 License

MIT License — feel free to use, modify, and deploy.
