# Vidhijna ⚖️

**AI-powered Indian Business Law Research Assistant**

Vidhijna is a multi-agent legal AI system built for Indian commercial law. It routes user queries to specialist agents — research, chat, document analysis, or legal drafting — using a LangGraph supervisor graph backed by Groq LLMs, Pinecone vector search, and Tavily web search.

---

## What It Does

| Mode | Description |
|---|---|
| 🔬 **Deep Research** | Multi-source legal research with a reflection loop — statutory provisions + commentary + live web |
| 💬 **Legal Chat** | Conversational Q&A with memory, grounded in retrieved legal context |
| 📄 **Document Analysis** | Upload a contract, judgment, or notice — get clause extraction, risk flags, and compliance checks (now with Groq-powered OCR and automated garbage collection) |
| ✍️ **Legal Drafting** | Generate NDAs, service agreements, legal notices, NCLT petitions, arbitration notices, and more |
| ✨ **Auto (Supervisor)** | The supervisor classifies intent and routes to the right agent automatically |

---

## Architecture

```
User Query
    │
    ▼
┌─────────────┐
│  Supervisor  │  — classifies intent, rewrites query, generates Tavily signals
└──────┬───────┘
       │
       ├──► research_agent  ──► [generate_query → propose_plan → parallel retrieval
       │                          → summarize → combine → extract_entities → reflect → loop/finalize]
       │
       ├──► chat_agent      ──► [retrieve → answer]
       │
       ├──► document_agent  ──► [validate → analyse → retrieve_law → flag_risks]
       │
       └──► draft_agent     ──► [validate_inputs → retrieve_law → draft → review]
                │
                ▼
        response_formatter  ──► citations + legal disclaimer
                │
                ▼
            Final Response (streamed via SSE)
```

The graph is compiled with `MemorySaver` so conversation history persists per `thread_id` across turns.

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLMs | Groq (`llama-3.3-70b-versatile` for research, `llama-3.1-8b-instant` for chat/supervisor) |
| Vector store | Pinecone (dual namespace: `vidhijna-legal` + `vidhijna-books`) |
| Embeddings | fastembed (baked into Docker image) |
| Web search | Tavily (targeted at Indian legal domains) |
| OCR | Groq API |
| Backend | FastAPI with SSE streaming (hosted on Google Cloud Run) |
| Frontend | Vanilla HTML/CSS/JS (hosted on Vercel, mobile responsive) |

---

## Project Structure

```
.
├── agents/
│   ├── graph.py              # Main supervisor graph — entry point
│   ├── state.py              # VidhijnaState, TavilyFetchSignal, I/O types
│   ├── configuration.py      # Central config — models, Pinecone, Tavily, weights
│   ├── prompts.py            # All LLM prompts for every agent
│   ├── utils.py              # Shared utilities — dedup, JSON extraction, text cleaning
│   │
│   ├── subgraphs/
│   │   ├── research.py       # Deep research pipeline with reflection loop
│   │   ├── chat.py           # Conversational agent with memory
│   │   ├── document.py       # Document analysis — OCR, clause extraction, risk flags
│   │   └── drafting.py       # Legal document drafting
│   │
│   └── tools/
│       ├── retrieval.py      # Pinecone dual-namespace retriever with authority weighting
│       ├── search.py         # Tavily web search
│       ├── ocr.py            # PDF/image/docx text extraction
│       └── drafting_tools.py # Standard clause libraries, act maps, draft validation
│
├── backend/
│   └── main.py               # FastAPI server — streaming SSE endpoints
│
└── frontend/
    ├── index.html
    ├── script.js
    └── style.css
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/sanatwalia896/Vidhijna.git
cd Vidhijna
git checkout staging
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
# LLMs
GROQ_API_KEY=your_groq_api_key

# Vector store
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=vidhijana-indexes
PINECONE_REGION=us-east-1

# Web search
TAVILY_API_KEY=your_tavily_api_key

# Optional overrides
GROQ_MODEL=llama-3.1-8b-instant
RESEARCH_MODEL=llama-3.3-70b-versatile
CHAT_MODEL=llama-3.1-8b-instant
SUPERVISOR_MODEL=llama-3.1-8b-instant
MAX_REFLECTION_LOOPS=3
DEV_MODE=true
```

### 3. Run the server

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

The frontend is served at `http://localhost:8000/app`.

---

## API Reference

### `POST /chat`

Run the multi-agent pipeline. Streams events via SSE.

```json
{
  "query": "What are the director's fiduciary duties under the Companies Act 2013?",
  "thread_id": "t_abc123",
  "mode": "auto",
  "reflection_loops": 3
}
```

**`mode` options:** `auto` | `research` | `chat` | `document` | `draft`

### `POST /upload`

Upload a document for analysis. Accepts `multipart/form-data` with fields `file`, `thread_id`, and `query`.

Max file size: 20MB. Supported formats: PDF, PNG, JPG, JPEG, TIFF, DOCX.

### `GET /threads`

List all active conversation threads with message counts.

### `DELETE /threads/{thread_id}`

Delete a conversation thread.

### `GET /modes`

Returns available modes and supported draft types for the frontend.

### `GET /health`

Returns system health status and flags any missing API keys.

---

## Streaming Events (SSE)

The `/chat` and `/upload` endpoints stream real-time events. Each event has a `type` field:

| Event | Description |
|---|---|
| `status` | Agent progress message (e.g. "Searching statutory provisions...") |
| `node_start` | A graph node has started executing |
| `research_card` | Flash card — act found, commentary excerpt, or summary |
| `risk_flag` | A risk identified in an uploaded document |
| `citations` | Source citations accumulated so far |
| `draft_preview` | Preview of the draft being generated |
| `final` | Complete final response with citations and entities |
| `error` | An error occurred |

---

## Research Pipeline (Deep Dive)

The research subgraph runs a full reflection loop:

1. **`generate_query`** — rewrites the user query for Pinecone retrieval
2. **`propose_plan`** — generates a research plan (acts to check, domains, complexity)
3. **Parallel retrieval:**
   - `retrieve_legal` → `vidhijna-legal` namespace (statutes, acts, provisions)
   - `retrieve_books` → `vidhijna-books` namespace (commentary, case digests)
   - `web_search` → Tavily (cases, regulations, circulars, recent judgments)
4. **`summarize_vectors`** — summarizes statutory + commentary chunks
5. **`summarize_web`** — summarizes web results
6. **`combine`** — merges all summaries into a running research doc
7. **`extract_entities`** — extracts statutes, cases, courts, parties, dates
8. **`reflect`** — checks for knowledge gaps and generates follow-up queries
9. **Loop** back to retrieval if gaps found (up to `MAX_REFLECTION_LOOPS`)
10. **`finalize`** — generates the structured final report

### Authority Weighting

Retrieved chunks are scored by cosine similarity then multiplied by authority weights:

| Category | Weight |
|---|---|
| Constitution / Act | 1.3× |
| Code | 1.2× |
| Regulation | 1.1× |
| Commentary | 1.2× |
| Case digest | 1.15× |

### Tavily Auto-Trigger

Topics not well covered in the vector store (GST, RBI circulars, SEBI notifications, RERA, patents, etc.) automatically trigger a Tavily web search targeting the relevant government domains.

---

## Supported Draft Types

| Draft Type | Key Acts Referenced |
|---|---|
| NDA | Indian Contract Act, 1872 |
| Service Agreement | Indian Contract Act 1872, Specific Relief Act 1963 |
| Employment Contract | Indian Contract Act, 1872 |
| Sale Deed | Transfer of Property Act 1882, Sale of Goods Act 1930 |
| Lease Agreement | Transfer of Property Act 1882, Indian Contract Act 1872 |
| Legal Notice | Indian Contract Act 1872, Limitation Act 1963 |
| Cease & Desist | Trade Marks Act 1999, Copyright Act 1957 |
| NCLT Petition | Companies Act 2013, IBC 2016 |
| Consumer Complaint | Consumer Protection Act, 2019 |
| Arbitration Notice | Arbitration and Conciliation Act, 1996 |

---

## Configuration Reference

All settings can be overridden via environment variables or a `RunnableConfig` passed to LangGraph. Key options:

| Variable | Default | Description |
|---|---|---|
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Default LLM for document/draft agents |
| `RESEARCH_MODEL` | `llama-3.3-70b-versatile` | LLM for research agent |
| `MAX_REFLECTION_LOOPS` | `3` | Max research reflection iterations |
| `RETRIEVAL_TOP_K_LEGAL` | `20` | Pinecone candidates before reranking (legal) |
| `RERANK_TOP_N_LEGAL` | `6` | Final chunks kept after reranking (legal) |
| `RETRIEVAL_SCORE_THRESHOLD` | `0.4` | Min cosine similarity to include a chunk |
| `FILTER_CONFIDENCE_THRESHOLD` | `0.8` | Min LLM confidence to apply metadata filter |
| `TAVILY_MAX_RESULTS` | `5` | Max Tavily results per search |
| `MAX_FILE_SIZE_MB` | `20` | Max document upload size |
| `DEV_MODE` | `true` | Suppress errors on missing API keys |

---

## Disclaimer

> ⚠️ Vidhijna generates AI-assisted legal information for research purposes only. It does not constitute legal advice. Always consult a qualified lawyer before taking legal action.