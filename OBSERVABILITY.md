# Observability

Vidhijna includes request-level tracing, token and cost metrics, and per-role latency summaries via **Langfuse**. This document covers the observability stack.

**Status: Not deployed online. Observability is available locally only.**

---

## What Was Added

On the `feat/langfuse_integration` branch, the following changes were made:

### 1. Langfuse Tracing Integration (`agents/graph.py`)
- `langfuse.langchain.CallbackHandler` is loaded **opt-in only** when all required env vars are present
- If any env var is missing, tracing silently skips — the system still runs normally
- A `langfuse_status()` helper exposes whether tracing is enabled

### 2. Config Metadata (`agents/graph.py:build_runtime_config`)
- Every graph run is tagged with: `thread_id`, `request_id`, `mode`, `model_used`
- Session and trace IDs are passed through for proper grouping in Langfuse

### 3. Metrics Collector (`agents/metrics.py`)
- `MetricsCollector` (implements `BaseCallbackHandler`) tracks:
  - Per-LLM-call: prompt/completion/total tokens, latency ms, cost USD, tokens/sec
  - Per-node: latency ms (with p95 per node)
  - Per-request: full summary written to `metrics.jsonl`
- `compute_rag_observability()` scores retrieval quality:
  - `retrieval_relevance` — average normalized cosine similarity of retrieved chunks
  - `context_utilization` — token overlap between answer and retrieved context
  - `citation_coverage` — what fraction of retrieved sources are actually cited
  - `faithfulness_proxy` — weighted combination of utilization + citation coverage
  - `source_diversity` — ratio of unique sources to total chunks
- `push_langfuse_rag_scores()` sends these scores to Langfuse as trace scores

### 4. Metrics Endpoint (`backend/main.py`)
- `GET /metrics` returns: recent requests, per-role token/cost breakdown, p95 latencies, aggregated RAG scores

---

## What You Get From Observability

| Metric | What It Tells You |
|---|---|
| Token count + cost | Cost per request, per role, total spend |
| Latency per node | Which part of the pipeline is slowest |
| Latency p95 | Worst-case response time per node |
| Tokens per second | LLM throughput efficiency |
| Retrieval relevance | Are retrieved chunks actually similar to the query? |
| Context utilization | Is the model actually using the retrieved context? |
| Citation coverage | Are sources being used vs. ignored? |
| Faithfulness proxy | Overall grounding quality of the answer |

---

## How To Run Locally

1. Start Langfuse locally (Docker or self-hosted)
2. Add to `.env`:
   ```
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_HOST=http://localhost:3000
   LANGFUSE_BASE_URL=http://localhost:3000
   ```
3. Run the backend: `python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload`
4. Send a request (deep research mode works best)
5. Open Langfuse at `http://localhost:3000` and inspect the trace

---

## What To Look For In Langfuse

The best demo trace is a **deep research** request, not a simple chat request.

Look for:

- One trace per user request
- `thread_id` session grouping
- `mode = research`
- Supervisor routing span
- Research subgraph spans
- Multiple LLM generations
- Reflection loop iterations
- Token usage and latency per generation

---

## Metrics Endpoint

`GET /metrics` returns:

```json
{
  "recent_requests": [...],
  "summary": {
    "request_count": 0,
    "llm_call_count": 0,
    "total_tokens": 0,
    "total_cost_usd": 0,
    "avg_tokens_per_second": 0,
    "by_role": {
      "supervisor": { "calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0 },
      "research": { ... },
      "chat": { ... },
      "document": { ... },
      "draft": { ... }
    },
    "rag_observability": {
      "sample_size": 0,
      "avg_retrieval_relevance": 0,
      "avg_context_utilization": 0,
      "avg_citation_coverage": 0,
      "avg_faithfulness_proxy": 0,
      "avg_source_diversity": 0
    }
  },
  "p95_latency_ms": {
    "supervisor": 0,
    "research_agent": 0,
    ...
  }
}
```

---

## Recommended Screenshots

Place these in the `images/` folder.

### 1. RAG Metrics Dashboard
> **Placeholder:** `images/rag_metrics.png`
>
> Show the aggregated RAG observability scores — retrieval_relevance, context_utilization, citation_coverage, faithfulness_proxy, source_diversity. This demonstrates retrieval quality tracking.

### 2. Trace Graph — Full Request
> **Placeholder:** `images/trace_graph.png`
>
> Show the full LangGraph trace for a deep research request. This is the single best screenshot for showing multi-agent orchestration end-to-end.

### 3. Trace Content — Node Details
> **Placeholder:** `images/trace_content.png`
>
> Expand a trace to show what was traced — generation inputs, outputs, and intermediate state. Proves the agent's reasoning path is visible.

### 4. Trace Types — Generation / Agent / Chain
> **Placeholder:** `images/trace_types.png`
>
> Show the different span types in Langfuse — LLM generation spans, agent nodes, and chain edges. Demonstrates the structured nature of the trace.

### 5. Metrics Endpoint Response
> **Placeholder:** `images/metrics_endpoint.png`
>
> Show the `GET /metrics` JSON response with `summary.by_role`, `p95_latency_ms`, and RAG observability averages.

---

## Suggested Demo Query

Use a query that naturally triggers multiple nodes and at least one reflection loop:

> "Legal Analysis: Director Duties and Oppression/Mis-management Remedies under the Companies Act, 2013 (with recent legal context)"

This produces a richer trace than a simple factual chat.

---

## Notes

- Observability is **opt-in** — if Langfuse env vars are missing, Vidhijna runs normally without tracing
- Tracing does not change the Cloud Run deployment path
- SSE streaming remains intact
- RAGAS evaluation was removed from this version
