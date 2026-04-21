# Observability

Vidhijna now includes request-level tracing, token and cost metrics, and per-role latency summaries. This file documents the observability stack that was added for debugging, demoing, and evaluating the system.

## What Was Added

- Langfuse tracing for the LangGraph pipeline
- Trace grouping by `thread_id`
- Trace metadata for:
  - `mode`
  - `model_used`
  - request ID
- Per-call LLM metrics:
  - prompt tokens
  - completion tokens
  - total tokens
  - tokens per second
  - estimated cost
- Request-level aggregation in `/metrics`
- Per-role latency summaries:
  - `supervisor`
  - `chat`
  - `research`
  - `document`
  - `draft`
- Local metrics persistence in `metrics.jsonl`

## Why It Matters

This observability layer makes the system easier to:

- debug production failures
- identify slow nodes
- compare model usage across roles
- estimate cost per request
- validate RAG quality and reflection loop behavior

For a forward deployed engineer workflow, this is the difference between a working prototype and an operable system.

## How To Run Locally

1. Start Langfuse locally.
2. Add the required env vars to `.env`:
   - `LANGFUSE_PUBLIC_KEY`
   - `LANGFUSE_SECRET_KEY`
   - `LANGFUSE_HOST`
   - `LANGFUSE_BASE_URL`
3. Run the backend.
4. Send a request in `research` mode.
5. Open Langfuse and inspect the resulting trace.

## What To Look For In Langfuse

The strongest demo trace is a **deep research** request, not a simple chat request.

Look for:

- one trace per user request
- `thread_id` session grouping
- `mode = research`
- supervisor routing span
- research subgraph spans
- multiple LLM generations
- reflection loop iterations
- token usage and latency per generation

## Metrics Endpoint

`GET /metrics` returns:

- `recent_requests`
- `summary`
- `p95_latency_ms`

The `summary` block includes:

- total request count
- total LLM call count
- total tokens
- total cost in USD
- average tokens per second
- per-role breakdown
- p95 latency by node

## Opik RAG Evaluation

The repository includes a live RAG eval runner at `tests/eval/rag_eval.py`.

It evaluates Vidhijna on a legal dataset using Opik metrics:

- `AnswerRelevance`
- `Hallucination`
- `ContextPrecision`
- `ContextRecall`

The eval writes a JSON report to:

`eval_results/rag_eval_report.json`

Run it with:

```bash
OPIK_API_KEY=... OPIK_HOST=... python3 tests/eval/rag_eval.py
```

This is the best artifact to show retrieval quality, grounding, and context usage.

## Recommended Screenshots

Add 4 to 6 screenshots in the README or a short portfolio note.

1. **Langfuse trace overview for a deep research query**
   - Show the full request trace
   - Best single screenshot for end-to-end observability

2. **Expanded trace timeline**
   - Show supervisor, research subgraph, reflection loop, and finalization
   - Proves multi-agent orchestration

3. **Trace metadata / session panel**
   - Show `thread_id`, `mode`, and `model_used`
   - Proves trace grouping and tagging

4. **Trace generation details**
   - Show token counts and latency per LLM call
   - Best evidence for LLM-native observability

5. **`GET /metrics` JSON response**
   - Show `summary.by_role`
   - Show `p95_latency_ms`
   - Useful for demoing a lightweight ops dashboard

6. **RAG eval report output**
   - Show answer relevance, hallucination, context precision, and context recall
   - Strong proof of quality control and retrieval grounding

## Suggested Demo Query

Use a query that naturally triggers multiple nodes and at least one reflection loop, such as:

- “Legal Analysis: Director Duties and Oppression/Mis-management Remedies under the Companies Act, 2013 (with recent legal context)”
- “Explain the exact wording of Section 7(5)(c) of the Insolvency and Bankruptcy Code, 2016 and recent context”

These queries produce richer traces than a simple factual chat.

## Notes

- Observability is opt-in and does not change the Cloud Run deployment path.
- If Langfuse env vars are missing, Vidhijna still runs normally.
- The current branch keeps SSE streaming intact.
