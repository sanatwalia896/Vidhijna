"""
Vidhijna FastAPI Backend — Production-grade legal AI server
Optimized for Always Free Tier (512MB RAM)
Streams real-time agent activity, legal entities, risk flags, and final reports.
"""

import os
import sys
import json
import uuid
import traceback
import asyncio
import gc
from datetime import datetime
from typing import AsyncGenerator, Dict, Any

# Add the project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from agents.graph import graph
from agents.graph import build_runtime_config, langfuse_status
from agents.metrics import METRICS_COLLECTOR
from agents.configuration import Configuration

# ── App Setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Vidhijna Legal AI API",
    description="Indian Business Law AI Assistant — Multi-Agent System",
    version="2.0.0",
)

print(f"[LANGFUSE] {langfuse_status()}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/app", StaticFiles(directory=frontend_dir, html=True), name="frontend")

# ── Traffic & Memory Control ──────────────────────────────────────────────────

# Limit to 2 concurrent heavy tasks to prevent OOM on 512MB RAM
process_limiter = asyncio.Semaphore(2)
threads_store: Dict[str, Dict[str, Any]] = {}

# ── Models ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    thread_id: str = Field(default_factory=lambda: f"t_{uuid.uuid4().hex[:12]}")
    mode: str = "auto"
    draft_type: str = ""
    draft_inputs: dict = {}
    reflection_loops: int = Field(default=3, ge=1, le=5, description="Number of reflection iterations for deep research")

class ThreadInfo(BaseModel):
    thread_id: str
    title: str = ""
    mode: str = ""
    created_at: str = ""
    message_count: int = 0

# ── SSE Helper ────────────────────────────────────────────────────────────────

def sse_event(event_type: str, data: dict) -> str:
    """Format a Server-Sent Event string."""
    payload = {"type": event_type, "timestamp": datetime.utcnow().isoformat(), **data}
    return f"data: {json.dumps(payload)}\n\n"

# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "online", "service": "Vidhijna Legal AI", "version": "2.0.0"}

@app.get("/health")
async def health():
    from agents.configuration import Configuration
    cfg = Configuration()
    missing = cfg.validate()
    return {
        "status": "healthy" if not missing else "degraded",
        "missing_keys": missing,
        "graph_ready": graph is not None,
        "timestamp": datetime.utcnow().isoformat(),
    }

# ── Thread Management ────────────────────────────────────────────────────────

@app.get("/threads")
async def list_threads():
    return [
        {
            "thread_id": tid,
            "title": info.get("title", "Untitled"),
            "mode": info.get("mode", "auto"),
            "created_at": info.get("created_at", ""),
            "message_count": info.get("message_count", 0),
        }
        for tid, info in threads_store.items()
    ]

@app.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str):
    if thread_id in threads_store:
        del threads_store[thread_id]
    return {"status": "deleted", "thread_id": thread_id}

# ── Core Streaming Agent Runner ───────────────────────────────────────────────

# Map internal node names to user-friendly labels
NODE_LABELS = {
    "supervisor":        "🧠 Classifying your query...",
    "research_agent":    "🔬 Launching deep research pipeline...",
    "chat_agent":        "💬 Preparing conversational response...",
    "document_agent":    "📄 Analyzing uploaded document...",
    "draft_agent":       "✍️ Initiating legal drafting engine...",
    "propose_plan":      "📋 Creating research strategy...",
    "retrieve_legal":    "⚖️ Searching statutory provisions...",
    "retrieve_books":    "📚 Searching legal commentary...",
    "web_search":        "🌐 Fetching recent judgments & regulations...",
    "summarize_legal":   "📝 Summarizing statutory provisions...",
    "summarize_books":   "📚 Summarizing legal commentary...",
    "summarize_web":     "📰 Analyzing web research results...",
    "combine":           "🔗 Consolidating all research...",
    "extract_entities":  "🏛️ Extracting legal entities...",
    "reflect":           "🤔 Checking for knowledge gaps...",
    "finalize":          "📊 Generating final research report...",
    "response_formatter":"✨ Formatting response...",
    "validate":          "🔍 Validating and OCRing document...",
    "analyse":           "🧠 Performing deep clause analysis...",
}

ENTITY_ICONS = {
    "statutes": "⚖️", "cases": "🏛️", "principles": "📜",
    "courts": "🏢", "jurisdictions": "🏢",
    "parties": "👥", "dates": "📅",
}


def _safe_serialize(obj):
    """Convert any non-serializable objects to plain types."""
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v) for v in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)


async def run_agent_stream(input_data: dict, thread_id: str) -> AsyncGenerator[str, None]:
    """
    Streams real-time events from the LangGraph agent using astream().
    
    Uses graph.astream() which yields state diffs per node completion.
    This avoids deepcopy/serialization issues from astream_events.
    
    Event types sent:
      - "status"       : Agent progress messages
      - "node_start"   : When a graph node begins executing
      - "research_card": Flash card data (acts, sections, summaries)
      - "risk_flag"    : Identified risk flags
      - "citations"    : Source citations
      - "final"        : Complete final response
      - "error"        : Error messages
    """
    request_id = f"req_{uuid.uuid4().hex[:16]}"
    runtime_cfg = Configuration()
    model_used = runtime_cfg.get_model_for_agent("research")
    if input_data.get("mode") == "chat":
        model_used = runtime_cfg.get_model_for_agent("chat")
    elif input_data.get("mode") == "document":
        model_used = runtime_cfg.get_model_for_agent("document")
    elif input_data.get("mode") == "draft":
        model_used = runtime_cfg.get_model_for_agent("draft")

    config = build_runtime_config(
        thread_id,
        request_id=request_id,
        mode=input_data.get("mode", "auto"),
        model_used=model_used,
        extra_callbacks=[METRICS_COLLECTOR],
    )
    
    accumulated_entities = {}
    accumulated_citations = []
    last_final_response = ""
    
    # Update thread store
    if thread_id not in threads_store:
        threads_store[thread_id] = {
            "title": input_data.get("query", "")[:60],
            "mode": input_data.get("mode", "auto"),
            "created_at": datetime.utcnow().isoformat(),
            "message_count": 0,
        }
    threads_store[thread_id]["message_count"] = threads_store[thread_id].get("message_count", 0) + 1

    yield sse_event("status", {"content": "Initializing Vidhijna Multi-Agent System..."})

    request_started = datetime.utcnow()
    METRICS_COLLECTOR.begin_request(request_id)
    try:
        # astream yields {node_name: state_update_dict} for each completed node
        async for chunk in graph.astream(input_data, config, stream_mode="updates"):
            if not isinstance(chunk, dict):
                continue

            for node_name, update in chunk.items():
                if not isinstance(update, dict):
                    continue
                
                # ── Announce node ────────────────────────────────────────
                label = NODE_LABELS.get(node_name, "")
                if label:
                    yield sse_event("node_start", {"node": node_name, "content": label})

                # ── Stream status_log messages ───────────────────────────
                status_log = update.get("status_log")
                if status_log and isinstance(status_log, list):
                    for msg in status_log:
                        if msg and isinstance(msg, str):
                            yield sse_event("status", {"content": msg})

                # ── Stream legal entities as flash cards ─────────────────
                entities = update.get("legal_entities")
                if entities and isinstance(entities, dict):
                    accumulated_entities.update(entities)
                    for category, items in entities.items():
                        if items and isinstance(items, list):
                            for item in items[:5]:
                                yield sse_event("research_card", {
                                    "category": category,
                                    "content": str(item),
                                    "icon": ENTITY_ICONS.get(category, "📌")
                                })

                # ── Stream risk flags ────────────────────────────────────
                risk_flags = update.get("risk_flags")
                if risk_flags and isinstance(risk_flags, list):
                    for flag in risk_flags:
                        yield sse_event("risk_flag", {
                            "content": str(flag),
                            "severity": "high"
                        })

                # ── Stream citations ─────────────────────────────────────
                citations = update.get("citations")
                if citations and isinstance(citations, list):
                    new_cites = [str(c) for c in citations if str(c) not in accumulated_citations]
                    accumulated_citations.extend(new_cites)
                    if new_cites:
                        yield sse_event("citations", {"items": new_cites})

                # ── Stream summaries as research cards ───────────────────
                for skey, slabel, sicon in [
                    ("legal_summary", "Statutory Analysis", "⚖️"),
                    ("books_summary", "Legal Commentary", "📚"),
                    ("web_summary",   "Web Research",      "🌐"),
                ]:
                    summary = update.get(skey)
                    if summary and isinstance(summary, str) and len(summary) > 50:
                        yield sse_event("research_card", {
                            "category": slabel,
                            "content": summary[:250] + "..." if len(summary) > 250 else summary,
                            "icon": sicon,
                            "full": True,
                        })

                # ── Stream draft preview ─────────────────────────────────
                draft = update.get("draft_output")
                if draft and isinstance(draft, str):
                    yield sse_event("draft_preview", {"content": draft[:500]})

                # ── Capture final response ───────────────────────────────
                final = update.get("final_response")
                if final and isinstance(final, str):
                    last_final_response = final

        # ── After stream completes, send the final event ─────────────────
        if last_final_response:
            yield sse_event("final", {
                "content": last_final_response,
                "citations": _safe_serialize(accumulated_citations),
                "entities": _safe_serialize(accumulated_entities),
                "mode": input_data.get("mode", "auto"),
                "thread_id": thread_id,
            })
        else:
            yield sse_event("final", {
                "content": "⚠️ No response was generated. The agent may have encountered an issue.",
                "citations": [],
                "entities": {},
                "mode": input_data.get("mode", "auto"),
                "thread_id": thread_id,
            })

    except Exception as e:
        error_detail = str(e)
        print(f"[VIDHIJNA ERROR] {traceback.format_exc()}")
        yield sse_event("error", {"content": f"Agent error: {error_detail}"})
        yield sse_event("final", {
            "content": f"⚠️ An error occurred during processing:\n\n```\n{error_detail}\n```\n\nPlease check that your API keys are configured in the `.env` file.",
            "citations": [],
            "entities": {},
            "mode": input_data.get("mode", "auto"),
            "thread_id": thread_id,
        })
    finally:
        request_latency_ms = (datetime.utcnow() - request_started).total_seconds() * 1000
        summary = METRICS_COLLECTOR.build_request_summary(
            request_id=request_id,
            thread_id=thread_id,
            mode=input_data.get("mode", "auto"),
            reflection_loop_count=int(input_data.get("reflection_loops", 0) or 0),
            latency_ms=request_latency_ms,
            model=model_used,
        )
        METRICS_COLLECTOR.log_request_summary(summary)
        try:
            from langfuse import get_client
            get_client().flush()
        except Exception:
            pass
        if Configuration().dev_mode:
            print(
                "[METRICS] "
                f"thread={thread_id} "
                f"mode={summary['mode']} "
                f"latency_ms={summary['latency_ms']} "
                f"loops={summary['reflection_loop_count']} "
                f"tokens={summary['total_tokens']} "
                f"cost_usd={summary['cost_usd']} "
                f"model={summary['model']}"
            )
        # Free uploaded file bytes and reclaim memory (critical for 512MB RAM)
        if "uploaded_file_bytes" in input_data:
            input_data["uploaded_file_bytes"] = None
        gc.collect()

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(request: ChatRequest):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    input_data = {
        "query": request.query.strip(),
        "thread_id": request.thread_id,
        "mode": request.mode,
        "draft_type": request.draft_type,
        "draft_inputs": request.draft_inputs,
        "reflection_loops": request.reflection_loops,  # User-controlled loop depth
    }
    
    return StreamingResponse(
        run_agent_stream(input_data, request.thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    thread_id: str = Form(default_factory=lambda: f"t_{uuid.uuid4().hex[:12]}"),
    query: str = Form(""),
):
    # Reject early if both processing slots are busy (prevent OOM)
    if process_limiter.locked():
        return JSONResponse(
            status_code=429,
            content={"message": "System busy processing documents. Please try again in a minute."},
        )

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        if len(content) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large (max 20MB)")

        input_data = {
            "query": query.strip() or "Analyze this document for legal risks and key clauses.",
            "thread_id": thread_id,
            "uploaded_file_bytes": content,
            "uploaded_file_name": file.filename,
            "uploaded_file_type": file.content_type or "",
            "mode": "document",
        }

        async def guarded_stream():
            async with process_limiter:
                async for event in run_agent_stream(input_data, thread_id):
                    yield event
            # Explicitly free file bytes after streaming completes
            nonlocal content
            del content
            gc.collect()

        return StreamingResponse(
            guarded_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Static info endpoint ──────────────────────────────────────────────────────

@app.get("/modes")
async def get_modes():
    """Return available modes and their descriptions for the frontend."""
    return {
        "modes": [
            {"id": "auto", "label": "AI Supervisor", "icon": "✨", "description": "Auto-routes to the best agent"},
            {"id": "research", "label": "Deep Research", "icon": "🔬", "description": "Multi-source legal research with reflection"},
            {"id": "chat", "label": "Legal Chat", "icon": "💬", "description": "Quick Q&A with legal context"},
            {"id": "document", "label": "Doc Analysis", "icon": "📄", "description": "Upload & analyze contracts/documents"},
            {"id": "draft", "label": "Draft Document", "icon": "✍️", "description": "Generate legal documents"},
        ],
        "draft_types": [
            {"id": "nda", "label": "Non-Disclosure Agreement"},
            {"id": "service_agreement", "label": "Service Agreement"},
            {"id": "employment", "label": "Employment Contract"},
            {"id": "lease", "label": "Lease Agreement"},
            {"id": "legal_notice", "label": "Legal Notice"},
            {"id": "nclt_petition", "label": "NCLT Petition"},
            {"id": "arbitration_notice", "label": "Arbitration Notice"},
        ]
    }


@app.get("/metrics")
async def metrics():
    return {
        "recent_requests": METRICS_COLLECTOR.request_summaries(),
        "summary": METRICS_COLLECTOR.aggregate_summary(),
        "p95_latency_ms": METRICS_COLLECTOR.latency_summary(),
    }

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
