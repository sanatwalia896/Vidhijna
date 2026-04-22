"""
metrics_rag.py — Full RAGAS evaluation for Vidhijna

Features:
  - Sequential execution with 15s cooldown (Fixes Groq 429 Rate Limits)
  - LangchainLLMWrapper (Fixes instructor/mistralai ImportError)
  - Full archival reporting & latest report sync
  - Regression detection against previous runs
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Path fix: add project root to sys.path ────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# ── Project imports ───────────────────────────────────────────────────────────
from agents.graph import build_runtime_config, graph
from agents.tools.retrieval import retrieve_books, retrieve_legal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPORT_DIR    = ROOT / "eval_results"
REPORT_LATEST = REPORT_DIR / "rag_eval_report.json"
DATASET_NAME  = "vidhijna-ragas-eval"

RAGAS_JUDGE_MODEL    = os.environ.get("RAGAS_JUDGE_MODEL", "llama-3.3-70b-versatile")
CONTEXT_CHAR_LIMIT   = int(os.environ.get("RAGAS_CONTEXT_CHAR_LIMIT", "600"))
REGRESSION_THRESHOLD = float(os.environ.get("RAGAS_REGRESSION_THRESHOLD", "0.10"))


@dataclass
class EvalSample:
    name:      str
    mode:      str
    thread_id: str
    query:     str
    reference: str


SAMPLES: list[EvalSample] = [
    EvalSample(
        name="deep_research",
        mode="research",
        thread_id="eval_research_director_duties",
        query=(
            "Legal Analysis: Director Duties and Oppression/Mis-management "
            "Remedies under the Companies Act, 2013 (with recent legal context)"
        ),
        reference=(
            "A strong answer should explain directors' statutory duties under "
            "Section 166 of the Companies Act, 2013, then discuss oppression "
            "and mismanagement remedies under Sections 241 and 242, including "
            "the threshold for bringing an application under Section 244 and "
            "the kinds of relief the NCLT can grant."
        ),
    ),
    EvalSample(
        name="chat",
        mode="chat",
        thread_id="eval_chat_section166",
        query="What is Section 166 of the Companies Act 2013 in simple terms?",
        reference=(
            "Section 166 imposes core duties on directors: act in good faith, "
            "exercise due and reasonable care, avoid conflicts of interest, not "
            "achieve undue gain, and use independent judgment."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Diagnostics & Collection
# ---------------------------------------------------------------------------

def _compact(text: str) -> str:
    return " ".join(text.split())[:CONTEXT_CHAR_LIMIT]


def _collect_context(query: str, mode: str) -> tuple[list[str], dict]:
    if mode == "chat":
        raw = retrieve_legal(query=query, top_k=4, top_n=2)
    else:
        raw = [
            *retrieve_legal(query=query, top_k=4, top_n=2),
            *retrieve_books(query=query, top_k=3, top_n=1),
        ]

    diag: dict[str, Any] = {
        "raw_chunk_count":   len(raw),
        "empty_dropped":     0,
        "truncated":         0,
        "final_chunk_count": 0,
        "avg_length_chars":  0,
    }

    contexts: list[str] = []
    for chunk in raw:
        meta = chunk.get("metadata", {})
        text = (meta.get("text") or meta.get("page_content") or chunk.get("text") or "").strip()

        if not text:
            diag["empty_dropped"] += 1
            continue
        if len(text) > CONTEXT_CHAR_LIMIT:
            diag["truncated"] += 1
        contexts.append(_compact(text))

    diag["final_chunk_count"] = len(contexts)
    if contexts:
        diag["avg_length_chars"] = round(sum(len(c) for c in contexts) / len(contexts), 1)
    return contexts, diag


def _extract_answer(state: dict[str, Any]) -> tuple[str, bool]:
    for key in ("final_response", "final_answer", "answer", "output"):
        val = state.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip(), False
    for msg in reversed(state.get("messages", [])):
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip(), False
    return "", True


def _run_sample(sample: EvalSample) -> dict[str, Any]:
    print(f"  → Running [{sample.mode}] {sample.name} …")
    input_data = {
        "query": sample.query, "thread_id": sample.thread_id,
        "mode": sample.mode, "reflection_loops": 2 if sample.mode == "research" else 0,
    }
    config = build_runtime_config(
        sample.thread_id,
        request_id=f"eval_{sample.name}",
        mode=sample.mode,
        model_used=os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
    )

    t0 = time.perf_counter()
    state = graph.invoke(input_data, config)
    latency = round(time.perf_counter() - t0, 2)

    answer, failed = _extract_answer(state)
    contexts, diag = _collect_context(sample.query, sample.mode)

    print(f"     {'done ✓' if not failed else 'FAILED ✗'} ({latency}s) | chunks={len(contexts)} | len={len(answer)}")

    return {
        "sample_name": sample.name, "mode": sample.mode, "question": sample.query,
        "answer": answer, "contexts": contexts, "reference": sample.reference,
        "thread_id": sample.thread_id, "latency_seconds": latency,
        "pipeline_failed": failed, "answer_length_chars": len(answer),
        "context_diagnostics": diag,
    }


# ---------------------------------------------------------------------------
# Scoring (Fixed for dependency issues)
# ---------------------------------------------------------------------------

def _score(rows: list[dict[str, Any]]) -> Any:
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from langchain_groq import ChatGroq
    from ragas.metrics import (
        Faithfulness, 
        AnswerRelevancy, 
        ContextPrecision, 
        ContextRecall
    )

    # 1. Manually create the wrapper
    groq_llm = ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model_name=RAGAS_JUDGE_MODEL,
        temperature=0
    )
    judge_llm = LangchainLLMWrapper(groq_llm)

    # 2. MANUALLY initialize metrics with the judge
    # This prevents RAGAS from calling its internal factory that triggers mistralai
    metrics = [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm),
        ContextPrecision(llm=judge_llm),
        ContextRecall(llm=judge_llm)
    ]

    scoreable = [r for r in rows if not r["pipeline_failed"]]
    dataset = EvaluationDataset(samples=[
        SingleTurnSample(
            user_input=r["question"], 
            response=r["answer"], 
            reference=r["reference"], 
            retrieved_contexts=r["contexts"]
        ) for r in scoreable
    ])

    # 3. Pass the metrics list directly. 
    # IMPORTANT: DO NOT pass llm=judge_llm here, as evaluate() might use it 
    # to trigger the factory again. The metrics already have the judge now.
    return evaluate(
        dataset,
        metrics=metrics
    )


# ---------------------------------------------------------------------------
# Report Building
# ---------------------------------------------------------------------------

def _to_float(v: Any) -> float | None:
    try: return round(float(v), 4)
    except: return None

def _build_summary(result: Any, rows: list[dict]) -> dict[str, Any]:
    table = result.to_pandas()
    metric_cols = [c for c in table.columns if c not in {"user_input", "retrieved_contexts", "response", "reference"}]
    
    averages = {col: round(table[col].mean(), 4) for col in metric_cols}
    
    scoreable = [r for r in rows if not r["pipeline_failed"]]
    per_sample = []
    for i, (_, df_row) in enumerate(table.iterrows()):
        orig = scoreable[i]
        per_sample.append({
            "sample_name": orig["sample_name"], "mode": orig["mode"],
            "latency": orig["latency_seconds"], "pipeline_failed": False,
            "faithfulness": _to_float(df_row.get("faithfulness")),
            "relevancy": _to_float(df_row.get("answer_relevancy")),
            "precision": _to_float(df_row.get("context_precision")),
            "recall": _to_float(df_row.get("context_recall")),
        })
    
    return {"averages": averages, "per_sample": per_sample}

def _check_regressions(current: dict[str, float]) -> list[str]:
    if not REPORT_LATEST.exists(): return []
    try:
        last = json.loads(REPORT_LATEST.read_text()).get("summary", {})
        issues = []
        for m, cur in current.items():
            prev = last.get(m)
            if prev is not None and (prev - cur) > REGRESSION_THRESHOLD:
                issues.append(f"{m}: {prev:.4f} → {cur:.4f}")
        return issues
    except: return []


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"\n{'='*60}\n Vidhijna RAGAS Eval | {len(SAMPLES)} samples\n{'='*60}")
    print(f" Judge model : {RAGAS_JUDGE_MODEL}\n Project root: {ROOT}\n")

    # 1. Pipeline Run (Sequential + Cooldown)
    print("Running pipeline samples…")
    rows = []
    for i, s in enumerate(SAMPLES):
        if i > 0:
            print(f"  🕒 Cooldown: Waiting 15s to reset Groq TPM quota...")
            time.sleep(15)
        rows.append(_run_sample(s))

    failed_count = sum(1 for r in rows if r["pipeline_failed"])

    # 2. Ragas Scoring
    print("\nScoring with RAGAS…")
    result = _score(rows)
    summary = _build_summary(result, rows)
    
    # 3. Persistence
    regressions = _check_regressions(summary["averages"])
    ts = datetime.now(timezone.utc)
    payload = {
        "generated_at": ts.isoformat(),
        "summary": summary["averages"],
        "regressions": regressions,
        "per_sample": summary["per_sample"],
        "failed_count": failed_count
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    archive = REPORT_DIR / f"rag_eval_{ts.strftime('%Y%m%dT%H%M%SZ')}.json"
    archive.write_text(json.dumps(payload, indent=2))
    REPORT_LATEST.write_text(json.dumps(payload, indent=2))

    # 4. Final Print
    print(f"\n{'─'*60}\n RAGAS Scores\n{'─'*60}")
    for m, v in summary["averages"].items():
        print(f"  {m:<25s} {v:.4f} {'█' * int(v * 20)}")
    
    if regressions:
        print("\n 🔴 REGRESSIONS DETECTED")
        for r in regressions: print(f" • {r}")
    else:
        print("\n ✅ No regressions vs last run.")

if __name__ == "__main__":
    main()