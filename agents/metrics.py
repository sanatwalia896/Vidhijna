"""
Local metrics collection for observability and demo purposes.

This stays optional and non-blocking:
- writes compact summaries to metrics.jsonl
- can attach to LangChain/LangGraph via callbacks
- can also forward custom scores to Langfuse when configured
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Deque, Dict, Optional

from langchain_core.callbacks import BaseCallbackHandler


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round(0.95 * (len(ordered) - 1)))
    return float(ordered[index])


def _groq_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Estimate Groq cost using the prices requested for this branch.

    70B: $0.59 / 1M input tokens
    8B:  $0.06 / 1M input tokens
    """
    model_lower = (model or "").lower()
    if "70b" in model_lower:
        input_rate = float(os.environ.get("GROQ_70B_INPUT_USD_PER_1M", "0.59")) / 1_000_000
        output_rate = float(os.environ.get("GROQ_70B_OUTPUT_USD_PER_1M", "0.79")) / 1_000_000
    else:
        input_rate = float(os.environ.get("GROQ_8B_INPUT_USD_PER_1M", "0.06")) / 1_000_000
        output_rate = float(os.environ.get("GROQ_8B_OUTPUT_USD_PER_1M", "0.09")) / 1_000_000
    return (prompt_tokens * input_rate) + (completion_tokens * output_rate)


def _role_from_model(model: str) -> str:
    model_lower = (model or "").lower()
    if "70b" in model_lower or "openai/gpt-oss-20b" in model_lower:
        return "research"
    if "chat" in model_lower:
        return "chat"
    if "supervisor" in model_lower:
        return "supervisor"
    if "document" in model_lower:
        return "document"
    if "draft" in model_lower:
        return "draft"
    return "chat"


@dataclass
class MetricsCollector(BaseCallbackHandler):
    metrics_path: Path = field(default_factory=lambda: Path("metrics.jsonl"))
    max_recent: int = 100
    dev_mode: bool = True
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _recent: Deque[dict] = field(default_factory=deque, init=False, repr=False)
    _request_llm: Dict[str, list[dict]] = field(default_factory=lambda: defaultdict(list), init=False, repr=False)
    _llm_start_times: Dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _run_models: Dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _chain_start_times: Dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _chain_names: Dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _node_latencies: Dict[str, list[float]] = field(default_factory=lambda: defaultdict(list), init=False, repr=False)
    _request_windows: Dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def _safe_write(self, record: dict) -> None:
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    def _record(self, record: dict) -> None:
        self._recent.append(record)
        while len(self._recent) > self.max_recent:
            self._recent.popleft()
        self._safe_write(record)

    def on_chain_start(self, serialized: dict, inputs: dict, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id", ""))
        if run_id:
            self._chain_start_times[run_id] = time.perf_counter()
            node_name = str(kwargs.get("name") or (serialized or {}).get("name") or "chain")
            self._chain_names[run_id] = node_name

    def on_chain_end(self, outputs: dict, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id", ""))
        start = self._chain_start_times.pop(run_id, None)
        if start is None:
            return
        latency_ms = (time.perf_counter() - start) * 1000
        node_name = self._chain_names.pop(run_id, "chain")
        self._node_latencies[node_name].append(latency_ms)

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id", ""))
        if run_id:
            self._llm_start_times[run_id] = time.perf_counter()
            model_name = str(kwargs.get("invocation_params", {}).get("model", "") or kwargs.get("model", "") or "")
            if model_name:
                self._run_models[run_id] = model_name

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id", ""))
        start = self._llm_start_times.pop(run_id, None)
        latency_ms = (time.perf_counter() - start) * 1000 if start is not None else 0.0

        llm_output = getattr(response, "llm_output", {}) or {}
        usage = llm_output.get("token_usage", {}) if isinstance(llm_output, dict) else {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))

        model_name = (
            llm_output.get("model_name")
            if isinstance(llm_output, dict)
            else ""
        ) or self._run_models.pop(run_id, "") or str(kwargs.get("model", ""))

        text = ""
        generations = getattr(response, "generations", []) or []
        if generations and generations[0]:
            text = getattr(generations[0][0], "text", "") or ""

        record = {
            "timestamp": time.time(),
            "event": "llm_end",
            "model": model_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_tokens": total_tokens or _estimate_tokens(text),
            "tokens_per_second": round((total_tokens / (latency_ms / 1000.0)), 3) if latency_ms > 0 and total_tokens else 0.0,
            "latency_ms": round(latency_ms, 2),
            "cost_usd": round(_groq_cost_usd(model_name, prompt_tokens, completion_tokens), 6),
            "tags": kwargs.get("tags", []),
            "metadata": kwargs.get("metadata", {}),
        }
        request_id = str((kwargs.get("metadata") or {}).get("request_id", "") or (kwargs.get("metadata") or {}).get("langfuse_trace_id", ""))
        if request_id:
            self._request_llm[request_id].append(record)
        self._record(record)

    async def async_record_request_summary(self, summary: dict) -> None:
        async with self._lock:
            self._record(summary)

    def request_summaries(self) -> list[dict]:
        return list(self._recent)[-self.max_recent :]

    def begin_request(self, request_id: str) -> None:
        self._request_windows[request_id] = len(self._recent)

    def aggregate_summary(self) -> dict:
        recent = self.request_summaries()
        llm_rows = [row for row in recent if row.get("event") == "llm_end"]
        request_rows = [row for row in recent if "thread_id" in row]
        total_cost = sum(float(row.get("cost_usd", 0) or 0) for row in llm_rows + request_rows)
        total_tokens = sum(int(row.get("total_tokens", 0) or 0) for row in llm_rows)
        total_latency = sum(float(row.get("latency_ms", 0) or 0) for row in request_rows)
        avg_tps = 0.0
        tps_values = [float(row.get("tokens_per_second", 0) or 0) for row in llm_rows if float(row.get("tokens_per_second", 0) or 0) > 0]
        if tps_values:
            avg_tps = round(sum(tps_values) / len(tps_values), 3)
        return {
            "request_count": len(request_rows),
            "llm_call_count": len(llm_rows),
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
            "avg_tokens_per_second": avg_tps,
            "total_latency_ms": round(total_latency, 2),
            "by_model": self.model_breakdown(),
            "p95_latency_ms": self.latency_summary(),
        }

    def model_breakdown(self) -> dict[str, dict[str, float]]:
        breakdown: dict[str, dict[str, float]] = defaultdict(lambda: {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "avg_tokens_per_second": 0.0,
        })

        for row in self.request_summaries():
            if row.get("event") != "llm_end":
                continue
            model = str(row.get("model", "") or "unknown")
            role = _role_from_model(model)
            item = breakdown[role]
            item["calls"] += 1
            item["prompt_tokens"] += int(row.get("prompt_tokens", 0) or 0)
            item["completion_tokens"] += int(row.get("completion_tokens", 0) or 0)
            item["total_tokens"] += int(row.get("total_tokens", 0) or 0)
            item["cost_usd"] += float(row.get("cost_usd", 0) or 0)

        for role, item in breakdown.items():
            tps_values = [
                float(row.get("tokens_per_second", 0) or 0)
                for row in self.request_summaries()
                if row.get("event") == "llm_end" and _role_from_model(str(row.get("model", "") or "unknown")) == role and float(row.get("tokens_per_second", 0) or 0) > 0
            ]
            if tps_values:
                item["avg_tokens_per_second"] = round(sum(tps_values) / len(tps_values), 3)
            item["cost_usd"] = round(item["cost_usd"], 6)
        return dict(breakdown)

    def latency_summary(self) -> dict[str, float]:
        return {name: _p95(values) for name, values in self._node_latencies.items()}

    def log_request_summary(self, summary: dict) -> None:
        self._record(summary)

    def build_request_summary(
        self,
        *,
        request_id: str = "",
        thread_id: str,
        mode: str,
        reflection_loop_count: int,
        total_tokens: int = 0,
        cost_usd: float = 0.0,
        latency_ms: float = 0.0,
        model: str = "",
    ) -> dict:
        window_start = self._request_windows.pop(request_id, None) if request_id else None
        llm_rows = []
        if window_start is not None:
            for row in list(self._recent)[window_start:]:
                if row.get("event") == "llm_end":
                    llm_rows.append(row)
        elif request_id:
            llm_rows = self._request_llm.pop(request_id, [])

        if llm_rows:
            total_tokens = sum(int(row.get("total_tokens", 0) or 0) for row in llm_rows)
            cost_usd = sum(float(row.get("cost_usd", 0) or 0) for row in llm_rows)
            tps_values = [float(row.get("tokens_per_second", 0) or 0) for row in llm_rows if float(row.get("tokens_per_second", 0) or 0) > 0]
            avg_tps = round(sum(tps_values) / len(tps_values), 3) if tps_values else 0.0
        else:
            avg_tps = 0.0

        return {
            "timestamp": time.time(),
            "request_id": request_id,
            "thread_id": thread_id,
            "mode": mode,
            "reflection_loop_count": reflection_loop_count,
            "total_tokens": total_tokens,
            "cost_usd": round(cost_usd, 6),
            "latency_ms": round(latency_ms, 2),
            "model": model,
            "avg_tokens_per_second": avg_tps,
            "by_role": self._breakdown_rows(llm_rows),
            "p95_latency_ms": self.latency_summary(),
        }

    def _breakdown_rows(self, rows: list[dict]) -> dict[str, dict[str, float]]:
        breakdown: dict[str, dict[str, float]] = defaultdict(lambda: {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "avg_tokens_per_second": 0.0,
        })
        for row in rows:
            model = str(row.get("model", "") or "unknown")
            role = _role_from_model(model)
            item = breakdown[role]
            item["calls"] += 1
            item["prompt_tokens"] += int(row.get("prompt_tokens", 0) or 0)
            item["completion_tokens"] += int(row.get("completion_tokens", 0) or 0)
            item["total_tokens"] += int(row.get("total_tokens", 0) or 0)
            item["cost_usd"] += float(row.get("cost_usd", 0) or 0)
        for role, item in breakdown.items():
            tps_values = [
                float(row.get("tokens_per_second", 0) or 0)
                for row in rows
                if _role_from_model(str(row.get("model", "") or "unknown")) == role and float(row.get("tokens_per_second", 0) or 0) > 0
            ]
            if tps_values:
                item["avg_tokens_per_second"] = round(sum(tps_values) / len(tps_values), 3)
            item["cost_usd"] = round(item["cost_usd"], 6)
        return dict(breakdown)


METRICS_COLLECTOR = MetricsCollector(
    metrics_path=Path(os.environ.get("METRICS_PATH", "metrics.jsonl")),
    dev_mode=os.environ.get("DEV_MODE", "true").lower() in ("true", "1", "t"),
)
