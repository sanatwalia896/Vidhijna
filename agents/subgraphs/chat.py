"""
subgraphs/chat.py — Conversational chat subgraph with memory

Nodes:
  retrieve → answer → update_memory
"""

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.graph import START, END, StateGraph

from agents.state import VidhijnaState
from agents.configuration import Configuration
from agents.prompts import CHAT_PROMPT
from agents.tools.retrieval import retrieve_legal, format_chunks


def _llm(model: str, temperature: float = 0.2):
    return ChatGroq(model=model, temperature=temperature)


def retrieve_for_chat(state: VidhijnaState, config: RunnableConfig) -> dict:
    cfg = Configuration.from_runnable_config(config)
    matches = retrieve_legal(
        query=state.rewritten_query or state.query,
        top_k=4,
        score_threshold=cfg.retrieval_score_threshold,
    )
    return {"legal_chunks": matches}


def answer(state: VidhijnaState, config: RunnableConfig) -> dict:
    cfg = Configuration.from_runnable_config(config)
    llm = _llm(cfg.chat_model)

    legal_context = format_chunks(state.legal_chunks) if state.legal_chunks else "No specific sections found."

    # Use conversation history from state.messages for memory
    history = list(state.messages[-cfg.max_memory_messages:])
    history.append(SystemMessage(content=CHAT_PROMPT.format(
        query=state.query,
        legal_context=legal_context,
    )))

    result = llm.invoke(history)
    response = result.content

    return {
        "final_response": response,
        "messages":       [AIMessage(content=response)],
    }


def build_chat_graph():
    b = StateGraph(VidhijnaState)

    b.add_node("retrieve", retrieve_for_chat)
    b.add_node("answer",   answer)

    b.add_edge(START,      "retrieve")
    b.add_edge("retrieve", "answer")
    b.add_edge("answer",   END)

    return b.compile()


chat_graph = build_chat_graph()