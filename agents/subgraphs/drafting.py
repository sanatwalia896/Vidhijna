"""
subgraphs/drafting.py — Legal document drafting subgraph

Nodes:
  validate_inputs → retrieve_act_sections → draft → review
"""

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.graph import START, END, StateGraph

from agents.state import VidhijnaState
from agents.configuration import Configuration
from agents.prompts import DRAFT_PROMPT
from agents.tools.retrieval import retrieve_legal, format_chunks
from agents.tools.drafting_tools import get_relevant_acts, get_standard_clauses, validate_draft_completeness
from agents.utils import clean_thinking_tags


def _llm(model: str, temperature: float = 0.2):
    return ChatGroq(model=model, temperature=temperature)


def validate_inputs(state: VidhijnaState, config: RunnableConfig) -> dict:
    """Validate draft type is supported and inputs are sufficient."""
    cfg = Configuration.from_runnable_config(config)
    draft_type = state.draft_type or ""

    if not draft_type:
        return {
            "error":          "No draft type specified",
            "final_response": (
                f"Please specify what to draft. Supported types:\n"
                f"{', '.join(cfg.supported_draft_types)}"
            ),
        }

    if draft_type not in cfg.supported_draft_types:
        return {
            "error":          f"Unsupported draft type: {draft_type}",
            "final_response": (
                f"'{draft_type}' is not supported. Supported types:\n"
                f"{', '.join(cfg.supported_draft_types)}"
            ),
        }
    return {}


def retrieve_act_sections(state: VidhijnaState, config: RunnableConfig) -> dict:
    """Retrieve relevant legal sections for the draft type."""
    if state.error:
        return {}

    cfg = Configuration.from_runnable_config(config)

    # Build retrieval query from the structured act map in drafting_tools
    relevant_acts = get_relevant_acts(state.draft_type)
    query = f"{state.draft_type.replace('_', ' ')} {' '.join(relevant_acts)} India"

    matches = retrieve_legal(query=query, top_k=5,
                             score_threshold=cfg.retrieval_score_threshold)
    return {"legal_chunks": matches}


def draft_document(state: VidhijnaState, config: RunnableConfig) -> dict:
    """Generate the draft document."""
    if state.error:
        return {}

    cfg = Configuration.from_runnable_config(config)
    llm = _llm(cfg.groq_model, temperature=0.2)

    # Include retrieved law as context for the draft
    law_context = ""
    if state.legal_chunks:
        law_context = f"\n\nRelevant legal provisions to incorporate:\n{format_chunks(state.legal_chunks[:3])}"

    inputs_text = "\n".join(
        f"- {k}: {v}" for k, v in (state.draft_inputs or {}).items()
    ) or "Use standard template with placeholders."

    result = llm.invoke([SystemMessage(content=DRAFT_PROMPT.format(
        draft_type=state.draft_type,
        draft_inputs=inputs_text + law_context,
        jurisdiction=cfg.default_jurisdiction,
    ))])

    draft_text = clean_thinking_tags(result.content)

    return {
        "draft_output":  draft_text,
        "final_response": draft_text,
        "draft_history": [{"version": 1, "content": draft_text,
                           "type": state.draft_type}],
    }


def review_draft(state: VidhijnaState, config: RunnableConfig) -> dict:
    """Review pass — check for missing standard clauses using structured checklist."""
    if state.error or not state.draft_output:
        return {}

    cfg = Configuration.from_runnable_config(config)
    llm = _llm(cfg.groq_model, temperature=0.1)

    # Run deterministic clause check first
    missing = validate_draft_completeness(state.draft_output, state.draft_type)
    expected = get_standard_clauses(state.draft_type)

    missing_section = ""
    if missing:
        missing_list = "\n".join(f"- {c}" for c in missing)
        missing_section = f"\n\nMISSING CLAUSES detected by automated check:\n{missing_list}"

    prompt = f"""Review this {state.draft_type} draft for completeness.

Standard clauses expected for this document type:
{chr(10).join(f'- {c}' for c in expected) if expected else 'No specific checklist available.'}
{missing_section}

Check for:
1. All standard clauses listed above are present and substantive
2. No contradictory terms
3. Proper legal language
4. Jurisdiction and governing law specified
5. Dispute resolution clause

Draft:
{state.draft_output[:4000]}

If issues found, append a "## Review Notes" section listing them.
Otherwise append "## Review: Draft is complete and legally sound."

Return the full draft with review notes appended."""

    result = llm.invoke([SystemMessage(content=prompt)])
    reviewed = clean_thinking_tags(result.content)

    return {
        "draft_output":   reviewed,
        "final_response": reviewed,
    }


def build_drafting_graph():
    b = StateGraph(VidhijnaState)

    b.add_node("validate",        validate_inputs)
    b.add_node("retrieve_law",    retrieve_act_sections)
    b.add_node("draft",           draft_document)
    b.add_node("review",          review_draft)

    b.add_edge(START,        "validate")
    b.add_edge("validate",   "retrieve_law")
    b.add_edge("retrieve_law", "draft")
    b.add_edge("draft",      "review")
    b.add_edge("review",     END)

    return b.compile()


drafting_graph = build_drafting_graph()