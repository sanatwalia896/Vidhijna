# state.py
import operator
from dataclasses import dataclass, field
from typing_extensions import TypedDict, Annotated


@dataclass(kw_only=True)
class SummaryState:
    research_topic: str = field(default=None)  # Report topic
    search_query: str = field(default=None)  # Search query
    laws_research_results: Annotated[list, operator.add] = field(default_factory=list)
    cases_research_results: Annotated[list, operator.add] = field(default_factory=list)
    web_research_results: Annotated[list, operator.add] = field(default_factory=list)
    complete_research_results: Annotated[list, operator.add] = field(
        default_factory=list
    )
    sources_gathered: Annotated[list, operator.add] = field(default_factory=list)
    websearch_loop_count: int = field(default=0)  # Research loop count
    vectorstore_loop_count: int = field(default=0)  # Research loop count

    running_summary: str = field(default=None)  # Final report
    vector_summary: str = field(default=None)  # Vector Store  report
    websearch_summary: str = field(default=None)  # web research report


@dataclass(kw_only=True)
class SummaryStateInput:
    research_topic: str = field(default=None)  # Report topic


# @dataclass(kw_only=True)
# class SummaryStateOutput:
#     running_summary: str = field(default=None)  # Final report
@dataclass(kw_only=True)
class SummaryStateOutput:
    running_summary: str = field(default="")  # Final report
    vector_summary: str = field(default="")  # Retrieved docs summary
    websearch_summary: str = field(default="")  # Webresearch summary
