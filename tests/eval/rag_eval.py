"""
Live RAG evaluation for Vidhijna using Opik.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

from opik import Opik
from opik.evaluation import evaluate
from opik.evaluation.metrics import AnswerRelevance, ContextPrecision, ContextRecall, Hallucination

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.graph import graph


REPORT_PATH = Path("eval_results/rag_eval_report.json")
DATASET_NAME = "vidhijna-rag-eval"
PROJECT_NAME = "vidhijna-rag-eval"
EXPERIMENT_NAME = "Vidhijna Live RAG Eval"


def _mk(category: str, difficulty: str, question: str, answer: str) -> Dict[str, Any]:
    return {
        "input": question if question.endswith("?") else f"{question}?",
        "expected_output": answer,
        "metadata": {"category": category, "difficulty": difficulty},
    }


def _items() -> List[Dict[str, Any]]:
    base = [
        _mk("company_law", "easy", "What are the duties of directors under Section 166 of the Companies Act 2013", "Section 166 requires directors to act in accordance with the company’s articles, in good faith, in the best interests of the company, its employees, shareholders, community and environment, and to exercise due and reasonable care, skill and diligence."),
        _mk("company_law", "medium", "When can minority shareholders seek relief for oppression and mismanagement under the Companies Act 2013", "Minority shareholders may seek relief when the affairs of the company are conducted oppressively or prejudicially to public interest or the interests of the company, commonly under Sections 241 and 242, subject to eligibility under Section 244."),
        _mk("company_law", "medium", "What remedies can NCLT grant in a petition alleging oppression and mismanagement", "The NCLT may regulate the conduct of affairs, order purchase of shares, remove or appoint directors, set aside transactions, and grant other equitable relief under Section 242 if oppression or mismanagement is established."),
        _mk("company_law", "medium", "What is the threshold for filing oppression and mismanagement proceedings under Section 244", "Section 244 sets eligibility thresholds based on shareholding or member count, with the Tribunal able to waive requirements in appropriate cases."),
        _mk("company_law", "medium", "Explain the fiduciary duties of directors and what counts as breach under Indian company law", "Directors owe duties of loyalty, good faith, care, skill and diligence, and must avoid conflicts and improper gain. Breach may include self-dealing, misuse of office, negligence, or acting for an improper purpose."),
        _mk("contract_law", "easy", "What are the key elements of a valid contract under the Indian Contract Act 1872", "A valid contract generally requires lawful offer and acceptance, lawful consideration, intention to create legal relations, free consent, capacity, lawful object, and certainty."),
        _mk("contract_law", "medium", "What damages are generally available for breach of contract in India", "The usual remedy is compensation for loss or damage that naturally arose in the usual course of things or was within the parties’ contemplation, subject to mitigation and proof."),
        _mk("contract_law", "hard", "Can a restraint of trade clause in a commercial agreement be enforced under Indian law", "Post-contract restraints are generally scrutinized and may be unenforceable if they are unreasonable or contrary to Section 27 of the Indian Contract Act, subject to narrow exceptions."),
        _mk("ip_law", "easy", "What does trademark infringement analysis focus on", "It focuses on whether the mark is used without authorization in a manner likely to cause confusion, deception, or unfair advantage, depending on the statute and facts."),
        _mk("ip_law", "medium", "How is copyright infringement usually evaluated in India", "Copyright infringement is usually evaluated by comparing protected expression, unauthorized copying, substantial similarity, and whether any statutory exception applies."),
        _mk("arbitration", "easy", "When can an arbitration clause be invoked in a commercial dispute", "An arbitration clause can be invoked when the underlying dispute falls within the clause and the dispute is arbitrable under Indian law."),
        _mk("arbitration", "medium", "What is the difference between seat and venue in arbitration", "The seat determines the juridical home of the arbitration and the supervisory court; the venue is the physical location of hearings unless otherwise agreed or interpreted differently by the clause."),
    ]

    additions = [
        ("company_law", "hard", "Explain how a legal answer should handle a Companies Act oppression question without hallucinating sections", "It should cite only directly relevant sections, avoid speculative section chains, and state uncertainty if the retrieved context is incomplete."),
        ("company_law", "easy", "Explain why Section 166 matters for director accountability", "Section 166 is the core statutory source for director duties, including good faith, care, skill, diligence, and avoiding conflicts or undue gain."),
        ("company_law", "medium", "Explain when a buyout order may be the best remedy in oppression litigation", "A buyout can be the best remedy where minority oppression or deadlock makes continuing the relationship impractical and unfair."),
        ("company_law", "hard", "Explain what facts usually distinguish mismanagement from a normal commercial dispute", "Mismanagement generally requires prejudicial conduct in governance or control, not merely poor commercial performance or business disagreement."),
        ("company_law", "medium", "Explain the practical legal significance of Section 244 in oppression petitions", "Section 244 controls who can file and whether the Tribunal may waive threshold requirements in appropriate cases."),
        ("company_law", "medium", "What is the difference between oppression and mismanagement in company law", "Oppression refers to burdensome, harsh and wrongful conduct toward members, usually minorities. Mismanagement refers to improper conduct of the company’s affairs that is prejudicial to the company, members or public interest."),
        ("company_law", "medium", "What is a practical red flag for breach of fiduciary duty by a director", "Self-dealing, undisclosed conflicts, diversion of business opportunities, or using corporate assets for personal benefit are practical red flags."),
        ("company_law", "hard", "How should a legal assistant answer a question about company law remedies without hallucinating sections", "It should cite only directly relevant statutory sections, avoid speculative section chains, and say when the law is uncertain or context dependent."),
        ("company_law", "medium", "What should a concise answer about director duties include", "It should include good faith, due care, proper purpose, avoidance of conflict, and no undue gain, ideally tied to Section 166."),
        ("company_law", "medium", "What is the role of equitable relief in oppression petitions", "Equitable relief lets the Tribunal craft practical remedies that restore fairness, control, or participation where the company’s affairs are being conducted oppressively."),
        ("company_law", "easy", "How can a company law answer stay grounded in retrieved context", "By citing only retrieved provisions, summarizing only supported holdings, and avoiding long irrelevant statutory lists or fabricated section ranges."),
        ("contract_law", "easy", "Explain the usual remedy for a breach of contract in India", "The usual remedy is compensation for loss that naturally arose from the breach or was within the parties' contemplation, subject to proof and mitigation."),
        ("contract_law", "hard", "Explain how Indian law treats an unreasonable restraint of trade clause", "Post-contract restraints are generally scrutinized under Section 27 and may be unenforceable if they are unreasonable or outside a narrow exception."),
        ("contract_law", "medium", "Explain what makes consent defective in a commercial contract", "Consent can be defective where it is affected by coercion, undue influence, fraud, misrepresentation, or mistake depending on the facts."),
        ("contract_law", "easy", "Explain why consideration matters in contract enforceability", "Consideration is a core element of enforceability because it shows the bargain and supports a valid contractual promise unless an exception applies."),
        ("contract_law", "medium", "Explain how a liquidated damages clause should be analyzed", "It should be checked for reasonableness and whether it is a genuine pre-estimate rather than a disguised penalty."),
        ("ip_law", "medium", "Explain how passing off differs from trademark infringement", "Passing off protects goodwill and misrepresentation even without registration, while infringement is the statutory claim based on registration rights."),
        ("ip_law", "medium", "Explain what the basic patentability inquiry looks like", "Patentability usually turns on novelty, inventive step, industrial applicability, and statutory exclusions under the Patents Act."),
        ("ip_law", "hard", "Explain why source quality matters in an IP answer", "The answer should stay grounded in the retrieved context and avoid inventing registrations, dates, or statutory exclusions not in the source material."),
        ("arbitration", "medium", "Explain why seat matters more than venue in arbitration", "The seat determines the juridical home and supervisory court, while venue is usually only the physical place of hearings."),
        ("arbitration", "medium", "Explain when an arbitral award may be challenged", "An award may be challenged only on limited statutory grounds such as invalid arbitration agreement, lack of notice, excess jurisdiction, or public policy issues."),
        ("arbitration", "hard", "Explain what a good arbitration enforcement answer should include", "It should identify the award, the enforcement route, available challenge grounds, and any procedural prerequisites under the Act."),
        ("arbitration", "hard", "Explain how to keep an arbitration answer grounded", "It should rely on the retrieved context, avoid unsupported procedural claims, and say when the point depends on the clause wording."),
    ]

    for category, difficulty, question, answer in additions:
        base.append(_mk(category, difficulty, question, answer))
        base.append(_mk(category, difficulty, f"Briefly {question.lower()}", answer))

    return base


def _dataset_items() -> List[Dict[str, Any]]:
    return _items()


def _collect_context(result_state: Dict[str, Any]) -> List[str]:
    context: List[str] = []
    for chunk in (result_state.get("legal_chunks") or [])[:4]:
        meta = chunk.get("metadata", {})
        act = meta.get("act_name", "")
        sec = meta.get("section_number", "")
        text = meta.get("text", "") or chunk.get("text", "") or ""
        if act and sec:
            context.append(f"{act} — Section {sec}: {text[:500]}")
        elif act:
            context.append(f"{act}: {text[:500]}")
        elif text:
            context.append(text[:500])
    for chunk in (result_state.get("book_chunks") or [])[:2]:
        meta = chunk.get("metadata", {})
        source = meta.get("source", meta.get("book_name", ""))
        text = meta.get("text", "") or chunk.get("text", "") or ""
        if source:
            context.append(f"{source}: {text[:500]}")
        elif text:
            context.append(text[:500])
    for result in (result_state.get("web_results") or [])[:2]:
        title = result.get("title", "")
        content = result.get("content", "") or result.get("snippet", "") or ""
        if title:
            context.append(f"{title}: {content[:500]}")
        elif content:
            context.append(content[:500])
    return context


def _run_task(sample: Dict[str, Any]) -> Dict[str, Any]:
    request_id = f"eval_{abs(hash(sample['input'])) % 10**12}"
    state = graph.invoke(
        {
            "query": sample["input"],
            "thread_id": request_id,
            "mode": "research",
            "reflection_loops": 2,
        },
        config={"configurable": {"thread_id": request_id}},
    )
    return {
        "input": sample["input"],
        "output": state.get("final_response") or state.get("running_summary") or "",
        "context": _collect_context(state),
        "expected_output": sample["expected_output"],
    }


def main() -> None:
    opik = Opik(
        project_name=PROJECT_NAME,
        host=os.environ.get("OPIK_HOST", "").strip() or None,
        api_key=os.environ.get("OPIK_API_KEY", "").strip() or None,
    )
    dataset = opik.get_or_create_dataset(DATASET_NAME, project_name=PROJECT_NAME)
    dataset.insert(_dataset_items())

    result = evaluate(
        dataset=dataset,
        task=_run_task,
        scoring_metrics=[AnswerRelevance(), Hallucination(), ContextPrecision(), ContextRecall()],
        experiment_name=EXPERIMENT_NAME,
        scoring_key_mapping={"expected_output": "expected_output"},
        verbose=1,
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_payload = result.model_dump() if hasattr(result, "model_dump") else json.loads(result.json())
    REPORT_PATH.write_text(json.dumps({
        "experiment_name": EXPERIMENT_NAME,
        "project_name": PROJECT_NAME,
        "dataset_name": DATASET_NAME,
        "samples": len(_dataset_items()),
        "result": result_payload,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "dataset_name": DATASET_NAME,
        "samples": len(_dataset_items()),
        "metrics": ["AnswerRelevance", "Hallucination", "ContextPrecision", "ContextRecall"],
        "report": str(REPORT_PATH),
    }, indent=2))


if __name__ == "__main__":
    main()
