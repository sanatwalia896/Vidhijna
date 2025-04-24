import os
import streamlit as st
import pandas as pd
import time
import json
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from old_codes.new_agent import LegalResearchChatbot, Config


# ✅ Dummy OPIK metrics function (replace with your real implementation)
def evaluate_opik_metrics(response: Dict[str, Any], query: str) -> Dict[str, float]:
    """
    Placeholder: Compute OPIK metrics from the response and query.
    Replace this function with your real OPIK scoring logic.
    """
    return {
        "Hallucination Score": round(10 * (0.5 + 0.5 * hash(query) % 100 / 100), 2),
        "Relevance Score": round(10 * (0.5 + 0.5 * hash(query[::-1]) % 100 / 100), 2),
        "Correctness Score": round(
            10 * (0.5 + 0.5 * hash(query + "correct") % 100 / 100), 2
        ),
        "Context Utilization": round(
            10 * (0.5 + 0.5 * hash(query + "context") % 100 / 100), 2
        ),
        "Context Precision": round(
            10 * (0.5 + 0.5 * hash(query + "precision") % 100 / 100), 2
        ),
    }


class GroqModelEvaluator:
    def __init__(self):
        self.models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
            "llama-guard-3-8b",
            "llama3-70b-8192",
            "allam-2-7b",
            "deepseek-r1-distill-llama-70b",
            "qwen-qwq-32b",
            "compound-beta",
        ]
        self.test_queries = [
            "What are the key provisions of the Indian Contract Act regarding breach of contract?",
            "Explain the legal framework for corporate insolvency in India.",
            "What are the requirements for valid patents in India?",
            "How does Indian law handle cases of negligence in medical practice?",
        ]
        self.results = {}

        if os.path.exists("model_evaluation_results.json"):
            try:
                with open("model_evaluation_results.json", "r") as f:
                    self.results = json.load(f)
            except Exception as e:
                st.error(f"Error loading previous results: {e}")

    def evaluate_model(self, model_name: str, query: str) -> Dict[str, Any]:
        Config.GROQ_MODEL = model_name
        chatbot = LegalResearchChatbot()
        start_time = time.time()
        response = chatbot.process_query(query)
        end_time = time.time()

        opik_scores = evaluate_opik_metrics(response, query)

        return {
            "model": model_name,
            "query": query,
            "response": response,
            "Response Time": end_time - start_time,
            "summary": response.get("summary", ""),
            **opik_scores,
        }

    def run_evaluations(
        self, selected_models: List[str], selected_queries: List[str]
    ) -> List[Dict[str, Any]]:
        all_results = []
        total = len(selected_models) * len(selected_queries)
        progress_bar = st.progress(0)
        status_text = st.empty()
        completed = 0

        for model in selected_models:
            for query in selected_queries:
                key = f"{model}_{query}"
                if key not in self.results:
                    status_text.text(f"Evaluating {model} on query: {query[:50]}...")
                    result = self.evaluate_model(model, query)
                    self.results[key] = result
                    with open("model_evaluation_results.json", "w") as f:
                        json.dump(self.results, f, indent=2)
                all_results.append(self.results[key])
                completed += 1
                progress_bar.progress(completed / total)

        status_text.text("Evaluation complete!")
        return all_results


def main():
    st.set_page_config(
        page_title="Groq Model Evaluator with OPIK",
        page_icon="📊",
        layout="wide",
    )
    st.title("📊 Vidhijan groq Models Evaluator using OPIK ")
    st.markdown("This evaluates LLMs on legal queries using automated OPIK metrics.")

    evaluator = GroqModelEvaluator()
    tab1, tab2 = st.tabs(["Run Evaluations", "Results Summary"])

    with tab1:
        st.header("Evaluate Models")
        col1, col2 = st.columns(2)

        with col1:
            selected_models = st.multiselect(
                "Select models:", evaluator.models, default=evaluator.models[:1]
            )

        with col2:
            default_queries = evaluator.test_queries
            custom_query = st.text_area("Add a custom query (optional):", height=100)
            if custom_query and custom_query not in default_queries:
                default_queries.append(custom_query)

            selected_queries = st.multiselect(
                "Select test queries:", default_queries, default=default_queries[:1]
            )

        if st.button("Run Evaluations", type="primary"):
            if not selected_models or not selected_queries:
                st.error("Please select at least one model and one query.")
            else:
                with st.spinner("Evaluating..."):
                    results = evaluator.run_evaluations(
                        selected_models, selected_queries
                    )
                st.success("Evaluations complete!")
                st.session_state.latest_results = results

    with tab2:
        st.header("Evaluation Results")
        if evaluator.results:
            data = []
            for result in evaluator.results.values():
                data.append(
                    {
                        "Model": result["model"],
                        "Query": (
                            (result["query"][:50] + "...")
                            if len(result["query"]) > 50
                            else result["query"]
                        ),
                        "Response Time": round(result["Response Time"], 2),
                        "Hallucination Score": result["Hallucination Score"],
                        "Relevance Score": result["Relevance Score"],
                        "Correctness Score": result["Correctness Score"],
                        "Context Utilization": result["Context Utilization"],
                        "Context Precision": result["Context Precision"],
                        "Average Score": round(
                            (
                                result["Correctness Score"]
                                + result["Relevance Score"]
                                + result["Context Utilization"]
                                + result["Context Precision"]
                                + (10 - result["Hallucination Score"])
                            )
                            / 5,
                            2,
                        ),
                    }
                )

            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)

            st.subheader("📈 Model Performance Summary")
            summary = df.groupby("Model").mean(numeric_only=True).reset_index()
            st.dataframe(summary.round(2), use_container_width=True)

            st.subheader("📊 Visualization")
            fig, ax = plt.subplots(figsize=(10, 6))
            width = 0.15
            metrics = [
                "Correctness Score",
                "Relevance Score",
                "Context Utilization",
                "Context Precision",
                "Average Score",
            ]
            models = summary["Model"]
            x = range(len(models))
            for i, metric in enumerate(metrics):
                ax.bar([p + i * width for p in x], summary[metric], width, label=metric)

            ax.set_xticks([p + width for p in x])
            ax.set_xticklabels(models, rotation=45, ha="right")
            ax.set_ylabel("Score")
            ax.set_ylim(0, 10)
            ax.set_title("Model Comparison (OPIK Metrics)")
            ax.legend()
            st.pyplot(fig)

            st.subheader("📁 Export Results")
            st.download_button(
                label="Download as CSV",
                data=df.to_csv(index=False),
                file_name="opik_model_evaluation.csv",
                mime="text/csv",
            )
        else:
            st.info("No evaluations found. Please run evaluations first.")


if __name__ == "__main__":
    main()
