from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import math
import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


def _zero_evaluation() -> dict:
    return {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
        "per_question": [],
    }


def _safe_float(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _build_ragas_models():
    """Build the configured OpenRouter judge and local BGE embeddings."""
    from config import LLM_BASE_URL, LLM_MODEL, OPENROUTER_API_KEY, get_llm_client

    if not OPENROUTER_API_KEY or get_llm_client() is None:
        return None, None

    from config import EMBEDDING_MODEL
    from langchain_core.embeddings import Embeddings
    from langchain_openai import ChatOpenAI
    from sentence_transformers import SentenceTransformer

    judge_llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=0,
    )

    class LocalEmbeddings(Embeddings):
        def __init__(self):
            self._model = None

        def _get_model(self):
            if self._model is None:
                self._model = SentenceTransformer(EMBEDDING_MODEL)
            return self._model

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self._get_model().encode(texts).tolist()

        def embed_query(self, text: str) -> list[float]:
            return self._get_model().encode(text).tolist()

    return judge_llm, LocalEmbeddings()


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    try:
        judge_llm, embeddings = _build_ragas_models()
        if judge_llm is None:
            return _zero_evaluation()

        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=judge_llm,
            embeddings=embeddings,
        )
        df = result.to_pandas()
        per_question = []
        for _, row in df.iterrows():
            row_contexts = row.get("contexts", [])
            if not isinstance(row_contexts, list):
                row_contexts = [] if row_contexts is None else [str(row_contexts)]
            per_question.append(EvalResult(
                question=str(row.get("question", "")),
                answer=str(row.get("answer", "")),
                contexts=row_contexts,
                ground_truth=str(row.get("ground_truth", "")),
                faithfulness=_safe_float(row.get("faithfulness", 0.0)),
                answer_relevancy=_safe_float(row.get("answer_relevancy", 0.0)),
                context_precision=_safe_float(row.get("context_precision", 0.0)),
                context_recall=_safe_float(row.get("context_recall", 0.0)),
            ))

        metric_names = [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]
        aggregate = {
            name: _safe_float(
                sum(getattr(item, name) for item in per_question) / len(per_question)
                if per_question else 0.0
            )
            for name in metric_names
        }
        return {**aggregate, "per_question": per_question}
    except Exception as e:
        print(f"  RAGAS evaluation failed: {e}")
        return _zero_evaluation()


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results or bottom_n <= 0:
        return []

    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }
    metric_names = list(diagnostic_tree)
    analyzed = []
    for result in eval_results:
        scores = {name: _safe_float(getattr(result, name, 0.0)) for name in metric_names}
        worst_metric = min(metric_names, key=lambda name: scores[name])
        average = sum(scores.values()) / len(metric_names)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        analyzed.append((average, {
            "question": result.question,
            "worst_metric": worst_metric,
            "score": scores[worst_metric],
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        }))

    analyzed.sort(key=lambda item: item[0])
    return [item[1] for item in analyzed[:bottom_n]]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
