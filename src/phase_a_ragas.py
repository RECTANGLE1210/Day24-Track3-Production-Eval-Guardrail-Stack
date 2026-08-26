from __future__ import annotations

"""Phase A: RAGAS Production Evaluation — 50q, 3 distributions, cluster analysis."""

import json
import math
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, ANSWERS_PATH

Distribution = str  # "factual" | "multi_hop" | "adversarial"

DIAGNOSTIC_TREE = {
    "faithfulness":      ("LLM hallucinating", "Tighten system prompt, lower temperature"),
    "context_recall":    ("Missing relevant chunks", "Improve chunking or add BM25"),
    "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
    "answer_relevancy":  ("Answer doesn't match question", "Improve prompt template"),
}

_DISTRIBUTIONS = ("factual", "multi_hop", "adversarial")
_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


@dataclass
class RagasResult:
    question_id: int
    distribution: Distribution
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    @property
    def avg_score(self) -> float:
        return (self.faithfulness + self.answer_relevancy +
                self.context_precision + self.context_recall) / 4

    @property
    def worst_metric(self) -> str:
        scores = {
            "faithfulness":      self.faithfulness,
            "answer_relevancy":  self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall":    self.context_recall,
        }
        return min(scores, key=scores.get)


# ─── Đã implement sẵn ────────────────────────────────────────────────────────

def load_test_set_50q(path: str = TEST_SET_PATH) -> list[dict]:
    """Load 50q test set với 3 distributions."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_answers(path: str = ANSWERS_PATH) -> list[dict]:
    """Load pre-generated answers từ setup_answers.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"answers_50q.json không tìm thấy tại {path}\n"
            "→ Chạy trước: python setup_answers.py"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_phase_a_report(results: list[RagasResult], clusters: dict,
                         path: str = "reports/ragas_50q.json") -> None:
    """Save Phase A report to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    per_dist: dict[str, dict] = {}
    for dist in _DISTRIBUTIONS:
        subset = [r for r in results if r.distribution == dist]
        count = len(subset)
        per_dist[dist] = {
            "count": count,
            "faithfulness": sum(r.faithfulness for r in subset) / count if count else 0.0,
            "answer_relevancy": sum(r.answer_relevancy for r in subset) / count if count else 0.0,
            "context_precision": sum(r.context_precision for r in subset) / count if count else 0.0,
            "context_recall": sum(r.context_recall for r in subset) / count if count else 0.0,
            "avg_score": sum(r.avg_score for r in subset) / count if count else 0.0,
        }

    report = {
        "total_questions": len(results),
        "per_distribution": per_dist,
        "failure_clusters": clusters,
        "bottom_10": bottom_10(results),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Phase A report saved -> {path}")


# ─── Tasks 1-4: Sinh viên implement ──────────────────────────────────────────

def group_by_distribution(test_set: list[dict]) -> dict[str, list[dict]]:
    """Task 1: Nhóm 50 câu hỏi theo 3 distributions.

    Returns:
        {"factual": [...], "multi_hop": [...], "adversarial": [...]}
    """
    groups = {distribution: [] for distribution in _DISTRIBUTIONS}
    for item in test_set:
        distribution = item.get("distribution")
        if distribution not in groups:
            raise ValueError(
                f"Invalid distribution {distribution!r}; expected one of {_DISTRIBUTIONS}."
            )
        groups[distribution].append(item)
    return groups


def _validated_score(value, metric: str, index: int) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid RAGAS score for {metric} at question index {index}: {value!r}"
        ) from exc
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(
            f"RAGAS score for {metric} at question index {index} must be finite and in [0, 1], "
            f"got {score!r}."
        )
    return score


def run_ragas_50q(answers: list[dict]) -> list[RagasResult]:
    """Task 2: Chạy RAGAS 4 metrics trên toàn bộ 50 câu hỏi.

    Gợi ý — import từ Day 18 của bạn:
        from src.m4_eval import evaluate_ragas

    Steps:
        1. Extract questions, answers, contexts, ground_truths từ answers list
        2. Gọi evaluate_ragas() từ m4_eval.py
        3. Kết hợp kết quả với distribution info từ answers list
        4. Return list[RagasResult]
    """
    if not answers:
        return []

    from src.m4_eval import evaluate_ragas

    questions = [a["question"] for a in answers]
    answer_texts = [a["answer"] for a in answers]
    contexts = [a["contexts"] for a in answers]
    ground_truths = [a["ground_truth"] for a in answers]
    raw = evaluate_ragas(questions, answer_texts, contexts, ground_truths)
    per_q = raw.get("per_question", []) if isinstance(raw, dict) else []
    if len(per_q) != len(answers):
        raise RuntimeError(
            f"RAGAS returned {len(per_q)} per-question results for {len(answers)} answers. "
            "Check the RAGAS/OpenRouter error above; refusing to save a fake report."
        )

    results = []
    for index, (answer, evaluation) in enumerate(zip(answers, per_q)):
        results.append(RagasResult(
            question_id=answer["id"],
            distribution=answer["distribution"],
            question=answer["question"],
            answer=answer["answer"],
            contexts=answer["contexts"],
            ground_truth=answer["ground_truth"],
            faithfulness=_validated_score(evaluation.faithfulness, "faithfulness", index),
            answer_relevancy=_validated_score(evaluation.answer_relevancy, "answer_relevancy", index),
            context_precision=_validated_score(evaluation.context_precision, "context_precision", index),
            context_recall=_validated_score(evaluation.context_recall, "context_recall", index),
        ))

    # The block below documents the original scaffold mapping; the implementation above
    # deliberately performs one batch call and refuses incomplete evaluation output.
    # Implementation:
    # try:
    #     from src.m4_eval import evaluate_ragas
    # except ImportError:
    #     print("⚠️  Không tìm thấy src/m4_eval.py — đã copy từ Day 18 chưa?")
    #     return []
    #
    # questions     = [a["question"]    for a in answers]
    # ans_texts     = [a["answer"]      for a in answers]
    # contexts      = [a["contexts"]    for a in answers]
    # ground_truths = [a["ground_truth"] for a in answers]
    #
    # raw = evaluate_ragas(questions, ans_texts, contexts, ground_truths)
    # per_q = raw.get("per_question", [])
    #
    # results = []
    # for a, pq in zip(answers, per_q):
    #     results.append(RagasResult(
    #         question_id=a["id"], distribution=a["distribution"],
    #         question=a["question"], answer=a["answer"],
    #         contexts=a["contexts"], ground_truth=a["ground_truth"],
    #         faithfulness=pq.faithfulness, answer_relevancy=pq.answer_relevancy,
    #         context_precision=pq.context_precision, context_recall=pq.context_recall,
    #     ))
    # return results
    return results


def bottom_10(results: list[RagasResult]) -> list[dict]:
    """Task 3: Lấy 10 câu hỏi có avg_score thấp nhất.

    Returns:
        [{"rank": 1, "question_id": ..., "distribution": ...,
          "question": ..., "avg_score": ..., "worst_metric": ...,
          "diagnosis": ..., "suggested_fix": ...}, ...]
    """
    output = []
    for rank, result in enumerate(sorted(results, key=lambda item: item.avg_score)[:10], start=1):
        diagnosis, suggested_fix = DIAGNOSTIC_TREE[result.worst_metric]
        output.append({
            "rank": rank,
            "question_id": result.question_id,
            "distribution": result.distribution,
            "question": result.question,
            "avg_score": round(result.avg_score, 4),
            "worst_metric": result.worst_metric,
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })
    # The block below retains the scaffold's expected output shape for reference.
    # Implementation:
    # sorted_asc = sorted(results, key=lambda r: r.avg_score)
    # bottom = sorted_asc[:10]
    # output = []
    # for i, r in enumerate(bottom):
    #     diag, fix = DIAGNOSTIC_TREE[r.worst_metric]
    #     output.append({
    #         "rank": i + 1,
    #         "question_id": r.question_id,
    #         "distribution": r.distribution,
    #         "question": r.question,
    #         "avg_score": round(r.avg_score, 4),
    #         "worst_metric": r.worst_metric,
    #         "diagnosis": diag,
    #         "suggested_fix": fix,
    #     })
    # return output
    return output


def cluster_analysis(results: list[RagasResult]) -> dict:
    """Task 4: Phân tích failure clusters theo (worst_metric × distribution).

    Mục tiêu: tìm ra distribution nào hay bị failure nhất và metric nào yếu nhất.

    Returns:
        {
          "matrix": {
            "faithfulness":      {"factual": 3, "multi_hop": 5, "adversarial": 2},
            "answer_relevancy":  {...},
            "context_precision": {...},
            "context_recall":    {...},
          },
          "dominant_failure_distribution": "multi_hop",
          "dominant_failure_metric": "context_recall",
          "insight": "..."
        }
    """
    matrix = {
        metric: {distribution: 0 for distribution in _DISTRIBUTIONS}
        for metric in _METRICS
    }
    for result in results:
        if result.worst_metric not in matrix:
            raise ValueError(f"Unknown failure metric: {result.worst_metric!r}")
        if result.distribution not in _DISTRIBUTIONS:
            raise ValueError(f"Invalid distribution: {result.distribution!r}")
        matrix[result.worst_metric][result.distribution] += 1

    if not results:
        return {
            "matrix": matrix,
            "distribution_avg_scores": {},
            "dominant_failure_distribution": None,
            "dominant_failure_metric": None,
            "insight": "No RAGAS results available for failure analysis.",
        }

    distribution_avg_scores = {
        distribution: sum(
            result.avg_score for result in results if result.distribution == distribution
        ) / sum(1 for result in results if result.distribution == distribution)
        for distribution in _DISTRIBUTIONS
        if any(result.distribution == distribution for result in results)
    }
    dominant_distribution = min(
        distribution_avg_scores,
        key=distribution_avg_scores.get,
    )
    dominant_metric = max(
        _METRICS,
        key=lambda metric: sum(matrix[metric].values()),
    )
    suggested_fix = DIAGNOSTIC_TREE[dominant_metric][1]
    insight = (
        f"'{dominant_distribution}' has the lowest mean RAGAS avg_score. "
        f"'{dominant_metric}' is the most frequent worst metric. "
        f"Suggested fix: {suggested_fix}."
    )
    # The block below retains the scaffold's intended matrix semantics for reference.
    # Implementation:
    # matrix = {
    #     metric: {"factual": 0, "multi_hop": 0, "adversarial": 0}
    #     for metric in DIAGNOSTIC_TREE
    # }
    # for r in results:
    #     matrix[r.worst_metric][r.distribution] += 1
    #
    # # Find dominant failure
    # dominant_dist   = max(["factual", "multi_hop", "adversarial"],
    #                       key=lambda d: sum(matrix[m][d] for m in matrix))
    # dominant_metric = max(matrix, key=lambda m: sum(matrix[m].values()))
    # insight = (f"Distribution '{dominant_dist}' có nhiều failure nhất. "
    #            f"Metric '{dominant_metric}' là điểm yếu chủ đạo. "
    #            f"Gợi ý: {DIAGNOSTIC_TREE[dominant_metric][1]}")
    #
    # return {"matrix": matrix, "dominant_failure_distribution": dominant_dist,
    #         "dominant_failure_metric": dominant_metric, "insight": insight}
    return {
        "matrix": matrix,
        "distribution_avg_scores": distribution_avg_scores,
        "dominant_failure_distribution": dominant_distribution,
        "dominant_failure_metric": dominant_metric,
        "insight": insight,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def _print_phase_a_summary(results: list[RagasResult], clusters: dict) -> None:
    print("\n===== RAGAS AGGREGATE =====")
    for distribution in _DISTRIBUTIONS:
        subset = [result for result in results if result.distribution == distribution]
        count = len(subset)
        print(f"{distribution}: {count}")
        if not subset:
            continue
        print(f"  faithfulness={sum(r.faithfulness for r in subset) / count:.6f}")
        print(f"  answer_relevancy={sum(r.answer_relevancy for r in subset) / count:.6f}")
        print(f"  context_precision={sum(r.context_precision for r in subset) / count:.6f}")
        print(f"  context_recall={sum(r.context_recall for r in subset) / count:.6f}")
        print(f"  avg_score={sum(r.avg_score for r in subset) / count:.6f}")

    print("\n===== BOTTOM 10 =====")
    for item in bottom_10(results):
        print(
            f"#{item['rank']} id={item['question_id']} "
            f"distribution={item['distribution']} avg_score={item['avg_score']:.4f} "
            f"worst_metric={item['worst_metric']} diagnosis={item['diagnosis']} "
            f"suggested_fix={item['suggested_fix']} question={item['question']}"
        )

    print("\n===== FAILURE CLUSTER MATRIX =====")
    print("metric               factual multi_hop adversarial total")
    for metric in _METRICS:
        cells = clusters["matrix"][metric]
        total = sum(cells.values())
        print(
            f"{metric:<20} {cells['factual']:>7} {cells['multi_hop']:>9} "
            f"{cells['adversarial']:>11} {total:>5}"
        )
    print(f"DOMINANT_DISTRIBUTION={clusters['dominant_failure_distribution']}")
    print(f"DOMINANT_METRIC={clusters['dominant_failure_metric']}")
    print(f"INSIGHT={clusters['insight']}")


if __name__ == "__main__":
    test_set = load_test_set_50q()
    groups = group_by_distribution(test_set)
    print("Distribution counts:")
    for distribution in _DISTRIBUTIONS:
        print(f"{distribution}: {len(groups[distribution])}")

    answers = load_answers()
    results = run_ragas_50q(answers)
    clusters = cluster_analysis(results)
    save_phase_a_report(results, clusters)
    _print_phase_a_summary(results, clusters)


if False and __name__ == "__main__":
    test_set = load_test_set_50q()
    print(f"Loaded {len(test_set)} questions")

    groups = group_by_distribution(test_set)
    for dist, qs in groups.items():
        print(f"  {dist}: {len(qs)} questions")

    answers = load_answers()
    results = run_ragas_50q(answers)

    if results:
        b10 = bottom_10(results)
        clusters = cluster_analysis(results)
        save_phase_a_report(results, clusters)
        print("\nBottom 10 worst questions:")
        for item in b10:
            print(f"  #{item['rank']} [{item['distribution']}] {item['question'][:50]}... "
                  f"avg={item['avg_score']:.3f} worst={item['worst_metric']}")
        print(f"\nDominant failure: {clusters.get('dominant_failure_distribution')} / "
              f"{clusters.get('dominant_failure_metric')}")
    else:
        print("⚠️  No results — implement run_ragas_50q() first.")
