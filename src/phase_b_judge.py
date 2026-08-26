from __future__ import annotations

"""Phase B: LLM-as-Judge — pairwise, swap-and-average, Cohen κ, bias analysis."""

import json
import math
import os
import sys
from dataclasses import dataclass, field
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_llm_client, JUDGE_MODEL, HUMAN_LABELS_PATH, TEST_SET_PATH

_VALID_WINNERS = {"A", "B", "tie"}
_PAIRWISE_SYSTEM_PROMPT = (
    "You are an impartial RAG answer-quality judge. Return only valid JSON. "
    "Judge answers using accuracy, completeness, and conciseness. "
    "Do not assume either answer is a reference or gold answer."
)
_REFERENCE_SYSTEM_PROMPT = (
    "You evaluate a candidate answer against an explicitly provided HR policy reference. "
    "Return only valid JSON. Label 1 when the candidate is substantially correct and complete; "
    "label 0 when it is materially wrong, contradictory, or misses an essential requirement. "
    "Minor paraphrases can still receive label 1."
)


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str       # "A" | "B" | "tie"  (original order)
    winner_pass2: str       # "A" | "B" | "tie"  (after swap, ALREADY converted back)
    final_winner: str       # consensus after swap-and-average
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool  # True if both passes agree on same answer
    scores_pass1: dict = field(default_factory=dict)  # {"A": float, "B": float}
    scores_pass2: dict = field(default_factory=dict)


# ─── Task 5: Pairwise Judge ───────────────────────────────────────────────────

def _request_json(system_prompt: str, user_prompt: str, validator, operation: str):
    client = get_llm_client()
    if client is None:
        raise RuntimeError("OpenRouter client unavailable. Check OPENROUTER_API_KEY.")

    last_error = None
    for attempt in range(3):
        prompt = user_prompt
        if attempt:
            prompt += "\nReturn valid JSON only, with no markdown fences or extra text."
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            content = response.choices[0].message.content
            payload = json.loads(content)
            return validator(payload)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"{operation} failed after 3 attempts: {last_error}") from last_error


def _validate_pairwise(payload: dict) -> tuple[str, str, float, float]:
    if not isinstance(payload, dict):
        raise ValueError("response must be a JSON object")
    winner = payload.get("winner")
    reasoning = payload.get("reasoning")
    scores = payload.get("scores")
    if winner not in _VALID_WINNERS:
        raise ValueError("winner must be A, B, or tie")
    if not isinstance(reasoning, str):
        raise ValueError("reasoning must be a string")
    if not isinstance(scores, dict) or set(("A", "B")) - set(scores):
        raise ValueError("scores must contain A and B")
    validated_scores = []
    for label in ("A", "B"):
        score = scores[label]
        if isinstance(score, bool):
            raise ValueError(f"score {label} must be numeric")
        try:
            score = float(score)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"score {label} must be numeric") from exc
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"score {label} must be finite and in [0, 1]")
        validated_scores.append(score)
    return winner, reasoning, validated_scores[0], validated_scores[1]


@lru_cache(maxsize=256)
def _pairwise_judge_cached(question: str, answer_a: str, answer_b: str) -> tuple:
    prompt = f"""
Question:
{question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Return JSON with exactly these semantic fields:
{{"winner":"A|B|tie","reasoning":"short explanation","scores":{{"A":0.0,"B":0.0}}}}
"""
    return _request_json(
        _PAIRWISE_SYSTEM_PROMPT,
        prompt,
        _validate_pairwise,
        "Pairwise judge",
    )


def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Task 5: Gọi LLM để chọn answer tốt hơn (A hoặc B) theo 3 tiêu chí.

    Tiêu chí đánh giá:
        - Độ chính xác (accuracy): có khớp với thực tế chính sách không?
        - Độ đầy đủ (completeness): có trả lời đủ câu hỏi không?
        - Tính súc tích (conciseness): có thừa / thiếu thông tin không?

    Returns:
        {"winner": "A"|"B"|"tie", "reasoning": str, "scores": {"A": float, "B": float}}
    """
    winner, reasoning, score_a, score_b = _pairwise_judge_cached(
        question, answer_a, answer_b
    )
    return {
        "winner": winner,
        "reasoning": reasoning,
        "scores": {"A": score_a, "B": score_b},
    }
    # PROMPT_TEMPLATE = '''Bạn là một expert đánh giá chất lượng câu trả lời RAG.
    #
    # Câu hỏi: {question}
    #
    # Answer A:
    # {answer_a}
    #
    # Answer B:
    # {answer_b}
    #
    # Đánh giá dựa trên 3 tiêu chí: độ chính xác, đầy đủ, súc tích.
    # Trả lời JSON (chỉ JSON, không text khác):
    # {{"winner": "A" hoặc "B" hoặc "tie", "reasoning": "giải thích ngắn gọn", "scores": {{"A": 0.0-1.0, "B": 0.0-1.0}}}}
    # '''
    #
    # client = get_llm_client()
    # resp = client.chat.completions.create(
    #     model=JUDGE_MODEL,
    #     messages=[
    #         {"role": "system", "content": "Bạn là expert đánh giá RAG. Chỉ trả lời JSON."},
    #         {"role": "user",   "content": PROMPT_TEMPLATE.format(
    #             question=question, answer_a=answer_a, answer_b=answer_b)},
    #     ],
    #     response_format={"type": "json_object"},
    # )
    # return json.loads(resp.choices[0].message.content)
    return {
        "winner": winner,
        "reasoning": reasoning,
        "scores": {"A": score_a, "B": score_b},
    }


# ─── Task 6: Swap-and-Average ─────────────────────────────────────────────────

def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    """Task 6: Chạy pairwise 2 lần (hoán đổi thứ tự), lấy kết quả nhất quán.

    Lý do: LLM thường có position bias (ưu tiên answer xuất hiện trước).
    Bằng cách swap, ta phát hiện và giảm bias này.

    Logic:
        Pass 1: judge(q, A, B) → winner_1 (trong không gian A/B)
        Pass 2: judge(q, B, A) → winner_2_raw (trong không gian B/A)
        Convert: nếu winner_2_raw="A" thì thực ra là B (vì đã swap)
        Final:   nếu winner_1 == winner_2 → final = winner_1
                 nếu khác nhau → final = "tie"
    """
    pass1 = pairwise_judge(question, answer_a, answer_b)
    pass2_raw = pairwise_judge(question, answer_b, answer_a)
    swap_map = {"A": "B", "B": "A", "tie": "tie"}
    winner_pass2 = swap_map[pass2_raw["winner"]]
    final_winner = pass1["winner"] if pass1["winner"] == winner_pass2 else "tie"
    position_consistent = pass1["winner"] == winner_pass2
    scores_pass2 = {
        "A": pass2_raw["scores"]["B"],
        "B": pass2_raw["scores"]["A"],
    }
    return JudgeResult(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
        winner_pass1=pass1["winner"],
        winner_pass2=winner_pass2,
        final_winner=final_winner,
        reasoning_pass1=pass1["reasoning"],
        reasoning_pass2=pass2_raw["reasoning"],
        position_consistent=position_consistent,
        scores_pass1=pass1["scores"].copy(),
        scores_pass2=scores_pass2,
    )
    # pass1 = pairwise_judge(question, answer_a, answer_b)
    # pass2_raw = pairwise_judge(question, answer_b, answer_a)  # SWAP!
    #
    # # Convert pass2 back to original A/B space
    # swap_map = {"A": "B", "B": "A", "tie": "tie"}
    # winner_pass2 = swap_map[pass2_raw["winner"]]
    #
    # # Average: consensus only if both agree
    # if pass1["winner"] == winner_pass2:
    #     final = pass1["winner"]
    # else:
    #     final = "tie"  # disagreement = inconclusive
    #
    # position_consistent = (pass1["winner"] == winner_pass2)
    #
    # return JudgeResult(
    #     question=question, answer_a=answer_a, answer_b=answer_b,
    #     winner_pass1=pass1["winner"], winner_pass2=winner_pass2,
    #     final_winner=final,
    #     reasoning_pass1=pass1["reasoning"], reasoning_pass2=pass2_raw["reasoning"],
    #     position_consistent=position_consistent,
    #     scores_pass1=pass1["scores"],
    #     scores_pass2={"A": pass2_raw["scores"]["B"], "B": pass2_raw["scores"]["A"]},
    # )
    return JudgeResult(
        question=question, answer_a=answer_a, answer_b=answer_b,
        winner_pass1=pass1["winner"], winner_pass2=winner_pass2,
        final_winner=final_winner,
        reasoning_pass1=pass1["reasoning"], reasoning_pass2=pass2_raw["reasoning"],
        position_consistent=position_consistent,
        scores_pass1=pass1["scores"].copy(), scores_pass2=scores_pass2,
    )


# ─── Task 7: Cohen's κ ────────────────────────────────────────────────────────

def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Task 7: Tính Cohen's κ giữa LLM judge và human labels.

    Args:
        judge_labels:  nhãn từ LLM judge (0 = bad answer, 1 = good answer)
        human_labels:  nhãn từ human_labels_10q.json

    Returns:
        κ ∈ [-1, 1]
        Thang đo Landis-Koch: <0=poor, 0-0.2=slight, 0.2-0.4=fair,
                               0.4-0.6=moderate, 0.6-0.8=substantial, 0.8-1=almost perfect

    Gợi ý A — dùng scikit-learn:
        from sklearn.metrics import cohen_kappa_score
        return cohen_kappa_score(human_labels, judge_labels)

    Gợi ý B — tính tay:
        n = len(judge_labels)
        p_o = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
        p_e = (judge_labels.count(1)/n * human_labels.count(1)/n +
               judge_labels.count(0)/n * human_labels.count(0)/n)
        κ = (p_o - p_e) / (1 - p_e) if p_e != 1 else 0
        return κ
    """
    if len(judge_labels) != len(human_labels):
        raise ValueError("judge_labels and human_labels must have the same length")
    if not judge_labels:
        raise ValueError("label lists must be non-empty")
    if any(label not in (0, 1) for label in judge_labels + human_labels):
        raise ValueError("labels must contain only 0 or 1")

    n = len(judge_labels)
    p_o = sum(judge == human for judge, human in zip(judge_labels, human_labels)) / n
    p_e = (
        (judge_labels.count(1) / n) * (human_labels.count(1) / n)
        + (judge_labels.count(0) / n) * (human_labels.count(0) / n)
    )
    if p_e == 1.0:
        kappa = 1.0 if judge_labels == human_labels else 0.0
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)
    if not math.isfinite(kappa):
        raise ValueError("Cohen's kappa is not finite")
    return max(-1.0, min(1.0, kappa))


def interpret_kappa(kappa: float) -> str:
    if kappa < 0:
        return "poor"
    if kappa < 0.2:
        return "slight"
    if kappa < 0.4:
        return "fair"
    if kappa < 0.6:
        return "moderate"
    if kappa < 0.8:
        return "substantial"
    return "almost perfect"


# ─── Task 8: Bias Report ──────────────────────────────────────────────────────

def bias_report(judge_results: list[JudgeResult]) -> dict:
    """Task 8: Đo lường position bias và verbosity bias.

    Position bias: LLM chọn answer theo vị trí (A hay B) thay vì chất lượng.
        → Đo bằng % cases where position_consistent = False

    Verbosity bias: LLM ưu tiên answer dài hơn dù không chính xác hơn.
        → Đo bằng: trong các case A thắng, A có dài hơn B không? Tương tự cho B.

    Returns:
        {
          "total_judged": int,
          "position_bias_rate": float,        # 0-1, cao = bias nhiều
          "position_bias_count": int,
          "verbosity_bias": float,            # 0-1, > 0.6 = đáng lo ngại
          "verbosity_details": {
            "a_wins_a_longer": int,           # A thắng VÀ A dài hơn
            "b_wins_b_longer": int,           # B thắng VÀ B dài hơn
            "total_decisive": int,            # tổng case có winner rõ ràng
          },
          "interpretation": str,
        }
    """
    total = len(judge_results)
    position_bias_count = sum(not result.position_consistent for result in judge_results)
    position_bias_rate = position_bias_count / total if total else 0.0

    a_wins_a_longer = sum(
        result.final_winner == "A" and len(result.answer_a) > len(result.answer_b)
        for result in judge_results
    )
    b_wins_b_longer = sum(
        result.final_winner == "B" and len(result.answer_b) > len(result.answer_a)
        for result in judge_results
    )
    decisive = sum(result.final_winner in {"A", "B"} for result in judge_results)
    verbosity_bias = (
        (a_wins_a_longer + b_wins_b_longer) / decisive if decisive else 0.0
    )
    position_text = "high" if position_bias_rate > 0.3 else "low"
    interpretation = (
        f"Position bias is {position_text} ({position_bias_rate:.3f}); "
        f"verbosity correlation proxy is {verbosity_bias:.3f}. "
        "Verbosity is a correlation signal, not evidence that length caused the winner."
    )
    return {
        "total_judged": total,
        "position_bias_rate": round(position_bias_rate, 3),
        "position_bias_count": position_bias_count,
        "verbosity_bias": round(verbosity_bias, 3),
        "verbosity_details": {
            "a_wins_a_longer": a_wins_a_longer,
            "b_wins_b_longer": b_wins_b_longer,
            "total_decisive": decisive,
        },
        "interpretation": interpretation,
    }
    # total = len(judge_results)
    # if total == 0:
    #     return {"total_judged": 0, "position_bias_rate": 0.0, "verbosity_bias": 0.0}
    #
    # position_bias_count = sum(1 for r in judge_results if not r.position_consistent)
    # position_bias_rate  = position_bias_count / total
    #
    # a_wins_a_longer = sum(
    #     1 for r in judge_results
    #     if r.final_winner == "A" and len(r.answer_a) > len(r.answer_b)
    # )
    # b_wins_b_longer = sum(
    #     1 for r in judge_results
    #     if r.final_winner == "B" and len(r.answer_b) > len(r.answer_a)
    # )
    # decisive = sum(1 for r in judge_results if r.final_winner != "tie")
    # verbosity_bias = (a_wins_a_longer + b_wins_b_longer) / decisive if decisive > 0 else 0.0
    #
    # interpretation = ("Position bias cao — nên dùng swap-and-average."
    #                   if position_bias_rate > 0.3 else "Position bias thấp — judge ổn định.")
    # return {
    #     "total_judged": total, "position_bias_rate": round(position_bias_rate, 3),
    #     "position_bias_count": position_bias_count,
    #     "verbosity_bias": round(verbosity_bias, 3),
    #     "verbosity_details": {"a_wins_a_longer": a_wins_a_longer,
    #                           "b_wins_b_longer": b_wins_b_longer,
    #                           "total_decisive": decisive},
    #     "interpretation": interpretation,
    # }
    return {
        "total_judged": total,
        "position_bias_rate": round(position_bias_rate, 3),
        "verbosity_bias": round(verbosity_bias, 3),
        "position_bias_count": position_bias_count,
        "verbosity_details": {
            "a_wins_a_longer": a_wins_a_longer,
            "b_wins_b_longer": b_wins_b_longer,
            "total_decisive": decisive,
        },
        "interpretation": interpretation,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def _validate_reference(payload: dict) -> tuple[int, str, float]:
    if not isinstance(payload, dict):
        raise ValueError("response must be a JSON object")
    label = payload.get("label")
    reasoning = payload.get("reasoning")
    score = payload.get("score")
    if isinstance(label, bool) or label not in (0, 1):
        raise ValueError("label must be 0 or 1")
    if not isinstance(reasoning, str):
        raise ValueError("reasoning must be a string")
    if isinstance(score, bool):
        raise ValueError("score must be numeric")
    try:
        score = float(score)
    except (TypeError, ValueError) as exc:
        raise ValueError("score must be numeric") from exc
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("score must be finite and in [0, 1]")
    return label, reasoning, score


@lru_cache(maxsize=256)
def _reference_judge_cached(
    question: str,
    model_answer: str,
    reference_answer: str,
) -> tuple:
    prompt = f"""
Question:
{question}

Candidate answer:
{model_answer}

Policy reference answer:
{reference_answer}

Return JSON with exactly these semantic fields:
{{"label":0,"reasoning":"short explanation","score":0.0}}
"""
    return _request_json(
        _REFERENCE_SYSTEM_PROMPT,
        prompt,
        _validate_reference,
        "Reference judge",
    )


def judge_answer_against_reference(
    question: str,
    model_answer: str,
    reference_answer: str,
) -> dict:
    label, reasoning, score = _reference_judge_cached(
        question, model_answer, reference_answer
    )
    return {"label": label, "reasoning": reasoning, "score": score}


def _load_judge_inputs() -> tuple[list[dict], dict[int, dict]]:
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        human_data = json.load(f)
    with open(TEST_SET_PATH, encoding="utf-8") as f:
        test_set = json.load(f)
    references = {item["id"]: item for item in test_set}
    return human_data, references


def _safe_markdown(value: str, limit: int = 120) -> str:
    return " ".join(str(value).split()).replace("|", "\\|")[:limit]


def save_bias_markdown(
    pairwise_results: list[dict],
    swap_results: list[dict],
    human_comparison: list[dict],
    kappa: float,
    kappa_label: str,
    bias: dict,
    path: str = "analysis/bias_report.md",
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        "# LLM Judge Bias Report — Phase B",
        "",
        "**Sinh viên:** [Họ Tên]  ",
        "**Ngày:** 26/08/2026  ",
        f"**Judge model:** {JUDGE_MODEL}",
        "",
        "---",
        "",
        "## 1. Pairwise Judge Results",
        "",
        "| ID | Question | Winner | Reasoning |",
        "|---:|---|---|---|",
    ]
    for item in pairwise_results:
        lines.append(
            f"| {item['question_id']} | {_safe_markdown(item['question'])} | "
            f"{item['winner']} | {_safe_markdown(item['reasoning'])} |"
        )
    lines.extend([
        "",
        "---",
        "",
        "## 2. Swap-and-Average Results",
        "",
        "| ID | Pass 1 | Pass 2 | Final | Position Consistent? |",
        "|---:|---|---|---|---|",
    ])
    for item in swap_results:
        lines.append(
            f"| {item['question_id']} | {item['winner_pass1']} | "
            f"{item['winner_pass2']} | {item['final_winner']} | "
            f"{item['position_consistent']} |"
        )
    lines.extend([
        "",
        f"**Position bias rate:** {bias['position_bias_rate']:.3f} "
        f"({bias['position_bias_count']} / {bias['total_judged']})",
        "",
        "---",
        "",
        "## 3. Cohen's κ Analysis",
        "",
        "| Question ID | Human Label | Judge Label | Agree? |",
        "|---:|---:|---:|---|",
    ])
    for item in human_comparison:
        lines.append(
            f"| {item['question_id']} | {item['human_label']} | "
            f"{item['judge_label']} | {item['agree']} |"
        )
    lines.extend([
        "",
        f"**Cohen's κ:** {kappa:.4f}",
        f"**Interpretation:** {kappa_label}",
        "",
        "---",
        "",
        "## 4. Verbosity Bias",
        "",
        f"- A wins + A longer: {bias['verbosity_details']['a_wins_a_longer']} / "
        f"{bias['verbosity_details']['total_decisive']} decisive cases",
        f"- B wins + B longer: {bias['verbosity_details']['b_wins_b_longer']} / "
        f"{bias['verbosity_details']['total_decisive']} decisive cases",
        f"- **Verbosity bias rate:** {bias['verbosity_bias']:.3f}",
        "",
        f"**Interpretation:** {bias['interpretation']}",
        "",
        "---",
        "",
        "## 5. Nhận xét chung",
        "",
        f"> Cohen's κ là {kappa:.4f} ({kappa_label}), nên judge được xem là "
        f"{'tương đối nhất quán' if kappa >= 0.6 else 'chưa đủ nhất quán'} "
        "theo thang đo đã dùng.",
        f"> Position bias rate là {bias['position_bias_rate']:.3f}, "
        f"{'vượt' if bias['position_bias_rate'] > 0.3 else 'không vượt'} ngưỡng 30%.",
        f"> Swap-and-average {'hữu ích để phát hiện bất nhất' if bias['position_bias_count'] else 'chưa cho thấy bất nhất'} "
        "trong các cặp đã đo.",
        "> Trong production, LLM judge nên được dùng như một tín hiệu đánh giá kết hợp, không phải thẩm quyền duy nhất.",
    ])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_phase_b() -> dict:
    human_data, references = _load_judge_inputs()
    pair_ids = [1, 5, 12, 29, 41, 46]
    human_by_id = {item["question_id"]: item for item in human_data}

    pairwise_results = []
    swap_results = []
    for question_id in pair_ids:
        human_item = human_by_id[question_id]
        reference = references[question_id]
        answer_a = human_item["model_answer"]
        answer_b = reference["ground_truth"]
        pairwise = pairwise_judge(human_item["question"], answer_a, answer_b)
        swapped = swap_and_average(human_item["question"], answer_a, answer_b)
        pairwise_results.append({
            "question_id": question_id,
            "question": human_item["question"],
            "winner": pairwise["winner"],
            "reasoning": pairwise["reasoning"],
            "scores": pairwise["scores"],
        })
        swap_results.append({
            "question_id": question_id,
            "winner_pass1": swapped.winner_pass1,
            "winner_pass2": swapped.winner_pass2,
            "final_winner": swapped.final_winner,
            "position_consistent": swapped.position_consistent,
            "reasoning_pass1": swapped.reasoning_pass1,
            "reasoning_pass2": swapped.reasoning_pass2,
            "scores_pass1": swapped.scores_pass1,
            "scores_pass2": swapped.scores_pass2,
        })

    # Generate all predictions before reading human labels for agreement.
    predictions = []
    for item in human_data:
        reference = references[item["question_id"]]
        prediction = judge_answer_against_reference(
            item["question"], item["model_answer"], reference["ground_truth"]
        )
        predictions.append((item, prediction))

    human_comparison = []
    for item, prediction in predictions:
        human_label = item["human_label"]
        human_comparison.append({
            "question_id": item["question_id"],
            "question": item["question"],
            "human_label": human_label,
            "judge_label": prediction["label"],
            "agree": human_label == prediction["label"],
            "judge_score": prediction["score"],
            "judge_reasoning": prediction["reasoning"],
        })

    human_labels = [item["human_label"] for item in human_comparison]
    judge_labels = [item["judge_label"] for item in human_comparison]
    kappa = cohen_kappa(judge_labels, human_labels)
    kappa_label = interpret_kappa(kappa)
    bias = bias_report([
        JudgeResult(
            question=human_by_id[item["question_id"]]["question"],
            answer_a=human_by_id[item["question_id"]]["model_answer"],
            answer_b=references[item["question_id"]]["ground_truth"],
            winner_pass1=item["winner_pass1"],
            winner_pass2=item["winner_pass2"],
            final_winner=item["final_winner"],
            reasoning_pass1=item["reasoning_pass1"],
            reasoning_pass2=item["reasoning_pass2"],
            position_consistent=item["position_consistent"],
            scores_pass1=item["scores_pass1"],
            scores_pass2=item["scores_pass2"],
        )
        for item in swap_results
    ])

    report = {
        "judge_model": JUDGE_MODEL,
        "pairwise_count": len(pairwise_results),
        "pairwise_results": pairwise_results,
        "swap_results": swap_results,
        "human_comparison": human_comparison,
        "human_labels": human_labels,
        "judge_labels": judge_labels,
        "cohen_kappa": kappa,
        "kappa_interpretation": kappa_label,
        "bias_report": bias,
    }
    os.makedirs("reports", exist_ok=True)
    with open("reports/judge_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    save_bias_markdown(
        pairwise_results, swap_results, human_comparison,
        kappa, kappa_label, bias,
    )
    return report


def _print_phase_b_summary(report: dict) -> None:
    print("===== PAIRWISE RESULTS =====")
    for item in report["pairwise_results"]:
        print(f"id={item['question_id']} winner={item['winner']} scores={item['scores']}")
    print("\n===== SWAP RESULTS =====")
    for item in report["swap_results"]:
        print(
            f"id={item['question_id']} pass1={item['winner_pass1']} "
            f"pass2={item['winner_pass2']} final={item['final_winner']} "
            f"consistent={item['position_consistent']}"
        )
    print("\n===== HUMAN VS JUDGE =====")
    for item in report["human_comparison"]:
        print(
            f"id={item['question_id']} human={item['human_label']} "
            f"judge={item['judge_label']} agree={item['agree']}"
        )
    print(f"\nCOHEN_KAPPA={report['cohen_kappa']:.4f}")
    print(f"KAPPA_INTERPRETATION={report['kappa_interpretation']}")
    bias = report["bias_report"]
    print(f"POSITION_BIAS_COUNT={bias['position_bias_count']}")
    print(f"POSITION_BIAS_RATE={bias['position_bias_rate']:.3f}")
    print(f"VERBOSITY_BIAS={bias['verbosity_bias']:.3f}")
    print(f"TOTAL_DECISIVE={bias['verbosity_details']['total_decisive']}")
    print("Saved -> reports/judge_results.json")
    print("Saved -> analysis/bias_report.md")


if __name__ == "__main__":
    _print_phase_b_summary(run_phase_b())


if False and __name__ == "__main__":
    # --- Demo pairwise + swap ---
    q   = "Nhân viên được nghỉ bao nhiêu ngày phép năm?"
    a_a = "Nhân viên được nghỉ 15 ngày phép năm theo chính sách v2024 hiện hành."
    a_b = "Theo quy định, nhân viên có 12 ngày phép hàng năm."

    print("Running swap-and-average judge...")
    result = swap_and_average(q, a_a, a_b)
    print(f"  Pass 1 winner: {result.winner_pass1}")
    print(f"  Pass 2 winner: {result.winner_pass2}")
    print(f"  Final:         {result.final_winner}")
    print(f"  Position consistent: {result.position_consistent}")

    # --- Cohen's κ vs human labels ---
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        human_data = json.load(f)
    human_labels = [item["human_label"] for item in human_data]
    print(f"\nHuman labels loaded: {len(human_labels)} questions")

    # In production: run judge on the same 10 questions to get judge_labels
    judge_labels = [0] * len(human_labels)  # placeholder — replace with real judge output
    kappa = cohen_kappa(judge_labels, human_labels)
    print(f"Cohen's κ (placeholder): {kappa:.3f}")

    # --- Bias report ---
    bias = bias_report([result])
    print(f"\nBias report: {bias}")
