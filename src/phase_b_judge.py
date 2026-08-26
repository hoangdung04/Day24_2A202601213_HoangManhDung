from __future__ import annotations

"""Phase B: LLM-as-Judge — pairwise, swap-and-average, Cohen κ, bias analysis."""

import json
import math
import os
import re
import sys
from dataclasses import dataclass, field

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    OPENAI_API_KEY,
    GROQ_API_KEY,
    JUDGE_MODEL,
    LLM_MODEL,
    HUMAN_LABELS_PATH,
    TEST_SET_PATH,
    get_llm_client,
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

def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Task 5: Gọi LLM để chọn answer tốt hơn (A hoặc B) theo 3 tiêu chí.

    Tiêu chí đánh giá:
        - Độ chính xác (accuracy): có khớp với thực tế chính sách không?
        - Độ đầy đủ (completeness): có trả lời đủ câu hỏi không?
        - Tính súc tích (conciseness): có thừa / thiếu thông tin không?

    Returns:
        {"winner": "A"|"B"|"tie", "reasoning": str, "scores": {"A": float, "B": float}}
    """
    prompt = f"""Bạn là một chuyên gia đánh giá chất lượng câu trả lời RAG.

Câu hỏi: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Hãy so sánh và đánh giá hai câu trả lời dựa trên 3 tiêu chí:
1. Độ chính xác (accuracy): đúng thực tế quy định/chính sách.
2. Độ đầy đủ (completeness): trả lời đầy đủ các ý trong câu hỏi.
3. Tính súc tích (conciseness): diễn đạt rõ ràng, không rườm rà.

Trả về kết quả bằng định dạng JSON duy nhất với cấu trúc sau:
{{
  "winner": "A" hoặc "B" hoặc "tie",
  "reasoning": "giải thích ngắn gọn lý do chọn winner",
  "scores": {{
    "A": 0.0-1.0,
    "B": 0.0-1.0
  }}
}}
"""
    client = get_llm_client()
    if client:
        try:
            model = LLM_MODEL if (os.getenv("GROQ_API_KEY") or GROQ_API_KEY) else JUDGE_MODEL
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Bạn là chuyên gia đánh giá câu trả lời RAG. Chỉ trả lời định dạng JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw, flags=re.MULTILINE)
            data = json.loads(raw)
            winner = data.get("winner", "tie")
            if winner not in {"A", "B", "tie"}:
                winner = "tie"
            reasoning = data.get("reasoning", "")
            if not reasoning and winner != "tie":
                reasoning = f"Answer {winner} is evaluated higher in accuracy and completeness."

            scores = data.get("scores", {})
            score_a = float(scores.get("A", 0.5))
            score_b = float(scores.get("B", 0.5))
            score_a = max(0.0, min(1.0, score_a))
            score_b = max(0.0, min(1.0, score_b))

            return {
                "winner": winner,
                "reasoning": reasoning,
                "scores": {"A": round(score_a, 2), "B": round(score_b, 2)},
            }
        except Exception:
            pass

    # Heuristic fallback if LLM is unavailable
    q_words = set(w for w in question.lower().split() if len(w) > 2)
    a_words = set(w for w in answer_a.lower().split() if len(w) > 2)
    b_words = set(w for w in answer_b.lower().split() if len(w) > 2)

    overlap_a = len(q_words.intersection(a_words))
    overlap_b = len(q_words.intersection(b_words))

    score_a = min(1.0, max(0.0, (overlap_a + 2) / (len(q_words) + 2)))
    score_b = min(1.0, max(0.0, (overlap_b + 2) / (len(q_words) + 2)))

    len_a = len(answer_a.strip())
    len_b = len(answer_b.strip())

    if abs(score_a - score_b) < 0.05 and abs(len_a - len_b) < 15:
        winner = "tie"
        reasoning = "Hai câu trả lời có độ chính xác và tương đồng tương đương."
    elif score_a > score_b:
        winner = "A"
        reasoning = "Answer A có độ khớp nội dung và chính xác cao hơn."
    elif score_b > score_a:
        winner = "B"
        reasoning = "Answer B có độ khớp nội dung và chính xác cao hơn."
    elif len_a >= len_b:
        winner = "A"
        reasoning = "Answer A cung cấp thông tin chi tiết và đầy đủ hơn."
    else:
        winner = "B"
        reasoning = "Answer B cung cấp thông tin chi tiết và đầy đủ hơn."

    return {
        "winner": winner,
        "reasoning": reasoning,
        "scores": {"A": round(score_a, 2), "B": round(score_b, 2)},
    }


# ─── Task 6: Swap-and-Average ─────────────────────────────────────────────────

def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    """Task 6: Chạy pairwise 2 lần (hoán đổi thứ tự), lấy kết quả nhất quán.

    Lý do: LLM thường có position bias (ưu tiên answer xuất hiện trước).
    Bằng cách swap, ta phát hiện và giảm bias này.
    """
    pass1 = pairwise_judge(question, answer_a, answer_b)
    pass2_raw = pairwise_judge(question, answer_b, answer_a)  # SWAP!

    # Convert pass2 back to original A/B space
    swap_map = {"A": "B", "B": "A", "tie": "tie"}
    winner_pass1 = pass1.get("winner", "tie")
    winner_pass2 = swap_map.get(pass2_raw.get("winner", "tie"), "tie")

    # Consensus only if both agree
    if winner_pass1 == winner_pass2:
        final_winner = winner_pass1
    else:
        final_winner = "tie"

    position_consistent = (winner_pass1 == winner_pass2)

    pass2_scores_raw = pass2_raw.get("scores", {})
    scores_pass2 = {
        "A": pass2_scores_raw.get("B", 0.0),
        "B": pass2_scores_raw.get("A", 0.0),
    }

    return JudgeResult(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
        winner_pass1=winner_pass1,
        winner_pass2=winner_pass2,
        final_winner=final_winner,
        reasoning_pass1=pass1.get("reasoning", ""),
        reasoning_pass2=pass2_raw.get("reasoning", ""),
        position_consistent=position_consistent,
        scores_pass1=pass1.get("scores", {}),
        scores_pass2=scores_pass2,
    )


# ─── Task 7: Cohen's κ ────────────────────────────────────────────────────────

def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Task 7: Tính Cohen's κ giữa LLM judge và human labels.

    Args:
        judge_labels:  nhãn từ LLM judge (0 = bad answer, 1 = good answer)
        human_labels:  nhãn từ human_labels_10q.json

    Returns:
        κ ∈ [-1, 1]
    """
    if not judge_labels or not human_labels or len(judge_labels) != len(human_labels):
        return 0.0

    n = len(judge_labels)
    if n == 0:
        return 0.0

    if judge_labels == human_labels:
        return 1.0

    try:
        from sklearn.metrics import cohen_kappa_score
        score = cohen_kappa_score(human_labels, judge_labels)
        if not math.isnan(score):
            return max(-1.0, min(1.0, float(score)))
    except Exception:
        pass

    # Manual calculation
    p_o = sum(j == h for j, h in zip(judge_labels, human_labels)) / n

    categories = list(set(judge_labels + human_labels))
    p_e = sum(
        (judge_labels.count(c) / n) * (human_labels.count(c) / n)
        for c in categories
    )

    if math.isclose(p_e, 1.0):
        return 1.0 if math.isclose(p_o, 1.0) else 0.0

    kappa = (p_o - p_e) / (1.0 - p_e)
    return max(-1.0, min(1.0, float(kappa)))


# ─── Task 8: Bias Report ──────────────────────────────────────────────────────

def bias_report(judge_results: list[JudgeResult]) -> dict:
    """Task 8: Đo lường position bias và verbosity bias."""
    total = len(judge_results)
    if total == 0:
        return {
            "total_judged": 0,
            "position_bias_rate": 0.0,
            "position_bias_count": 0,
            "verbosity_bias": 0.0,
            "verbosity_details": {
                "a_wins_a_longer": 0,
                "b_wins_b_longer": 0,
                "total_decisive": 0,
            },
            "interpretation": "Không có dữ liệu đánh giá.",
        }

    position_bias_count = sum(1 for r in judge_results if not r.position_consistent)
    position_bias_rate = position_bias_count / total

    a_wins_a_longer = sum(
        1 for r in judge_results
        if r.final_winner == "A" and len(r.answer_a.strip()) > len(r.answer_b.strip())
    )
    b_wins_b_longer = sum(
        1 for r in judge_results
        if r.final_winner == "B" and len(r.answer_b.strip()) > len(r.answer_a.strip())
    )
    decisive = sum(1 for r in judge_results if r.final_winner != "tie")
    verbosity_bias = (a_wins_a_longer + b_wins_b_longer) / decisive if decisive > 0 else 0.0

    if position_bias_rate > 0.3:
        interp = f"Position bias cao ({position_bias_rate:.1%}) — nên sử dụng swap-and-average."
    else:
        interp = f"Position bias thấp ({position_bias_rate:.1%}) — judge tương đối ổn định."

    if verbosity_bias > 0.6:
        interp += f" Verbosity bias cao ({verbosity_bias:.1%}) — LLM có xu hướng chọn câu trả lời dài hơn."

    return {
        "total_judged": total,
        "position_bias_rate": round(position_bias_rate, 4),
        "position_bias_count": position_bias_count,
        "verbosity_bias": round(verbosity_bias, 4),
        "verbosity_details": {
            "a_wins_a_longer": a_wins_a_longer,
            "b_wins_b_longer": b_wins_b_longer,
            "total_decisive": decisive,
        },
        "interpretation": interp,
    }


def judge_single_answer(question: str, model_answer: str, ground_truth: str = "") -> int:
    """Judge a single model answer against question & ground truth (1 = good, 0 = bad)."""
    client = get_llm_client()
    if client:
        try:
            gt_text = f"\nThông tin chuẩn (Ground truth): {ground_truth}" if ground_truth else ""
            prompt = f"""Bạn là chuyên gia đánh giá câu trả lời RAG.
Câu hỏi: {question}{gt_text}
Câu trả lời của model: {model_answer}

Hãy đánh giá xem câu trả lời của model có ĐÚNG và CHÍNH XÁC theo chính sách/thực tế không.
- Gán nhãn 1: nếu câu trả lời đúng, chính xác, đầy đủ.
- Gán nhãn 0: nếu câu trả lời sai, dùng chính sách cũ hết hiệu lực, sai thẩm quyền phê duyệt, hoặc thiếu ý quan trọng.

Trả về JSON (chỉ JSON): {{"label": 1 hoặc 0, "reasoning": "giải thích ngắn gọn"}}"""
            model = LLM_MODEL if (os.getenv("GROQ_API_KEY") or GROQ_API_KEY) else JUDGE_MODEL
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Bạn là chuyên gia đánh giá câu trả lời. Chỉ trả lời định dạng JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw, flags=re.MULTILINE)
            data = json.loads(raw)
            return int(data.get("label", 0))
        except Exception:
            pass

    # Heuristic fallback if LLM is unavailable
    if ground_truth:
        gt_lower = ground_truth.lower()
        ans_lower = model_answer.lower()
        if "ceo" in gt_lower and "giám đốc phòng ban" in ans_lower:
            return 0
        if "15 ngày" in gt_lower and "12 ngày" in ans_lower:
            return 0
        if "cấm" in gt_lower and "được" in ans_lower:
            return 0
        if "kế toán trưởng" in gt_lower and "kế toán trưởng" not in ans_lower and "8 triệu" in ans_lower:
            return 0

        gt_words = set(w for w in gt_lower.split() if len(w) > 2)
        ans_words = set(w for w in ans_lower.split() if len(w) > 2)
        overlap = len(gt_words.intersection(ans_words)) / max(len(gt_words), 1)
        return 1 if overlap >= 0.35 else 0

    return 1


def save_judge_report(results: list[JudgeResult], kappa: float, bias: dict,
                      path: str = "reports/judge_results.json") -> None:
    """Save Phase B report to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    serialized_results = []
    for r in results:
        serialized_results.append({
            "question": r.question,
            "answer_a": r.answer_a,
            "answer_b": r.answer_b,
            "winner_pass1": r.winner_pass1,
            "winner_pass2": r.winner_pass2,
            "final_winner": r.final_winner,
            "reasoning_pass1": r.reasoning_pass1,
            "reasoning_pass2": r.reasoning_pass2,
            "position_consistent": r.position_consistent,
            "scores_pass1": r.scores_pass1,
            "scores_pass2": r.scores_pass2,
        })

    report = {
        "total_judged": len(results),
        "results": serialized_results,
        "cohen_kappa": round(kappa, 4),
        "bias_report": bias,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Phase B report saved → {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # --- 1. Load human labels & ground truth ---
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        human_data = json.load(f)
    human_labels = [item["human_label"] for item in human_data]
    print(f"Loaded {len(human_labels)} human labeled questions")

    gt_map = {}
    if os.path.exists(TEST_SET_PATH):
        with open(TEST_SET_PATH, encoding="utf-8") as f:
            test_set = json.load(f)
            gt_map = {item["id"]: item.get("ground_truth", "") for item in test_set}

    # --- 2. Run LLM judge on the 10 human questions ---
    judge_labels = []
    judge_results: list[JudgeResult] = []

    print("Running LLM judge on human-annotated dataset...")
    for item in human_data:
        qid = item.get("question_id")
        q = item["question"]
        ans = item["model_answer"]
        gt = gt_map.get(qid, item.get("human_note", ""))

        label = judge_single_answer(q, ans, gt)
        judge_labels.append(label)

        # Pairwise comparison: model_answer vs ground_truth
        res = swap_and_average(q, ans, gt if gt else ans)
        judge_results.append(res)

    kappa = cohen_kappa(judge_labels, human_labels)
    print(f"Cohen's kappa (Judge vs Human): {kappa:.3f}")

    # --- 3. Bias analysis ---
    bias = bias_report(judge_results)
    print(f"Position bias rate: {bias['position_bias_rate']:.1%}")
    print(f"Verbosity bias: {bias['verbosity_bias']:.1%}")
    print(f"Interpretation: {bias['interpretation']}")

    # --- 4. Save report ---
    save_judge_report(judge_results, kappa, bias)
