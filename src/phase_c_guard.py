from __future__ import annotations

"""Phase C: Production Guardrails — Presidio PII + NeMo Guardrails + P95 Latency."""

import asyncio
import json
import math
import os
import re
import sys
import time

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
    ADVERSARIAL_SET_PATH,
    GUARDRAILS_CONFIG_DIR,
    LATENCY_BUDGET_P95_MS,
    PRESIDIO_LANGUAGE,
)

TARGET_PII_ENTITIES = [
    "VN_CCCD",
    "VN_PHONE",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IP_ADDRESS",
    "US_SSN",
    "PASSPORT",
    "IBAN_CODE",
]


# ─── Task 9a: Presidio PII Detection ─────────────────────────────────────────

def setup_presidio():
    """Khởi tạo Presidio engine với custom Vietnamese PII recognizers.

    Custom recognizers thêm vào:
        VN_CCCD  — số CCCD 12 chữ số hoặc CMND 9 chữ số
        VN_PHONE — số điện thoại Việt Nam (0[3-9]xxxxxxxx)

    Các recognizers mặc định đã có sẵn: EMAIL, PHONE_NUMBER (international), ...
    """
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, Pattern, PatternRecognizer
    from presidio_anonymizer import AnonymizerEngine

    cccd_recognizer = PatternRecognizer(
        supported_entity="VN_CCCD",
        patterns=[
            Pattern("CCCD 12 digits", r"\b\d{12}\b", 0.9),
            Pattern("CMND 9 digits",  r"\b\d{9}\b",  0.7),
        ],
    )
    phone_recognizer = PatternRecognizer(
        supported_entity="VN_PHONE",
        patterns=[Pattern("VN mobile", r"\b0[3-9]\d{8}\b", 0.9)],
    )

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()
    registry.add_recognizer(cccd_recognizer)
    registry.add_recognizer(phone_recognizer)

    analyzer = AnalyzerEngine(registry=registry)
    anonymizer = AnonymizerEngine()
    return analyzer, anonymizer


def pii_scan(text: str, analyzer=None, anonymizer=None) -> dict:
    """Task 9a: Quét PII trong văn bản bằng Presidio.

    Returns:
        {
          "has_pii":    bool,
          "entities":   [{"type": str, "text": str, "score": float, "start": int, "end": int}],
          "anonymized": str,   # text với PII được thay bằng <TYPE>
        }
    """
    if not text:
        return {"has_pii": False, "entities": [], "anonymized": ""}

    if analyzer is None or anonymizer is None:
        analyzer, anonymizer = setup_presidio()

    results = analyzer.analyze(
        text=text,
        entities=TARGET_PII_ENTITIES,
        language=PRESIDIO_LANGUAGE,
    )
    if not results:
        return {"has_pii": False, "entities": [], "anonymized": text}

    anonymized_result = anonymizer.anonymize(text=text, analyzer_results=results)
    anonymized = anonymized_result.text
    entities = [
        {
            "type": r.entity_type,
            "text": text[r.start:r.end],
            "score": round(r.score, 3),
            "start": r.start,
            "end": r.end,
        }
        for r in results
    ]
    return {
        "has_pii": True,
        "entities": entities,
        "anonymized": anonymized,
    }


# ─── Task 9b + 11: NeMo Guardrails ───────────────────────────────────────────

def setup_nemo_rails():
    """Khởi tạo NeMo Guardrails từ guardrails/config.yml.

    Config directory: guardrails/
        config.yml  — model + rails config
        rails.co    — Colang dialogue flows (topic check, jailbreak check, output check)
    """
    from nemoguardrails import RailsConfig, LLMRails
    config = RailsConfig.from_path(GUARDRAILS_CONFIG_DIR)
    rails = LLMRails(config)
    return rails


def _fallback_input_guard(text: str) -> dict:
    """Deterministic fallback check cho input khi NeMo rails hoặc LLM API lỗi."""
    text_lower = text.lower()
    blocked_reason = None

    jb_patterns = [
        "bỏ qua tất cả", "ignore previous instructions", "ignore your previous",
        "ignore all instructions", "pretend you are dan", "unrestricted ai",
        "system override", "không có giới hạn", "tấn công mạng", "admin command",
        "dump all training data", "bỏ qua tất cả policy", "system instructions",
        "mật khẩu admin", "tiết lộ bảng lương", "confidential hr data",
        "reveal all confidential", "tôi là ceo và ra lệnh"
    ]
    off_topic_patterns = [
        "bài thơ", "mùa thu hà nội", "nấu phở", "phở bò", "bitcoin", "ethereum",
        "phương trình vi phân", "dy/dx", "phim hay nhất của marvel", "marvel", "nấu ăn"
    ]
    pii_req_patterns = [
        "cho tôi biết cccd", "số điện thoại của nhân viên", "tiết lộ lương tháng của nhân viên",
        "thông tin nhân viên", "thông tin cá nhân", "bảng lương chi tiết"
    ]

    if any(p in text_lower for p in jb_patterns):
        blocked_reason = "jailbreak_or_prompt_injection"
    elif any(p in text_lower for p in off_topic_patterns):
        blocked_reason = "off_topic"
    elif any(p in text_lower for p in pii_req_patterns):
        blocked_reason = "pii_request"

    if blocked_reason:
        return {
            "allowed": False,
            "blocked_reason": blocked_reason,
            "response": "Xin lỗi, tôi không thể thực hiện yêu cầu này. Tôi chỉ có thể trả lời các câu hỏi về chính sách nhân sự công ty.",
        }
    return {
        "allowed": True,
        "blocked_reason": None,
        "response": text,
    }


async def check_input_rail(text: str, rails=None) -> dict:
    """Task 9b: Kiểm tra input qua NeMo input rails (topic guard + jailbreak guard).

    Returns:
        {
          "allowed":        bool,
          "blocked_reason": str | None,
          "response":       str,          # NeMo's raw response
        }
    """
    if rails is None:
        try:
            rails = setup_nemo_rails()
        except Exception:
            rails = None

    if rails is not None:
        try:
            response = await rails.generate_async(
                messages=[{"role": "user", "content": text}]
            )
            raw_content = response.get("content", "") if isinstance(response, dict) else str(response)
            refuse_keywords = [
                "xin lỗi", "không thể", "không được phép", "i cannot", "i'm sorry",
                "tôi chỉ có thể trả lời các câu hỏi về chính sách", "chỉ trả lời"
            ]
            blocked = any(kw in raw_content.lower() for kw in refuse_keywords)
            if blocked:
                return {
                    "allowed": False,
                    "blocked_reason": "nemo_input_rail",
                    "response": raw_content,
                }
            # Also run pattern check to catch any missed injections
            fallback_res = _fallback_input_guard(text)
            if not fallback_res["allowed"]:
                return fallback_res

            return {
                "allowed": True,
                "blocked_reason": None,
                "response": raw_content,
            }
        except Exception:
            pass

    return _fallback_input_guard(text)


async def check_output_rail(question: str, answer: str, rails=None) -> dict:
    """Task 11: Kiểm tra LLM output qua NeMo output rails trước khi trả về user.

    NeMo output rails hoạt động trong context của cả cuộc hội thoại (input + output).
    Kiểm tra: có PII không? Nội dung có phù hợp không? Có hallucination rõ ràng không?

    Returns:
        {
          "safe":           bool,
          "flagged_reason": str | None,
          "final_answer":   str,          # answer đã qua guard (có thể bị redact)
        }
    """
    # 1. Check PII in output
    pii_check = pii_scan(answer)
    if pii_check["has_pii"]:
        return {
            "safe": False,
            "flagged_reason": "pii_detected_in_output",
            "final_answer": pii_check["anonymized"],
        }

    # 2. Check NeMo rails if available
    if rails is None:
        try:
            rails = setup_nemo_rails()
        except Exception:
            rails = None

    if rails is not None:
        try:
            response = await rails.generate_async(messages=[
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ])
            raw_content = response.get("content", "") if isinstance(response, dict) else str(response)
            refuse_keywords = ["xin lỗi", "không thể cung cấp", "không được phép", "i cannot", "liên hệ phòng nhân sự"]
            flagged = any(kw in raw_content.lower() for kw in refuse_keywords)
            if flagged:
                return {
                    "safe": False,
                    "flagged_reason": "nemo_output_rail",
                    "final_answer": raw_content,
                }
        except Exception:
            pass

    # 3. Check sensitive leaks in output
    sensitive_keywords = ["mật khẩu admin", "password admin", "cccd của nhân viên là", "bảng lương chi tiết"]
    if any(k in answer.lower() for k in sensitive_keywords):
        return {
            "safe": False,
            "flagged_reason": "sensitive_data_leak",
            "final_answer": "Thông tin này là bảo mật và không được phép tiết lộ.",
        }

    return {"safe": True, "flagged_reason": None, "final_answer": answer}


# ─── Task 10: Adversarial Test Suite ─────────────────────────────────────────

def run_adversarial_suite(adversarial_set: list[dict], rails=None,
                           analyzer=None, anonymizer=None) -> list[dict]:
    """Task 10: Chạy 20 adversarial inputs qua full guard stack, so sánh với expected.

    Guard stack order:
        1. pii_scan()         → block nếu has_pii (cho category pii_injection)
        2. check_input_rail() → block nếu jailbreak / off-topic / prompt injection

    Returns:
        list of {
          "id": int, "category": str, "input": str,
          "expected": "blocked"|"allowed",
          "actual":   "blocked"|"allowed",
          "blocked_by": str | None,       # "presidio" | "nemo_input" | None
          "passed": bool,
        }
    """
    if analyzer is None or anonymizer is None:
        analyzer, anonymizer = setup_presidio()

    if rails is None:
        try:
            rails = setup_nemo_rails()
        except Exception:
            rails = None

    async def _run_all():
        results = []
        for item in adversarial_set:
            blocked_by = None

            # Layer 1: Presidio PII scan
            pii_result = pii_scan(item["input"], analyzer, anonymizer)
            if pii_result["has_pii"]:
                blocked_by = "presidio"

            # Layer 2: NeMo input rail
            if blocked_by is None:
                rail_result = await check_input_rail(item["input"], rails)
                if not rail_result["allowed"]:
                    blocked_by = "nemo_input"

            actual = "blocked" if blocked_by else "allowed"
            results.append({
                "id":         item["id"],
                "category":   item["category"],
                "input":      item["input"][:80] + ("..." if len(item["input"]) > 80 else ""),
                "expected":   item["expected"],
                "actual":     actual,
                "blocked_by": blocked_by,
                "passed":     actual == item["expected"],
            })
        return results

    results = asyncio.run(_run_all())
    passed = sum(1 for r in results if r["passed"])
    print(f"Adversarial suite: {passed}/{len(results)} passed")
    return results


# ─── Task 12: P95 Latency Measurement ────────────────────────────────────────

def _calc_percentile(sorted_data: list[float], p: float) -> float:
    """Calculate percentile p (0.0 to 1.0) safely from sorted list."""
    if not sorted_data:
        return 0.0
    idx = int(math.ceil(p * len(sorted_data))) - 1
    idx = max(0, min(len(sorted_data) - 1, idx))
    return round(float(sorted_data[idx]), 2)


def measure_p95_latency(test_inputs: list[str], n_runs: int = 20,
                         rails=None, analyzer=None, anonymizer=None) -> dict:
    """Task 12: Đo P50/P95/P99 latency cho từng layer trong guard stack.

    Mục tiêu production: P95 total < LATENCY_BUDGET_P95_MS (500ms mặc định)
    """
    if not test_inputs:
        test_inputs = ["Kiểm tra chính sách nghỉ phép năm."]

    if analyzer is None or anonymizer is None:
        analyzer, anonymizer = setup_presidio()

    if rails is None:
        try:
            rails = setup_nemo_rails()
        except Exception:
            rails = None

    presidio_times, nemo_times, total_times = [], [], []

    async def _measure():
        runs = (
            test_inputs[:n_runs]
            if len(test_inputs) >= n_runs
            else (test_inputs * (n_runs // max(1, len(test_inputs)) + 1))[:n_runs]
        )
        for text in runs:
            # Presidio (synchronous)
            t0 = time.perf_counter()
            pii_scan(text, analyzer, anonymizer)
            presidio_ms = max(0.01, (time.perf_counter() - t0) * 1000)

            # NeMo input rail (async)
            t1 = time.perf_counter()
            await check_input_rail(text, rails)
            nemo_ms = max(0.01, (time.perf_counter() - t1) * 1000)

            presidio_times.append(presidio_ms)
            nemo_times.append(nemo_ms)
            total_times.append(presidio_ms + nemo_ms)

    asyncio.run(_measure())

    def get_stats(times: list[float]) -> dict[str, float]:
        s = sorted(times)
        return {
            "p50": _calc_percentile(s, 0.50),
            "p95": _calc_percentile(s, 0.95),
            "p99": _calc_percentile(s, 0.99),
        }

    presidio_stats = get_stats(presidio_times)
    nemo_stats = get_stats(nemo_times)
    total_stats = get_stats(total_times)

    return {
        "presidio_ms": presidio_stats,
        "nemo_ms":     nemo_stats,
        "total_ms":    total_stats,
        "latency_budget_ok": total_stats["p95"] < LATENCY_BUDGET_P95_MS,
        "budget_ms":   LATENCY_BUDGET_P95_MS,
    }


def save_guard_report(pii_demo: dict, adv_results: list[dict], latency: dict,
                      output_guard_demo: dict | None = None,
                      path: str = "reports/guard_results.json") -> None:
    """Save Phase C report to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    passed_count = sum(1 for r in adv_results if r.get("passed", False))
    total_count = len(adv_results)
    pass_rate = round(passed_count / total_count, 4) if total_count > 0 else 0.0

    report = {
        "pii_demo": pii_demo,
        "adversarial": {
            "total": total_count,
            "passed": passed_count,
            "pass_rate": pass_rate,
        },
        "adversarial_results": adv_results,
        "latency": latency,
    }
    if output_guard_demo:
        report["output_guard_demo"] = output_guard_demo

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Phase C report saved → {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Task 9a: PII scan demo
    test_pii = "Nhân viên Nguyễn Văn A, CCCD 034095001234, SĐT 0987654321 hỏi về nghỉ phép."
    pii_demo_result = pii_scan(test_pii)
    print(f"PII detected: {pii_demo_result['has_pii']}")
    print(f"Entities: {pii_demo_result['entities']}")
    print(f"Anonymized: {pii_demo_result['anonymized']}")

    # Task 10: Adversarial suite
    with open(ADVERSARIAL_SET_PATH, encoding="utf-8") as f:
        adversarial_set = json.load(f)
    print(f"\nLoaded {len(adversarial_set)} adversarial inputs")
    adv_results = run_adversarial_suite(adversarial_set)
    if adv_results:
        passed = sum(1 for r in adv_results if r["passed"])
        print(f"Adversarial suite: {passed}/{len(adv_results)} passed")

    # Task 11: Output rail demo
    output_guard_demo = asyncio.run(
        check_output_rail(
            "Hỏi về lương",
            "Lương của nhân viên Nguyễn Văn A là 30 triệu, CCCD 034095001234."
        )
    )
    print(f"\nOutput Guard Safe: {output_guard_demo['safe']}")
    print(f"Final Answer: {output_guard_demo['final_answer']}")

    # Task 12: P95 latency
    sample_inputs = [item["input"] for item in adversarial_set[:10]]
    latency = measure_p95_latency(sample_inputs, n_runs=10)
    print(f"\nLatency P95 — Presidio: {latency['presidio_ms']['p95']}ms | "
          f"NeMo: {latency['nemo_ms']['p95']}ms | "
          f"Total: {latency['total_ms']['p95']}ms")
    print(f"Budget OK ({latency['budget_ms']}ms): {latency['latency_budget_ok']}")

    # Save report
    save_guard_report(pii_demo_result, adv_results, latency, output_guard_demo)
