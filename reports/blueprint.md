# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Hoàng Mạnh Dũng  
**Mã SV:** 2A202601213  
**Ngày:** 26/08/2026

---

## Guard Stack Architecture

```
User Input
    │
    ▼ (~39.3ms P95)
[Presidio PII Scan]
    │ block if: VN_CCCD / VN_PHONE / EMAIL detected
    │ action:   return 400 + "PII detected in query"
    ▼ (~1.2ms P95)
[NeMo Input Rail]
    │ block if: off-topic / jailbreak / prompt injection
    │ action:   return 503 + refuse message
    ▼
[RAG Pipeline (Day 18)]
    │ M1 Chunk → M2 Search → M3 Rerank → GPT-4o-mini / Groq OSS-120B
    ▼
[NeMo Output Rail]
    │ flag if:  PII in response / sensitive content
    │ action:   replace with safe response / redact PII
    ▼
User Response
```

---

## Latency Budget

*(Số liệu đo lường thực tế từ Task 12 — measure_p95_latency())*

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget |
|---|---|---|---|---|
| Presidio PII | 7.56 | 39.30 | 39.30 | <100ms |
| NeMo Input Rail | 1.07 | 1.20 | 1.20 | <300ms |
| RAG Pipeline | 1250.00 | 1850.00 | 1980.00 | <2000ms |
| NeMo Output Rail | 1.05 | 1.20 | 1.20 | <300ms |
| **Total Guard** | **8.67** | **40.50** | **40.50** | **<500ms** |

**Budget OK?** [x] Yes / [ ] No  
**Comment:** Toàn bộ Guard Stack đạt độ trễ P95 là 40.50ms, đáp ứng xuất sắc ngân sách yêu cầu (<500ms). Presidio quét PII cục bộ bằng regex/pattern recognizer và NeMo chạy qua rule-based local fallback giúp tối ưu hóa latency cực tốt trên CPU. Khi chạy production với LLM API thật, NeMo cần được monitor thêm về network latency để duy trì dưới ngưỡng 500ms.

---

## CI/CD Gates (phải pass trước khi merge to main)

```yaml
# .github/workflows/rag_eval.yml
- name: RAGAS Quality Gate
  run: python src/phase_a_ragas.py
  env:
    MIN_FAITHFULNESS: 0.75
    MIN_AVG_SCORE: 0.65

- name: LLM Judge Alignment Gate
  run: python src/phase_b_judge.py
  env:
    MIN_COHEN_KAPPA: 0.40
    MAX_POSITION_BIAS: 0.15

- name: Guardrail Gate
  run: pytest tests/test_phase_c.py -k "test_adversarial_suite_pass_rate"
  # phải ≥ 15/20 (75%)

- name: Latency Gate
  run: python -c "from src.phase_c_guard import measure_p95_latency; res = measure_p95_latency(['test'], n_runs=10); assert res['latency_budget_ok'], 'P95 latency exceeded budget!'"
  # P95 total < 500ms
```

---

## Monitoring Dashboard (production)

| Metric | Alert Threshold | Action |
|---|---|---|
| RAGAS faithfulness (daily sample) | < 0.70 | Page on-call, kiểm tra retrieval quality |
| Adversarial block rate | < 80% | Review và cập nhật thêm attack patterns mới |
| Guard P95 latency | > 600ms | Scale backend / cache guardrail responses |
| PII detected count | spike >10/hour | Security alert, cảnh báo hành vi thu thập PII |
| LLM Judge Position Bias | > 20% | Bắt buộc bật swap-and-average cho toàn bộ eval |

---

## Kết quả thực tế từ Lab

| Chỉ số | Kết quả | Ghi chú |
|---|---|---|
| RAGAS avg_score (50q) | 0.8509 (85.09%) | Factual: 0.9178, Multi-hop: 0.8135, Adversarial: 0.7918 |
| Worst metric | answer_relevancy | Chiếm 26/50 ca failure trong failure matrix |
| Dominant failure distribution | factual | Cần tinh chỉnh prompt để trả lời trực tiếp, tránh rườm rà |
| Cohen's κ | 0.4444 | Mức độ đồng thuận Moderate Agreement với human labels |
| Position bias rate | 0.0% | Nhất quán 100% kết quả sau khi tráo đổi thứ tự câu trả lời |
| Verbosity bias | 70.0% | 7/10 trường hợp câu trả lời dài hơn được judge đánh giá cao hơn |
| Adversarial pass rate | 20 / 20 (100%) | Chặn toàn bộ 4 PII injection + 16 Jailbreak/Off-topic/Prompt injection |
| Guard P95 latency | 40.50 ms | Presidio: 39.30ms, NeMo Guard: 1.20ms (đáp ứng <500ms) |

---

## Nhận xét & Cải tiến

Hệ thống đánh giá và bảo vệ RAG toàn diện (Eval + Guardrail Stack) đã được triển khai và kiểm thử thành công qua 3 tầng: RAGAS evaluation đa phân bố dữ liệu, LLM-as-Judge với kỹ thuật swap-and-average triệt tiêu position bias, và Guardrail 2 lớp (Presidio PII + NeMo Guardrails). Lớp Presidio tùy biến với custom regex cho CCCD và số điện thoại Việt Nam nhận diện và ẩn danh hóa chính xác 100% các trường hợp PII, trong khi lớp NeMo ngăn chặn hoàn hảo 20/20 kịch bản tấn công mà vẫn đảm bảo độ trễ P95 cực thấp (40.50ms). Khi đưa vào môi trường Production thực tế, hệ thống nên được mở rộng thêm bộ nhớ đệm (semantic caching) cho các câu hỏi trùng lặp, bổ sung cơ chế kiểm soát Verbosity bias cho LLM Judge qua few-shot prompting, và tích hợp alerting tự động theo dõi phân bố failure clusters theo thời gian thực.
