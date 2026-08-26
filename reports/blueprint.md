# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Chưa cung cấp<br>
**Ngày:** 26/08/2026

---

## Guard Stack Architecture

```
User Input
    │
    ▼ (Presidio P95 38.47ms)
[Presidio PII Scan]
    │ block if: VN_CCCD / VN_PHONE / EMAIL detected
    ▼ (NeMo input P95 2596.40ms)
[NeMo Input Rail]
    │ block if: off-topic / jailbreak / prompt injection
    ▼
[RAG Pipeline (Day 18)]
    │ M1 Chunk → M2 Search → M3 Rerank → GPT-4o-mini
    ▼
[NeMo Output Rail]
    │ flag if: PII or sensitive content is returned
    ▼
User Response
```

## Latency Budget

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget |
|---|---:|---:|---:|---:|
| Presidio PII | 22.02 | 38.47 | 38.47 | <10ms |
| NeMo Input Rail | 1177.83 | 2596.40 | 2596.40 | <300ms |
| RAG Pipeline | N/A | N/A | N/A | outside Task 12 measurement |
| NeMo Output Rail | N/A | N/A | N/A | not measured by Task 12 |
| **Total Guard** | **1204.51** | **2634.87** | **2634.87** | **<500ms** |

**Budget OK?** No<br>
The total is measured as Presidio plus NeMo input; RAG and output latency are not claimed because Task 12 does not measure them.

## CI/CD Gates

```yaml
- name: RAGAS Quality Gate
  run: python src/phase_a_ragas.py
  # weighted avg >= 0.65; weighted faithfulness >= 0.75
- name: Guardrail Gate
  run: python src/phase_c_guard.py
  # minimum >= 15/20; bonus target >= 18/20
- name: Latency Gate
  run: python -c "from src.phase_c_guard import measure_p95_latency; ..."
  # measured P95 total < 500ms
```

Observed: RAGAS average PASS, weighted faithfulness FAIL, adversarial minimum PASS, bonus PASS, latency FAIL.

## Monitoring Dashboard

| Metric | Alert Threshold | Action |
|---|---|---|
| RAGAS faithfulness | < 0.70 | Page on-call |
| Adversarial block rate | < 80% | Review new attack patterns |
| Guard P95 latency | > 600ms | Scale or optimize NeMo |
| PII detected count | spike >10/hour | Security alert |

## Kết quả thực tế từ Lab

| | Kết quả |
|---|---|
| RAGAS avg_score (50q) | 0.76 |
| Worst metric | faithfulness |
| Dominant failure distribution | adversarial |
| Cohen's κ | 1.00 |
| Adversarial pass rate | 20 / 20 |
| Guard P95 latency | 2634.87 ms |

## Nhận xét & Cải tiến

- Phase A có điểm RAGAS trung bình có trọng số 0.76; phân bố yếu nhất là adversarial và metric lỗi nổi bật là faithfulness.
- Phase B đạt Cohen's κ = 1.00, nhưng verbosity bias quan sát được là 1.00 nên chỉ nên xem đây là tín hiệu tương quan.
- Phase C chặn 20/20 adversarial cases; failed IDs được giữ nguyên trong guard report để review.
- Guard P95 thực tế là 2634.87 ms; nếu vượt ngân sách 500 ms thì NeMo là layer cần ưu tiên tối ưu hoặc scale.
- Judge và adversarial suite nên là tín hiệu kiểm soát bổ sung, không phải nguồn sự thật duy nhất.
