# Failure Cluster Analysis — Phase A

**Sinh viên:** Nguyễn Công Hùng - 2A202601071<br>
**Ngày:** 26/08/2026

---

## 1. Aggregate RAGAS Scores theo Distribution

| Metric | factual | multi_hop | adversarial |
|---|---:|---:|---:|
| faithfulness | 0.8833 | 0.4417 | 0.5000 |
| answer_relevancy | 0.8032 | 0.6268 | 0.4404 |
| context_precision | 0.9167 | 0.9417 | 0.9667 |
| context_recall | 0.8750 | 0.7875 | 0.7167 |
| **avg_score** | **0.8695** | **0.6994** | **0.6559** |

---

## 2. Bottom 10 Questions

| Rank | ID | Distribution | Question | avg_score | worst_metric |
|---:|---:|---|---|---:|---|
| 1 | 39 | multi_hop | So sánh yêu cầu mật khẩu giữa policy v1.0 và v2.0 về độ dài tối thiểu, thời hạn đổi và MFA. | 0.0000 | faithfulness |
| 2 | 33 | multi_hop | Nhân viên Manager có thâm niên 12 năm: tổng phụ cấp hàng tháng và số ngày phép năm theo v2024 là bao nhiêu? | 0.3750 | faithfulness |
| 3 | 7 | factual | Nghỉ phép không lương 20 ngày cần ai phê duyệt? | 0.4041 | faithfulness |
| 4 | 43 | adversarial | Mật khẩu phải có tối thiểu bao nhiêu ký tự? | 0.4167 | faithfulness |
| 5 | 22 | multi_hop | Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT? | 0.4167 | faithfulness |
| 6 | 42 | adversarial | Thâm niên bao nhiêu năm thì được cộng thêm ngày phép? | 0.4167 | faithfulness |
| 7 | 48 | adversarial | Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không? | 0.4167 | faithfulness |
| 8 | 50 | adversarial | Nhân viên Manager có thể dùng VPN cá nhân (như NordVPN) khi WFH để tăng bảo mật thêm không? | 0.4167 | faithfulness |
| 9 | 44 | adversarial | Bao lâu phải đổi mật khẩu một lần? | 0.4583 | faithfulness |
| 10 | 9 | factual | Nam nhân viên được nghỉ bao nhiêu ngày khi vợ sinh con? | 0.5000 | faithfulness |

---

## 3. Failure Cluster Matrix

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---:|---:|---:|---:|
| faithfulness | 3 | 13 | 5 | 21 |
| answer_relevancy | 12 | 3 | 1 | 16 |
| context_precision | 3 | 0 | 0 | 3 |
| context_recall | 2 | 4 | 4 | 10 |

---

## 4. Dominant Failure Analysis

**Dominant distribution:** adversarial
**Dominant metric:** faithfulness

Adversarial có avg_score 0.6559 thấp nhất, tiếp theo là multi_hop 0.6994 và factual 0.8695. Trong bottom 10, có 5 câu thuộc adversarial, gồm các câu về mật khẩu, thâm niên, bảo hiểm thử việc và VPN cá nhân. Faithfulness là worst metric thường xuyên nhất với 21/50 câu; riêng multi_hop có 13 câu worst ở metric này. Answer relevancy của adversarial là 0.4404 và context recall là 0.7167, cho thấy nhóm câu hỏi có version conflict, phủ định hoặc ngoại lệ chính sách khó hơn retrieval factual đơn giản; đây là kết quả gợi ý phù hợp với đặc điểm test set, không phải kết luận về từng lỗi riêng lẻ.

---

## 5. Suggested Fixes

| Metric yếu | Root cause | Suggested fix |
|---|---|---|
| faithfulness | LLM hallucinating | Tighten system prompt, lower temperature |
| context_recall | Missing relevant chunks | Improve chunking or add BM25 |
| context_precision | Too many irrelevant chunks | Add reranking or metadata filter |
| answer_relevancy | Answer doesn't match question | Improve prompt template |

---

## 6. Nhận xét về Adversarial Distribution

Adversarial có avg_score 0.6559, thấp hơn factual 0.8695 khoảng 0.2136 và thấp hơn multi_hop 0.6994 khoảng 0.0435. Năm câu adversarial trong bottom 10 là các ID 43, 42, 48, 50 và 44, tương ứng với các chủ đề password minimum length, tenure/extra leave, probation health insurance, personal VPN khi WFH và password rotation interval. Pattern này cho thấy pipeline nhạy với các policy fact có nhiều version, exception hoặc contradiction trap; để kết luận một lỗi cụ thể là version conflict vẫn cần inspect contexts và answers của từng ID. Không có đủ bằng chứng từ aggregate này để khẳng định pipeline sai version cho toàn bộ adversarial distribution.

### Evaluation note

Trong lần chạy RAGAS có 1 JSONDecodeError ở 1/200 metric jobs. Pipeline vẫn hoàn tất 50/50 questions và report validation pass. Các score được giữ nguyên theo output runtime; không rerun hoặc chỉnh tay metric.
