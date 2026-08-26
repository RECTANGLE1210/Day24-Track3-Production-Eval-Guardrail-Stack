# LLM Judge Bias Report — Phase B

**Sinh viên:** Nguyễn Công Hùng - 2A202601071<br>
**Ngày:** 26/08/2026  
**Judge model:** openai/gpt-4o-mini

---

## 1. Pairwise Judge Results

| ID | Question | Winner | Reasoning |
|---:|---|---|---|
| 1 | Nhân viên được nghỉ bao nhiêu ngày khi kết hôn? | tie | Both answers provide the same information regarding the number of days off for employees getting married. |
| 5 | Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt? | B | Answer B provides a specific threshold for approval, indicating that purchases over 50 million VNĐ require the CEO's app |
| 12 | Thưởng Tết tối thiểu cho nhân viên chính thức có từ 6 tháng trở lên là bao nhiêu? | B | Answer B provides a complete and accurate response, including details about employees with less than 6 months of service |
| 29 | Nhân viên tạm ứng 8 triệu, chưa thanh toán sau 30 ngày. Ai phê duyệt và phí phạt là bao nhiêu? | B | Answer B provides a more detailed explanation of the approval process and calculates the penalty accurately based on the |
| 41 | Nhân viên được nghỉ bao nhiêu ngày phép năm? | B | Answer B provides the most current information regarding the number of vacation days, including details about the policy |
| 46 | Nhân viên thử việc có được nghỉ phép năm không? | tie | Both answers provide the same information regarding the lack of annual leave for probationary employees, with Answer B b |

---

## 2. Swap-and-Average Results

| ID | Pass 1 | Pass 2 | Final | Position Consistent? |
|---:|---|---|---|---|
| 1 | tie | tie | tie | True |
| 5 | B | B | B | True |
| 12 | B | B | B | True |
| 29 | B | B | B | True |
| 41 | B | B | B | True |
| 46 | tie | tie | tie | True |

**Position bias rate:** 0.000 (0 / 6)

---

## 3. Cohen's κ Analysis

| Question ID | Human Label | Judge Label | Agree? |
|---:|---:|---:|---|
| 1 | 1 | 1 | True |
| 5 | 0 | 0 | True |
| 12 | 1 | 1 | True |
| 21 | 1 | 1 | True |
| 23 | 1 | 1 | True |
| 29 | 0 | 0 | True |
| 33 | 1 | 1 | True |
| 41 | 0 | 0 | True |
| 46 | 1 | 1 | True |
| 50 | 0 | 0 | True |

**Cohen's κ:** 1.0000
**Interpretation:** almost perfect

---

## 4. Verbosity Bias

- A wins + A longer: 0 / 4 decisive cases
- B wins + B longer: 4 / 4 decisive cases
- **Verbosity bias rate:** 1.000

**Interpretation:** Position bias is low (0.000); verbosity correlation proxy is 1.000. Verbosity is a correlation signal, not evidence that length caused the winner.

---

## 5. Nhận xét chung

> Cohen's κ là 1.0000 (almost perfect), nên judge được xem là tương đối nhất quán theo thang đo đã dùng.
> Position bias rate là 0.000, không vượt ngưỡng 30%.
> Swap-and-average chưa cho thấy bất nhất trong các cặp đã đo.
> Trong production, LLM judge nên được dùng như một tín hiệu đánh giá kết hợp, không phải thẩm quyền duy nhất.
