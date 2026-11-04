# ============================================================
# AI_AGENT_INSTRUCTIONS
# ============================================================
# THÀNH VIÊN  : CV06 – Phương
# DEADLINE    : 14/11
# NHIỆM VỤ   : Tổng hợp báo cáo + trả lời 6 câu hỏi lý luận
#
# INPUT CONTRACT:
#   - outputs/tables/*.csv  (tất cả kết quả từ CV01-CV05)
#   - outputs/figures/*.png (tất cả biểu đồ từ CV01-CV05)
#
# OUTPUT CONTRACT:
#   - docs/BAO_CAO_LAB4_NHOM7.md   (báo cáo Markdown hoàn chỉnh)
#   - outputs/tables/final_summary.csv  (bảng tổng hợp cuối cùng)
#
# LƯU Ý QUAN TRỌNG:
#   - Báo cáo phải trả lời ĐẦY ĐỦ 6 câu hỏi lý luận (không để TODO)
#   - Tham chiếu biểu đồ và bảng số liệu cụ thể để minh họa
#   - Nhận xét phải dựa trên kết quả RMSE/MAE/MAPE thực tế
#   - Cấu trúc báo cáo: I. Giới thiệu → II. EDA → III. VAR → IV. XGB → V. LSTM
#     → VI. So sánh → VII. Kết luận → Tài liệu tham khảo
# ============================================================

import sys
sys.stdout.reconfigure(encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
import datetime

# ----------------------------------------------------------
# Định nghĩa đường dẫn
# ----------------------------------------------------------
BASE_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

TABLES_DIR  = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
DOCS_DIR    = PROJECT_ROOT / "docs"

DOCS_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = DOCS_DIR / "BAO_CAO_LAB4_NHOM7.md"


# ----------------------------------------------------------
# Load tất cả kết quả
# ----------------------------------------------------------
def load_all_results() -> dict:
    """
    Đọc tất cả file CSV trong outputs/tables/ và trả về dict.

    Returns
    -------
    results : dict  {tên_file: pd.DataFrame}
    """
    print("=" * 60)
    print("[load_all_results] Đọc kết quả từ outputs/tables/")
    print("=" * 60)

    results = {}
    csv_files = list(TABLES_DIR.glob("*.csv"))

    if not csv_files:
        print("  [CẢNH BÁO] Không tìm thấy file CSV nào trong outputs/tables/")
        print("  Hãy chạy CV01→CV05 trước khi tổng hợp báo cáo.")
        return results

    for f in sorted(csv_files):
        try:
            df = pd.read_csv(f)
            results[f.stem] = df
            print(f"  ✓ Loaded: {f.name}  ({df.shape})")
        except Exception as e:
            print(f"  ✗ Lỗi đọc {f.name}: {e}")

    return results


# ----------------------------------------------------------
# Tạo bảng so sánh cuối cùng
# ----------------------------------------------------------
def generate_summary_table(results: dict) -> pd.DataFrame:
    """
    Tổng hợp RMSE, MAE, MAPE của 3 mô hình thành 1 bảng so sánh duy nhất.

    Đọc từ: var_metrics.csv, xgb_metrics.csv, lstm_metrics.csv

    Returns
    -------
    summary_df : pd.DataFrame
    """
    print("=" * 60)
    print("[generate_summary_table] Tổng hợp bảng so sánh")
    print("=" * 60)

    rows = []
    model_map = {
        "var_metrics"  : "VAR",
        "xgb_metrics"  : "XGBoost",
        "lstm_metrics" : "LSTM",
    }

    for key, model_name in model_map.items():
        if key in results:
            df = results[key]
            row = {"Model": model_name}
            for col in ["RMSE", "MAE", "MAPE"]:
                if col in df.columns:
                    row[col] = df[col].iloc[0]
            rows.append(row)
        else:
            print(f"  [WARNING] Không tìm thấy {key}.csv")

    # Bổ sung từ comparison_table nếu có train/infer time
    if "comparison_table" in results:
        comp = results["comparison_table"].set_index("Model")
        for row in rows:
            model = row["Model"]
            if model in comp.index:
                for c in ["InferTime(s)", "TrainTime(s)"]:
                    if c in comp.columns:
                        row[c] = comp.loc[model, c]

    summary_df = pd.DataFrame(rows).set_index("Model")
    summary_df.to_csv(TABLES_DIR / "final_summary.csv")
    print(summary_df.to_string())
    print(f"\n  Đã lưu: {TABLES_DIR / 'final_summary.csv'}")
    return summary_df


# ----------------------------------------------------------
# 6 Câu hỏi lý luận
# ----------------------------------------------------------
def answer_q1_weather_importance(results: dict) -> str:
    """
    Q1: Biến thời tiết nào ảnh hưởng mạnh nhất đến phụ tải điện?

    TODO (CV06):
    Dựa vào:
      - Correlation heatmap từ CV01 (correlation_heatmap.png)
      - XGBoost feature importance từ CV03 (xgb_feature_importance.csv)

    Trả lời gợi ý:
      - Nhìn cột 'load' trong heatmap: temp thường có correlation cao nhất
      - Feature importance: temp_lag1, temp_lag2 thường top trong XGBoost
      - Giải thích: vào mùa hè, nhiệt độ tăng → điều hòa hoạt động → load tăng
    """
    answer = """
## Câu 1: Biến thời tiết nào ảnh hưởng mạnh nhất đến phụ tải điện?

**TODO (CV06) – Hướng dẫn trả lời:**

1. **Correlation heatmap** (Figure: correlation_heatmap.png):
   - Quan sát hệ số tương quan Pearson giữa `load` và các biến thời tiết
   - Biến có |r| cao nhất thường là biến ảnh hưởng mạnh nhất (tuyến tính)

2. **XGBoost Feature Importance** (Table: xgb_feature_importance.csv):
   - Liệt kê top-5 feature theo Gain → phần lớn nào là thời tiết?
   - Ví dụ: nếu `temp_lag1` đứng top 1 → nhiệt độ ảnh hưởng mạnh nhất

3. **Giải thích kinh tế kỹ thuật**:
   - Nhiệt độ cao → điều hòa không khí hoạt động nhiều → load tăng (hè)
   - Nhiệt độ thấp → sưởi điện hoạt động → load tăng (đông)
   - Độ ẩm, gió thường có ảnh hưởng gián tiếp (tăng cảm nhận nhiệt)

**Điền kết quả cụ thể từ output vào đây...**
    """
    return answer.strip()


def answer_q2_var_failure(results: dict) -> str:
    """
    Q2: Khi nào VAR thất bại (dự báo kém)?

    TODO (CV06):
    Dựa vào error_heatmap_by_hour.png và overlay plot.
    """
    answer = """
## Câu 2: Khi nào mô hình VAR thất bại?

**TODO (CV06) – Hướng dẫn trả lời:**

1. **Hạn chế lý thuyết của VAR**:
   - VAR giả định mối quan hệ TUYẾN TÍNH giữa các biến
   - VAR giả định chuỗi DỪNG → phải lấy sai phân → mất thông tin mức tuyệt đối
   - VAR không mô hình hoá phi tuyến, không nhớ pattern dài hạn

2. **Khi nào VAR thất bại trong thực tế** (quan sát từ heatmap + overlay):
   - Giờ cao điểm (7-9h sáng, 19-21h tối): load biến động phi tuyến → VAR sai nhiều
   - Ngày bất thường (lễ, Tết): VAR không có cơ chế encode event → dự báo sai
   - Sau sự kiện đột biến (power outage, heat wave): VAR phục hồi chậm

3. **Chứng minh bằng số liệu**:
   - So sánh MAE của VAR vs XGBoost/LSTM tại các giờ cao điểm từ heatmap
   - **Điền số liệu cụ thể từ output...**
    """
    return answer.strip()


def answer_q3_xgboost_learns(results: dict) -> str:
    """
    Q3: XGBoost học gì từ các biến thời tiết?

    TODO (CV06):
    Dựa vào feature importance của XGBoost.
    """
    answer = """
## Câu 3: XGBoost học gì từ các biến thời tiết?

**TODO (CV06) – Hướng dẫn trả lời:**

1. **Cơ chế học của XGBoost**:
   - XGBoost là Gradient Boosted Decision Trees → phân vùng không gian feature bằng splits
   - Không có khái niệm "sequence" → học qua lag features và rolling features

2. **Biến thời tiết nào được dùng nhiều nhất** (từ feature importance Gain):
   - `temp_lag1`, `temp_lag2`: nhiệt độ 1-2h trước → proxy tốt cho load hiện tại
   - `rolling_mean_6h`: xu hướng load trong 6h qua có tính đến thời tiết gián tiếp
   - Ghi số liệu top features từ xgb_feature_importance.csv

3. **Phi tuyến vs VAR**:
   - XGBoost có thể học: "khi temp > 32°C VÀ humidity > 80% → load tăng mạnh"
   - Đây là tương tác phi tuyến mà VAR không nắm được

**Điền kết quả cụ thể từ output...**
    """
    return answer.strip()


def answer_q4_lstm_cycles(results: dict) -> str:
    """
    Q4: LSTM học được chu kỳ nào trong dữ liệu?

    TODO (CV06):
    Dựa vào kết quả LSTM và đặc tính dữ liệu.
    """
    answer = """
## Câu 4: LSTM học được chu kỳ nào trong dữ liệu?

**TODO (CV06) – Hướng dẫn trả lời:**

1. **Chu kỳ ngắn (24h) – LSTM nắm được với lookback=24**:
   - Pattern ngày: load thấp 2-5h sáng, tăng mạnh 7-9h, đỉnh 10-12h và 18-21h
   - LSTM nhớ sequence 24h qua → học được pattern ngày đêm

2. **Chu kỳ dài hơn – LSTM CÓ HẠN với lookback=24**:
   - Chu kỳ tuần (168h): cuối tuần vs ngày thường
   - lookback=24 không đủ để capture chu kỳ tuần → cần lookback ≥ 168 hoặc thêm time features
   - Gợi ý: thêm `hour_of_day`, `day_of_week` vào input

3. **Bằng chứng từ kết quả**:
   - Nếu MAPE cao vào cuối tuần → LSTM chưa học được chu kỳ tuần
   - Xem error_heatmap_by_hour.png để xác nhận

**Điền phân tích cụ thể từ kết quả thực nghiệm...**
    """
    return answer.strip()


def answer_q5_lstm_overfitting(results: dict) -> str:
    """
    Q5: LSTM có bị overfitting không? Làm sao biết?

    TODO (CV06):
    Dựa vào learning curve từ CV04.
    """
    answer = """
## Câu 5: LSTM có bị overfitting không? Nhận biết như thế nào?

**TODO (CV06) – Hướng dẫn trả lời:**

1. **Cách nhận biết overfitting trong LSTM**:
   - **Learning Curve**: train_loss giảm liên tục nhưng val_loss bắt đầu tăng sau N epoch
   - **Gap lớn**: |val_loss - train_loss| >> 0 tại epoch cuối
   - Quan sát Figure: lstm_learning_curve.png

2. **Kết quả thực tế** (điền vào):
   - Best epoch: [từ kết quả CV04]
   - Train MSE cuối: [từ learning curve]
   - Val MSE tốt nhất: [từ learning curve]
   - Kết luận: có / không bị overfitting?

3. **Biện pháp xử lý nếu overfitting**:
   - Tăng Dropout (0.2 → 0.3-0.4)
   - Giảm số units LSTM (64 → 32, 32 → 16)
   - Thêm L2 regularization vào layer LSTM
   - Thu thập thêm dữ liệu
   - Data augmentation (ít hiệu quả với time-series)

**Điền nhận xét cụ thể từ lstm_learning_curve.png...**
    """
    return answer.strip()


def answer_q6_ml_vs_dl(results: dict) -> str:
    """
    Q6: Sự khác biệt cốt lõi giữa ML (XGBoost) và DL (LSTM)?

    TODO (CV06):
    Phân tích lý thuyết + so sánh kết quả thực nghiệm.
    """
    answer = """
## Câu 6: Sự khác biệt cốt lõi giữa ML (XGBoost) và DL (LSTM)?

**TODO (CV06) – Hướng dẫn trả lời:**

| Tiêu chí              | XGBoost (ML)                        | LSTM (DL)                           |
|-----------------------|-------------------------------------|-------------------------------------|
| Biểu diễn dữ liệu     | Flat vector (manual feature eng.)   | Tensor 3D (sequence tự động)        |
| Học temporal patterns | Thông qua lag/rolling features      | Thông qua hidden state (memory)     |
| Khả năng phi tuyến    | Cao (decision tree splits)          | Rất cao (neural non-linearity)      |
| Giải thích được       | Có (feature importance, SHAP)       | Khó (black box)                     |
| Dữ liệu cần           | Ít (vài nghìn mẫu đủ)              | Nhiều (cần vài chục nghìn mẫu tốt) |
| Tốc độ train          | Nhanh (~phút)                       | Chậm hơn (~giờ với GPU)             |
| Tốc độ inference      | Rất nhanh (ms)                      | Nhanh (ms) nhưng cần sequence       |
| Hyperparameter tuning | n_estimators, max_depth, lr, ...    | units, layers, lr, dropout, ...     |
| Overfitting risk      | Trung bình (early stopping helps)   | Cao (cần Dropout + EarlyStopping)   |

**Kết luận từ thực nghiệm** (điền vào):
- RMSE: VAR={?}, XGBoost={?}, LSTM={?}
- Mô hình tốt nhất cho bài toán này: [kết luận và giải thích]
- Khi nào nên dùng ML thay DL: ít dữ liệu, cần interpretability, resource hạn chế

**Điền số liệu cụ thể từ comparison_table.csv...**
    """
    return answer.strip()


# ----------------------------------------------------------
# Viết báo cáo Markdown
# ----------------------------------------------------------
def write_markdown_report(all_sections: dict, summary_df: pd.DataFrame) -> None:
    """
    Tổng hợp tất cả sections và ghi ra file BAO_CAO_LAB4_NHOM7.md.

    Parameters
    ----------
    all_sections : dict  {tên_section: nội_dung_str}
    summary_df   : pd.DataFrame  (bảng so sánh cuối)
    """
    print("=" * 60)
    print("[write_markdown_report] Viết báo cáo Markdown")
    print("=" * 60)

    now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")

    # Chuyển summary_df thành Markdown table
    summary_md = summary_df.reset_index().to_markdown(index=False) if not summary_df.empty else \
                 "_Chưa có dữ liệu – hãy chạy CV01→CV05 trước_"

    lines = [
        f"# Báo cáo Lab 4: Dự báo Chuỗi Thời Gian Đa Biến",
        f"",
        f"> **Nhóm 7** | Môn: Chuỗi Thời Gian | Ngày tổng hợp: {now}",
        f"",
        f"## Thành viên nhóm",
        f"| Mã   | Họ tên  | Nhiệm vụ                    | Deadline |",
        f"|------|---------|-----------------------------|----------|",
        f"| CV01 | Hưng    | EDA & Preprocessing (Lead)  | 05/11    |",
        f"| CV02 | Khải    | Mô hình VAR                 | 08/11    |",
        f"| CV03 | Hoàng   | Mô hình XGBoost             | 08/11    |",
        f"| CV04 | Trâm    | Mô hình LSTM                | 12/11    |",
        f"| CV05 | Hiệp    | Backtesting & So sánh       | 12/11    |",
        f"| CV06 | Phương  | Tổng hợp báo cáo            | 14/11    |",
        f"",
        f"---",
        f"",
        f"## I. Giới thiệu bài toán",
        f"",
        f"**Mục tiêu**: Dự báo phụ tải điện (load) 1 bước trước (one-step-ahead) "
        f"sử dụng 3 biến thời tiết hỗ trợ: nhiệt độ (temp), độ ẩm (humidity), "
        f"tốc độ gió (wind_speed).",
        f"",
        f"**Dữ liệu**: Chuỗi thời gian đa biến, tần suất giờ (hourly), "
        f"split 70/15/15 theo thứ tự thời gian.",
        f"",
        f"**3 mô hình so sánh**:",
        f"- **VAR** – Statistical baseline (tuyến tính, đa biến)",
        f"- **XGBoost** – Machine Learning (Gradient Boosted Trees)",
        f"- **LSTM** – Deep Learning (Recurrent Neural Network)",
        f"",
        f"---",
        f"",
        f"## II. EDA & Tiền xử lý (CV01)",
        f"",
        f"<!-- TODO (CV06): Tóm tắt kết quả EDA từ CV01 -->",
        f"- Mô tả dữ liệu: số quan sát, khoảng thời gian, tần suất",
        f"- Missing values: bao nhiêu %, cách xử lý",
        f"- Outlier: phương pháp IQR, số outlier phát hiện",
        f"- Nhận xét từ correlation heatmap",
        f"",
        f"![Correlation Heatmap](../outputs/figures/correlation_heatmap.png)",
        f"",
        f"---",
        f"",
        f"## III. Mô hình VAR (CV02)",
        f"",
        f"<!-- TODO (CV06): Tóm tắt kết quả VAR từ CV02 -->",
        f"- Kết quả kiểm định ADF cho từng biến",
        f"- Bậc tối ưu p* theo AIC/BIC",
        f"- RMSE/MAE/MAPE trên tập test",
        f"",
        f"![VAR AIC/BIC](../outputs/figures/var_aic_bic.png)",
        f"",
        f"---",
        f"",
        f"## IV. Mô hình XGBoost (CV03)",
        f"",
        f"<!-- TODO (CV06): Tóm tắt kết quả XGBoost từ CV03 -->",
        f"- Số features sử dụng (lag + rolling + trend)",
        f"- Best iteration (early stopping)",
        f"- RMSE/MAE/MAPE trên tập test",
        f"",
        f"![XGBoost Feature Importance](../outputs/figures/xgb_feature_importance.png)",
        f"",
        f"---",
        f"",
        f"## V. Mô hình LSTM (CV04)",
        f"",
        f"<!-- TODO (CV06): Tóm tắt kết quả LSTM từ CV04 -->",
        f"- Kiến trúc: LSTM(64) → LSTM(32) → Dense(1)",
        f"- Best epoch, val_loss tốt nhất",
        f"- RMSE/MAE/MAPE trên tập test",
        f"",
        f"![LSTM Learning Curve](../outputs/figures/lstm_learning_curve.png)",
        f"",
        f"---",
        f"",
        f"## VI. So sánh tổng hợp (CV05)",
        f"",
        f"### Bảng so sánh kết quả",
        f"",
        summary_md,
        f"",
        f"![Comparison Overlay](../outputs/figures/comparison_overlay.png)",
        f"",
        f"![Error Heatmap by Hour](../outputs/figures/error_heatmap_by_hour.png)",
        f"",
        f"---",
        f"",
        f"## VII. Trả lời câu hỏi lý luận",
        f"",
        all_sections.get("q1", "_Chưa trả lời_"),
        f"",
        f"---",
        f"",
        all_sections.get("q2", "_Chưa trả lời_"),
        f"",
        f"---",
        f"",
        all_sections.get("q3", "_Chưa trả lời_"),
        f"",
        f"---",
        f"",
        all_sections.get("q4", "_Chưa trả lời_"),
        f"",
        f"---",
        f"",
        all_sections.get("q5", "_Chưa trả lời_"),
        f"",
        f"---",
        f"",
        all_sections.get("q6", "_Chưa trả lời_"),
        f"",
        f"---",
        f"",
        f"## VIII. Kết luận",
        f"",
        f"<!-- TODO (CV06): Viết kết luận tổng thể -->",
        f"- Mô hình nào tốt nhất cho bài toán dự báo phụ tải điện?",
        f"- Đánh đổi giữa độ chính xác, tốc độ và khả năng giải thích",
        f"- Hướng phát triển: Transformer, TFT, N-BEATS, ...",
        f"",
        f"## Tài liệu tham khảo",
        f"",
        f"1. Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.",
        f"2. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. "
        f"*KDD 2016*.",
        f"3. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. "
        f"*Neural Computation*, 9(8), 1735–1780.",
        f"4. Hyndman, R.J., & Athanasopoulos, G. (2021). "
        f"*Forecasting: Principles and Practice* (3rd ed). OTexts.",
        f"",
    ]

    report_content = "\n".join(lines)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"  Đã lưu báo cáo: {REPORT_PATH}")
    print(f"  Dung lượng    : {REPORT_PATH.stat().st_size / 1024:.1f} KB")


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
def main():
    """
    Pipeline chính cho CV06: Tổng hợp báo cáo.
    """
    print("\n" + "=" * 60)
    print("  CV06 – REPORT SYNTHESIS PIPELINE")
    print("=" * 60 + "\n")

    # --- Load kết quả ---
    results = load_all_results()

    # --- Tạo bảng tổng hợp ---
    summary_df = generate_summary_table(results)

    # --- Trả lời 6 câu hỏi lý luận ---
    print("=" * 60)
    print("[QUESTIONS] Tổng hợp 6 câu hỏi lý luận")
    print("=" * 60)

    all_sections = {
        "q1": answer_q1_weather_importance(results),
        "q2": answer_q2_var_failure(results),
        "q3": answer_q3_xgboost_learns(results),
        "q4": answer_q4_lstm_cycles(results),
        "q5": answer_q5_lstm_overfitting(results),
        "q6": answer_q6_ml_vs_dl(results),
    }

    for key, ans in all_sections.items():
        print(f"\n{'─' * 50}")
        print(f"  [{key.upper()}]")
        # In tối đa 5 dòng đầu để preview
        preview_lines = ans.strip().split("\n")[:5]
        for line in preview_lines:
            print(f"  {line}")
        if len(ans.strip().split("\n")) > 5:
            print("  ...")

    # --- Viết báo cáo Markdown ---
    write_markdown_report(all_sections, summary_df)

    print("\n" + "=" * 60)
    print(f"  [CV06 DONE] Báo cáo đã được lưu tại:")
    print(f"  {REPORT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
