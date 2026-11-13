# =============================================================================
# AI_AGENT_INSTRUCTIONS — ĐỌC TRƯỚC KHI THỰC HIỆN BẤT KỲ THAY ĐỔI NÀO
# =============================================================================
# MODULE     : CV06 — Quản trị Pipeline, Lý luận chuyên sâu (12 câu hỏi)
#                     & Báo cáo Kỹ thuật Tổng hợp
# PHỤ TRÁCH  : Hà Duy Hưng (Lead)
# MÔ TẢ TASK : Tổng kết toàn bộ kết quả thực nghiệm từ CV01 đến CV05.
#              Giải đáp chi tiết và có bằng chứng thực nghiệm cho toàn bộ
#              12 câu hỏi lý luận chuyên sâu trong đề bài DE_BAI_LAB02.md.
#              Xuất bản Báo cáo Kỹ thuật nghiệm thu hoàn chỉnh bằng Markdown.
#              Thiết kế Dashboard trực quan hóa tổng kết toàn diện bài Lab.
#
# DELIVERABLES (ĐẦU RA):
#   • docs/BAO_CAO_LAB02_NHOM7.md              – Báo cáo kỹ thuật hoàn chỉnh
#   • outputs/tables/final_summary_report.csv  – Bảng tổng hợp toàn diện
#   • outputs/figures/cv06_summary_dashboard.png
#
# RÀNG BUỘC BẮT BUỘC (HARD CONSTRAINTS):
#   [1] KHÔNG diễn giải lại kết quả mà không có bằng chứng số liệu thực nghiệm.
#   [2] Mọi câu trả lời lý luận PHẢI đi kèm giá trị số (AIC, BIC, MAE, RMSE,
#       MAPE, p-value ADF, p-value Ljung-Box) đọc từ outputs/tables/*.csv.
#   [3] KHÔNG tự ý thay đổi kết quả của các thành viên khác.
#   [4] Dashboard phải hiển thị tất cả biểu đồ chính trong 1 figure.
#   [5] Báo cáo Markdown PHẢI có đầy đủ: Header, Phương pháp, Kết quả số,
#       Nhận xét & Kết luận, Bảng Benchmark cuối cùng.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------------------------
# 0. THIẾT LẬP ĐƯỜNG DẪN GỐC
# ---------------------------------------------------------------------------
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR  = os.path.join(ROOT, "docs")
FIG_DIR   = os.path.join(ROOT, "outputs", "figures")
TABLE_DIR = os.path.join(ROOT, "outputs", "tables")

for d in [DOCS_DIR, FIG_DIR, TABLE_DIR]:
    os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. NẠP TOÀN BỘ KẾT QUẢ THỰC NGHIỆM TỪ CV01 → CV05
# ---------------------------------------------------------------------------
def load_all_results() -> dict:
    """
    Đọc toàn bộ bảng kết quả từ outputs/tables/:
      - cv01_data_summary.csv      (thống kê mô tả)
      - cv01_split_info.csv        (thông tin phân chia)
      - cv02_adf_test_summary.csv  (kết quả ADF)
      - cv03_candidate_orders.csv  (danh sách tham số ứng viên)
      - cv04_arima_metrics.csv     (AIC, BIC, MAE, RMSE, MAPE)
      - cv05_sarima_metrics.csv    (AIC, BIC, MAE, RMSE, MAPE)
      - cv05_model_benchmark_table.csv (Benchmark cuối cùng)

    Returns:
        dict: {'data_summary', 'split_info', 'adf', 'candidate_orders',
               'arima_metrics', 'sarima_metrics', 'benchmark'}
    """
    # TODO: Đọc từng file bằng pd.read_csv() — xử lý FileNotFoundError gracefully
    raise NotImplementedError("Lead Hưng cài đặt tại đây")


# ---------------------------------------------------------------------------
# 2. TỔNG HỢP BẢNG KẾT QUẢ CUỐI CÙNG
# ---------------------------------------------------------------------------
def build_final_summary(results: dict) -> pd.DataFrame:
    """
    Tạo bảng tổng hợp toàn diện gồm:
      - Thông tin dataset: 204 tháng, freq='MS', Train 163 / Test 41 tháng.
      - Kết quả ADF: Chuỗi gốc (p-value, is_stationary), Chuỗi sai phân (p-value, is_stationary).
      - Bộ tham số đã chọn: (p, d, q) và (P, D, Q, s).
      - Chỉ số đánh giá: AIC, BIC, MAE, RMSE, MAPE cho cả ARIMA và SARIMA.
      - Kết luận mô hình tốt hơn.
    Lưu: outputs/tables/final_summary_report.csv

    Returns:
        pd.DataFrame
    """
    # TODO: Ghép các bảng con thành 1 bảng tổng hợp
    raise NotImplementedError("Lead Hưng cài đặt tại đây")


# ---------------------------------------------------------------------------
# 3. VẼ DASHBOARD TỔNG KẾT (MULTI-PANEL FIGURE)
# ---------------------------------------------------------------------------
def plot_summary_dashboard(results: dict) -> None:
    """
    Vẽ Dashboard 2×3 panel gồm:
      [1,1] Chuỗi thời gian gốc (Trend & Seasonality)
      [1,2] ADF: Chuỗi gốc vs chuỗi sau sai phân
      [1,3] ACF/PACF (tóm tắt 12 lags đầu)
      [2,1] Dự báo ARIMA vs Thực tế (tập Test)
      [2,2] Dự báo SARIMA vs Thực tế (tập Test)
      [2,3] Bảng Benchmark: ARIMA vs SARIMA (dạng bảng màu)

    Lưu: outputs/figures/cv06_summary_dashboard.png (dpi=150)
    """
    # TODO: fig = plt.figure(figsize=(20, 12))
    #       gs  = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)
    raise NotImplementedError("Lead Hưng cài đặt tại đây")


# ---------------------------------------------------------------------------
# 4. GIẢI ĐÁP 12 CÂU HỎI LÝ LUẬN CHUYÊN SÂU
# ---------------------------------------------------------------------------
def generate_analytical_answers(results: dict) -> str:
    """
    Tổng hợp lý luận khoa học và bằng chứng thực nghiệm cho 12 câu hỏi
    trong đề bài DE_BAI_LAB02.md. Trả về chuỗi Markdown.

    Danh sách 12 câu hỏi (đọc từ docs/DE_BAI_LAB02.md để lấy nội dung chi tiết):
      Bài 1 (EDA):   Câu 1, 2, 3 — Xu hướng, Mùa vụ, Tính dừng trực quan.
      Bài 2 (ADF):   Câu 1, 2, 3 — Bản chất H₀, ý nghĩa p-value, kết luận.
      Bài 3 (Diff):  Câu 1, 2    — Giá trị d, chuỗi dừng sau sai phân.
      Bài 4 (ACF):   Câu 1, 2, 3 — Điểm cắt cụt, đề xuất (p,d,q), gai mùa vụ.
      Bài 5 (Model): Câu 1–5 (ARIMA), Câu 6–9 (SARIMA so sánh).

    Mỗi câu trả lời PHẢI:
      • Nêu giả thuyết / định nghĩa toán học.
      • Dẫn chiếu số liệu thực nghiệm: 'ADF p-value = X.XXXX' / 'RMSE = X.XXX'.
      • Kết luận ngắn gọn, chính xác.
    """
    # TODO: Xây dựng dict questions và tổng hợp answers thành Markdown string
    raise NotImplementedError("Lead Hưng cài đặt tại đây")


# ---------------------------------------------------------------------------
# 5. XUẤT BÁO CÁO KỸ THUẬT MARKDOWN
# ---------------------------------------------------------------------------
def export_report(summary_df: pd.DataFrame, analytical_answers: str) -> None:
    """
    Xuất bản Báo cáo Kỹ thuật nghiệm thu hoàn chỉnh:
    docs/BAO_CAO_LAB02_NHOM7.md

    Cấu trúc báo cáo chuẩn:
      # Báo cáo Kỹ thuật — Lab 02 Nhóm 7
      ## 1. Thông tin nhóm & Dataset
      ## 2. Kết quả EDA (Phân tích khám phá dữ liệu)
      ## 3. Kết quả ADF & Sai phân
      ## 4. Phân tích ACF / PACF & Lựa chọn tham số
      ## 5. Kết quả mô hình ARIMA(p,d,q)
      ## 6. Kết quả mô hình SARIMA(p,d,q)(P,D,Q)[12]
      ## 7. Bảng Benchmark đối chuẩn ARIMA vs SARIMA
      ## 8. Giải đáp 12 câu hỏi lý luận chuyên sâu
      ## 9. Kết luận tổng thể & Hướng phát triển
    """
    # TODO: Tổng hợp chuỗi Markdown và ghi ra DOCS_DIR/BAO_CAO_LAB02_NHOM7.md
    raise NotImplementedError("Lead Hưng cài đặt tại đây")


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("CV06 — Tổng hợp Báo cáo Nghiệm thu (Hà Duy Hưng - Lead)")
    print("=" * 70)

    # Bước 1: Nạp toàn bộ kết quả
    results = load_all_results()
    print("[OK] Đã nạp toàn bộ bảng kết quả từ outputs/tables/")

    # Bước 2: Tổng hợp bảng cuối
    summary = build_final_summary(results)
    print(f"[OK] Bảng tổng hợp:\n{summary.to_string()}")

    # Bước 3: Dashboard
    plot_summary_dashboard(results)
    print("[OK] Đã lưu Dashboard tổng kết.")

    # Bước 4: Giải đáp 12 câu hỏi
    answers_md = generate_analytical_answers(results)
    print(f"[OK] Đã soạn thảo lý luận {answers_md.count('###')} câu hỏi.")

    # Bước 5: Xuất Báo cáo Markdown
    export_report(summary, answers_md)
    print("[OK] Đã xuất bản docs/BAO_CAO_LAB02_NHOM7.md")
    print("[DONE] CV06 hoàn tất. Pipeline Lab 02 đã hoàn chỉnh toàn bộ!")

