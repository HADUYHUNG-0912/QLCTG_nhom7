# -*- coding: utf-8 -*-
"""
DỰ ÁN: QUẢN LÝ & KHAI PHÁ CHUỖI THỜI GIAN (QLCTG) - LAB 2
BÀI TẬP 1: TÍNH VÀ PHÂN TÍCH BIỂU ĐỒ ACF & PACF
---------------------------------------------------------------------------------
Thành viên phụ trách: Thành viên 1
Thư mục: LAB2/bai_tap_1/
Tệp thực thi: bai_tap_1.py
Dataset đầu vào: LAB2/data/raw/AirPassengers.csv

=================================================================================
🤖 CHỈ THỊ DÀNH CHO AI AGENT (AI AGENT PROMPT & GUARDRAILS SPECIFICATION)
=================================================================================
Nếu bạn là một AI Agent (Cursor, GitHub Copilot, Gemini, ChatGPT, Claude...):
1. TUÂN THỦ KIẾN TRÚC ĐỒNG NHẤT:
   - Không được đổi tên 4 hàm chuẩn: `step1_load_data`, `step2_execute_analysis`,
     `step3_visualize`, `step4_conclude_and_report`.
   - Tất cả các hàm phải có type hint (gợi ý kiểu dữ liệu) rõ ràng.
2. NGUYÊN TẮC XỬ LÝ DỮ LIỆU:
   - Dữ liệu phải được thiết lập DatetimeIndex với tần suất tháng ('MS').
   - Tuyệt đối không thay đổi chuỗi gốc khi tính toán.
3. NGUYÊN TẮC TRỰC QUAN HÓA & LƯU TRỮ:
   - KHÔNG tự ý lưu tệp hình ảnh (.png/.jpg) ra đĩa để tránh làm nặng repository.
   - Hàm `step3_visualize` tạo biểu đồ, có thể gọi `plt.show()` nếu cần xem trực quan hoặc giải phóng bộ nhớ.
   - Console log và file text phải in rõ: Trị số ACF tại lag 1, lag 12 và kết luận về Trend / Seasonality.
4. CÂU HỎI NGHIỆM THU CẦN TRẢ LỜI:
   - Câu 1: Chuỗi có tự tương quan mạnh không? (ACF giảm chậm hay nhanh?)
   - Câu 2: Có xu hướng (Trend) và mùa vụ (Seasonality) thể hiện qua ACF/PACF không?
   - Câu 3: Những lag nào có ý nghĩa thống kê quan trọng nhất?
=================================================================================
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Cấu hình encoding UTF-8 để không bị lỗi console trên Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# -------------------------------------------------------------------------------
# 1. CẤU HÌNH ĐƯỜNG DẪN & THAM SỐ TOÀN CỤC
# -------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "AirPassengers.csv"

CONFIG = {
    "lags": 40,
    "alpha": 0.05,  # Mức ý nghĩa 5% (Dải tin cậy 95%)
    "target_col": "Passengers",
    "datetime_col": "Month"
}


# -------------------------------------------------------------------------------
# 2. BƯỚC 1: TẢI VÀ KIỂM ĐỊNH ĐỊNH DẠNG DỮ LIỆU
# -------------------------------------------------------------------------------
def step1_load_data(data_path: Path) -> pd.Series:
    """
    Tải tệp CSV, chuyển đổi cột thời gian thành DatetimeIndex và kiểm tra tính toàn vẹn.
    """
    print(f"[BƯỚC 1] Đang tải dữ liệu từ: {data_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu tại {data_path}. Hãy kiểm tra lại!")

    df = pd.read_csv(data_path)
    df[CONFIG["datetime_col"]] = pd.to_datetime(df[CONFIG["datetime_col"]])
    df.set_index(CONFIG["datetime_col"], inplace=True)
    df.sort_index(inplace=True)

    series = df[CONFIG["target_col"]].astype(float)
    expected_index = pd.date_range(series.index.min(), series.index.max(), freq="MS")
    if not series.index.equals(expected_index):
        raise ValueError("Dữ liệu phải có các tháng liên tiếp, không trùng và bắt đầu ngày 1.")
    series = series.asfreq("MS")
    if not np.isfinite(series.to_numpy()).all():
        raise ValueError("Dữ liệu chứa giá trị thiếu hoặc không hữu hạn.")
    print(f" -> Tần suất dữ liệu: {series.index.freqstr}")
    print(f" -> Đã tải {len(series)} quan sát ({series.index.min().strftime('%Y-%m')} đến {series.index.max().strftime('%Y-%m')}).")
    print(f" -> Kiểm tra giá trị khuyết thiếu (NaN): {series.isna().sum()} điểm.")
    return series


# -------------------------------------------------------------------------------
# 2. BƯỚC 2: TÍNH TOÁN CÁC HỆ SỐ TỰ TƯƠNG QUAN (ACF & PACF)
# -------------------------------------------------------------------------------
def step2_execute_analysis(series: pd.Series, nlags: int = 40) -> Dict[str, Any]:
    """
    Tính toán chi tiết các giá trị ACF và PACF lên tới 40 lags.
    """
    print(f"[BƯỚC 2] Tính toán hệ số ACF và PACF với số lag = {nlags}...")
    if not 12 <= nlags < len(series) // 2:
        raise ValueError("Số lag phải >= 12 và nhỏ hơn một nửa số quan sát.")
    acf_values, acf_confint = acf(series, nlags=nlags, alpha=CONFIG["alpha"])
    pacf_values, pacf_confint = pacf(series, nlags=nlags, alpha=CONFIG["alpha"], method="ywm")

    # Xác định các lag vượt ngưỡng tin cậy 95%
    n = len(series)
    threshold = 1.96 / np.sqrt(n)
    significant_acf_lags = [lag for lag in range(1, len(acf_values))
                            if acf_confint[lag, 0] > 0 or acf_confint[lag, 1] < 0]
    significant_pacf_lags = [lag for lag in range(1, len(pacf_values))
                             if pacf_confint[lag, 0] > 0 or pacf_confint[lag, 1] < 0]

    results = {
        "nlags": nlags,
        "acf_confint": acf_confint,
        "pacf_confint": pacf_confint,
        "significant_pacf_lags": significant_pacf_lags,
        "acf_values": acf_values,
        "pacf_values": pacf_values,
        "threshold_95": threshold,
        "significant_acf_lags": significant_acf_lags,
        "acf_lag_1": acf_values[1],
        "acf_lag_12": acf_values[12] if len(acf_values) > 12 else None
    }

    print(f" -> Hệ số tự tương quan tại Lag 1:  {results['acf_lag_1']:.4f}")
    if results["acf_lag_12"] is not None:
        print(f" -> Hệ số tự tương quan tại Lag 12: {results['acf_lag_12']:.4f}")
    print(f" -> Ngưỡng xấp xỉ cho PACF 95% (+/- 1.96 / sqrt(N)): +/- {threshold:.4f}")
    print(f" -> Số lượng lag vượt ngưỡng tin cậy: {len(significant_acf_lags)} / {nlags} lags.")
    print(f" -> Các lag ACF có ý nghĩa (CI Bartlett 95%): {significant_acf_lags}")
    print(f" -> Các lag PACF có ý nghĩa (CI 95%): {significant_pacf_lags}")
    return results


# -------------------------------------------------------------------------------
# 3. BƯỚC 3: TRỰC QUAN HÓA VÀ XUẤT ĐỒ THỊ
# -------------------------------------------------------------------------------
def step3_visualize(series: pd.Series, analysis_results: Dict[str, Any], show_plot: bool = False) -> None:
    """
    Khởi tạo 2 biểu đồ trực quan ACF và PACF cạnh nhau để kiểm tra hình dạng tương quan.
    Lưu ý: Không tự ý xuất file ảnh ra ổ đĩa theo yêu cầu của dự án.
    """
    print(f"[BƯỚC 3] Đang khởi tạo biểu đồ ACF và PACF...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Đồ thị ACF
    plot_acf(series, lags=analysis_results["nlags"], alpha=CONFIG["alpha"], ax=axes[0], title="Autocorrelation Function (ACF) - AirPassengers", color="royalblue")
    axes[0].set_xlabel("Độ trễ (Lags)")
    axes[0].set_ylabel("Hệ số ACF")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Đồ thị PACF
    plot_pacf(series, lags=analysis_results["nlags"], alpha=CONFIG["alpha"], ax=axes[1], title="Partial Autocorrelation Function (PACF) - AirPassengers", color="crimson", method="ywm")
    axes[1].set_xlabel("Độ trễ (Lags)")
    axes[1].set_ylabel("Hệ số PACF")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    print(" -> Đã hoàn thành trực quan hóa ACF/PACF (không xuất tệp ảnh ra đĩa).")
    return None


# -------------------------------------------------------------------------------
# 4. BƯỚC 4: TỔNG KẾT VÀ TRẢ LỜI CÂU HỎI BÀI TOÁN
# -------------------------------------------------------------------------------
def step4_conclude_and_report(analysis_results: Dict[str, Any]) -> str:
    """
    Tổng hợp nhận xét học thuật và trả lời 3 câu hỏi nghiệm thu của đề bài.
    """
    print("\n[BƯỚC 4] TỔNG HỢP KẾT QUẢ VÀ TRẢ LỜI CÂU HỎI NGHIỆM THU BÀI 1:")
    report_text = f"""
=================================================================================
BÁO CÁO PHÂN TÍCH KẾT QUẢ BÀI TẬP 1 (ACF & PACF)
=================================================================================
1. TÍNH CHẤT TỰ TƯƠNG QUAN:
   - Chuỗi AirPassengers có tự tương quan RẤT MẠNH.
   - Hệ số ACF tại Lag 1 đạt mức cao ({analysis_results['acf_lag_1']:.4f}) và suy giảm rất chậm qua các lag
     (tail-off / decaying slowly), phản ánh tính 'ghi nhớ' dài hạn của dữ liệu.

2. DẤU HIỆU XU HƯỚNG (TREND) VÀ MÙA VỤ (SEASONALITY):
   - Xu hướng (Trend): Thể hiện rõ nét qua việc các hệ số ACF dương lớn suy giảm từ từ qua thời gian.
     Đây là dấu hiệu chuỗi không dừng; có thể thử sai phân bậc 1 (d = 1),
     sau đó kiểm định tính dừng để xác nhận.
   - Mùa vụ (Seasonality): Biểu đồ ACF xuất hiện các đỉnh cục bộ lặp lại theo chu kỳ
     tại các bội số của 12, gợi ý chu kỳ mùa vụ 12 tháng.
   - ACF tại Lag 12/24/36: {analysis_results['acf_values'][12]:.4f} / {analysis_results['acf_values'][24] if analysis_results['nlags'] >= 24 else float('nan'):.4f} / {analysis_results['acf_values'][36] if analysis_results['nlags'] >= 36 else float('nan'):.4f}.

3. CÁC LAG CÓ Ý NGHĨA QUAN TRỌNG NHẤT:
   - Lag 1: Đo lường tương quan trễ ngắn hạn mạnh nhất.
   - Lag 12 ({analysis_results['acf_lag_12']:.4f}): Lag phản ánh đỉnh chu kỳ mùa vụ hàng năm.
   - Các lag ACF có ý nghĩa theo khoảng tin cậy Bartlett 95%: {analysis_results['significant_acf_lags']}.
   - Các lag PACF có ý nghĩa theo khoảng tin cậy 95%: {analysis_results['significant_pacf_lags']}.
   - PACF tại Lag 1/2/12/13: {analysis_results['pacf_values'][1]:.4f} / {analysis_results['pacf_values'][2]:.4f} / {analysis_results['pacf_values'][12]:.4f} / {analysis_results['pacf_values'][13] if analysis_results['nlags'] >= 13 else float('nan'):.4f}.
     PACF không cắt cụt rõ ngay sau Lag 1; không thể chốt bậc AR chỉ từ chuỗi gốc.
   - Dải ACF dùng công thức Bartlett, độ rộng tăng theo lag; dải PACF xấp xỉ
     +/- {analysis_results['threshold_95']:.4f}. Các kiểm tra theo từng lag chưa hiệu chỉnh đa kiểm định.

4. GỢI Ý MÔ HÌNH BAN ĐẦU:
   - Có thể thử ARIMA(1,1,1) và SARIMA(1,1,1)(1,1,1)[12].
   - Đây chỉ là cấu hình khởi đầu; cần xem lại ACF/PACF sau sai phân, so sánh
     AIC/BIC, đánh giá dự báo và kiểm định phần dư trước khi chọn mô hình.
=================================================================================
"""
    print(report_text)

    # Ghi báo cáo ra file txt
    report_file = BASE_DIR / "ket_qua_bai_1.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f" -> Đã lưu báo cáo phân tích tại: {report_file}")
    return report_text


# -------------------------------------------------------------------------------
# HÀM THỰC THI CHÍNH
# -------------------------------------------------------------------------------
def main() -> None:
    print("=" * 80)
    print("KHỞI CHẠY BÀI TẬP 1: TÍNH VÀ PHÂN TÍCH ACF/PACF")
    print("=" * 80)
    series = step1_load_data(DATA_PATH)
    analysis_results = step2_execute_analysis(series, nlags=CONFIG["lags"])
    step3_visualize(series, analysis_results)
    step4_conclude_and_report(analysis_results)
    print("\n[HOÀN THÀNH BÀI TẬP 1 THÀNH CÔNG]")


if __name__ == "__main__":
    main()
