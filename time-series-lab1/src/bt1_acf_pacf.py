# -*- coding: utf-8 -*-
"""
DỰ ÁN: QUẢN LÝ & KHAI PHÁ CHUỖI THỜI GIAN (QLCTG) - TIME SERIES LAB
BÀI TẬP 1: TÍNH VÀ PHÂN TÍCH BIỂU ĐỒ ACF & PACF
---------------------------------------------------------------------------------
Thành viên phụ trách: Thành viên 1 - Nhóm 7
Thư mục module: src/bt1_acf_pacf.py
Dataset đầu vào: data/raw/AirPassengers.csv
Outputs xuất ra: outputs/figures/bai1_acf_pacf.png, outputs/tables/

MỤC TIÊU BÀI TẬP:
- Tính toán hệ số tự tương quan (ACF) và tự tương quan riêng phần (PACF) đến 40 lags.
- Đánh giá tính chất tự tương quan, xu hướng (Trend) và tính mùa vụ (Seasonality).
- Xác định các độ trễ (lags) có ý nghĩa thống kê quan trọng nhất.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Tuple

# Cấu hình encoding UTF-8 để không bị lỗi console trên Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Cấu hình phông chữ hỗ trợ tiếng Việt trên Windows
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "Tahoma", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

# -------------------------------------------------------------------------------
# 1. CẤU HÌNH ĐƯỜNG DẪN & THAM SỐ TOÀN CỤC
# -------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "AirPassengers.csv"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

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
    """Tải tệp CSV, chuyển đổi cột thời gian thành DatetimeIndex và kiểm tra tính toàn vẹn."""
    print(f"[BƯỚC 1] Đang tải dữ liệu từ: {data_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu tại {data_path}. Hãy kiểm tra lại!")

    df = pd.read_csv(data_path)
    df[CONFIG["datetime_col"]] = pd.to_datetime(df[CONFIG["datetime_col"]])
    df.set_index(CONFIG["datetime_col"], inplace=True)
    df.sort_index(inplace=True)

    series = df[CONFIG["target_col"]]
    print(f" -> Tải thành công {len(series)} quan sát từ {series.index.min().strftime('%Y-%m')} đến {series.index.max().strftime('%Y-%m')}.")
    return series


# -------------------------------------------------------------------------------
# 3. BƯỚC 2: TÍNH TOÁN ACF VÀ PACF
# -------------------------------------------------------------------------------
def step2_execute_analysis(series: pd.Series, nlags: int = 40) -> Dict[str, Any]:
    """Tính toán mảng ACF, PACF cùng khoảng tin cậy 95% Bartlett."""
    print(f"\n[BƯỚC 2] Tính toán ACF và PACF cho {nlags} lags...")
    acf_values, acf_confint = acf(series, nlags=nlags, alpha=CONFIG["alpha"])
    pacf_values, pacf_confint = pacf(series, nlags=nlags, alpha=CONFIG["alpha"], method="yw")

    n = len(series)
    bartlett_ci = 1.96 / np.sqrt(n)

    results = {
        "acf_values": acf_values,
        "acf_confint": acf_confint,
        "pacf_values": pacf_values,
        "pacf_confint": pacf_confint,
        "bartlett_ci": bartlett_ci,
        "acf_lag_1": float(acf_values[1]),
        "acf_lag_12": float(acf_values[12]),
        "pacf_lag_1": float(pacf_values[1]),
        "pacf_lag_12": float(pacf_values[12]),
    }
    return results


# -------------------------------------------------------------------------------
# 4. BƯỚC 3: TRỰC QUAN HÓA ACF & PACF
# -------------------------------------------------------------------------------
def step3_visualize(series: pd.Series, results: Dict[str, Any], show_plot: bool = False):
    """Vẽ và lưu biểu đồ ACF/PACF vào outputs/figures/bai1_acf_pacf.png."""
    print("\n[BƯỚC 3] Đang kết xuất biểu đồ trực quan hóa ACF & PACF...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    nlags = CONFIG["lags"]

    plot_acf(series, lags=nlags, ax=axes[0], alpha=CONFIG["alpha"], title="Autocorrelation Function (ACF) - AirPassengers (40 Lags)")
    axes[0].set_xlabel("Độ trễ (Lags - Tháng)")
    axes[0].set_ylabel("Hệ số tự tương quan (ACF)")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    plot_pacf(series, lags=nlags, ax=axes[1], alpha=CONFIG["alpha"], method="yw",
              title="Partial Autocorrelation Function (PACF) - AirPassengers (40 Lags)")
    axes[1].set_xlabel("Độ trễ (Lags - Tháng)")
    axes[1].set_ylabel("Hệ số tự tương quan riêng phần (PACF)")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    output_fig = FIGURES_DIR / "bai1_acf_pacf.png"
    plt.savefig(output_fig, dpi=300)
    print(f" -> Đã lưu biểu đồ thành công tại: {output_fig}")

    if show_plot:
        plt.show()
    plt.close()


# -------------------------------------------------------------------------------
# 5. BƯỚC 4: TỔNG HỢP VÀ BÁO CÁO KẾT QUẢ
# -------------------------------------------------------------------------------
def step4_conclude_and_report(analysis_results: Dict[str, Any]) -> str:
    """Tổng hợp nhận xét học thuật và trả lời câu hỏi nghiệm thu."""
    print("\n[BƯỚC 4] Tổng hợp kết quả phân tích...")
    report_text = f"""=================================================================================
BÁO CÁO PHÂN TÍCH KẾT QUẢ BÀI TẬP 1 (ACF & PACF)
=================================================================================
1. TÍNH CHẤT TỰ TƯƠNG QUAN:
   - Chuỗi AirPassengers có tự tương quan RẤT MẠNH.
   - Hệ số ACF tại Lag 1 đạt mức cao ({analysis_results['acf_lag_1']:.4f}) và suy giảm rất chậm qua các lag
     (tail-off / decaying slowly), phản ánh tính 'ghi nhớ' dài hạn của dữ liệu.

2. DẤU HIỆU XU HƯỚNG (TREND) VÀ MÙA VỤ (SEASONALITY):
   - Xu hướng (Trend): Thể hiện rõ nét qua việc các hệ số ACF dương lớn suy giảm từ từ qua thời gian.
     Chuỗi này là chuỗi không dừng (Non-stationary), cần lấy sai phân bậc 1 (d = 1) để khử xu hướng.
   - Mùa vụ (Seasonality): Biểu đồ ACF xuất hiện các đỉnh nhô cao vượt dải tin cậy lặp lại theo chu kỳ 
     đều đặn tại các bội số của 12 (Lag 12: {analysis_results['acf_lag_12']:.4f}, Lag 24, Lag 36).
     Khẳng định chuỗi có chu kỳ mùa vụ 12 tháng rõ rệt.

3. CÁC LAG CÓ Ý NGHĨA QUAN TRỌNG NHẤT:
   - Lag 1: Đo lường tương quan trễ ngắn hạn mạnh nhất.
   - Lag 12: Phản ánh đỉnh chu kỳ mùa vụ hàng năm.
   - Biểu đồ PACF cắt cụt (cut-off) sau Lag 1 và có xung đáng chú ý tại Lag 12/13, gợi ý việc xây dựng
     mô hình SARIMA với thành phần tự hồi quy p = 1 và mùa vụ P = 1.
================================================================================="""
    print(report_text)
    return report_text


def main(show_plot: bool = False):
    print("=" * 80)
    print("KHỞI CHẠY BÀI TẬP 1: TÍNH VÀ PHÂN TÍCH ACF/PACF")
    print("=" * 80)
    series = step1_load_data(DATA_PATH)
    results = step2_execute_analysis(series, nlags=CONFIG["lags"])
    step3_visualize(series, results, show_plot=show_plot)
    step4_conclude_and_report(results)
    print("\n[HOÀN THÀNH BÀI TẬP 1 THÀNH CÔNG]")


if __name__ == "__main__":
    show = "--show" in sys.argv
    main(show_plot=show)
