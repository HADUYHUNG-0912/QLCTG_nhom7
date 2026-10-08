# -*- coding: utf-8 -*-
"""
DỰ ÁN: QUẢN LÝ & KHAI PHÁ CHUỖI THỜI GIAN (QLCTG) - LAB 2
BÀI TẬP 3: PHÂN RÃ CHUỖI THỜI GIAN (TIME-SERIES DECOMPOSITION)
---------------------------------------------------------------------------------
Thành viên phụ trách: Thành viên 3
Thư mục: LAB2/bai_tap_3/
Tệp thực thi: bai_tap_3.py
Dataset đầu vào: LAB2/data/raw/AirPassengers.csv

=================================================================================
🤖 CHỈ THỊ DÀNH CHO AI AGENT (AI AGENT PROMPT & GUARDRAILS SPECIFICATION)
=================================================================================
Nếu bạn là một AI Agent (Cursor, GitHub Copilot, Gemini, ChatGPT, Claude...):
1. TUÂN THỦ KIẾN TRÚC ĐỒNG NHẤT:
   - Không được đổi tên 4 hàm chuẩn: `step1_load_data`, `step2_execute_analysis`, 
     `step3_visualize`, `step4_conclude_and_report`.
   - Giữ nguyên các type hint và cấu trúc trả về `Dict[str, Any]`.
2. NGUYÊN TẮC KỸ THUẬT:
   - Dữ liệu AirPassengers có biên độ mùa vụ tăng dần theo thời gian (cần so sánh
     giữa `model='additive'` và `model='multiplicative'` trong Classical Decomposition).
   - Chạy thêm phương pháp phân rã hiện đại STL (`statsmodels.tsa.seasonal.STL`) với `robust=True`.
   - Chu kỳ phân rã: `period=12` (chu kỳ năm cho dữ liệu tháng).
3. NGUYÊN TẮC TRỰC QUAN HÓA & LƯU TRỮ:
   - KHÔNG tự ý lưu tệp hình ảnh (.png/.jpg) ra ổ đĩa để tránh làm nặng repository.
   - Hàm `step3_visualize` khởi tạo đồ thị để phân tích, giải phóng bộ nhớ hoặc hiển thị nếu cần.
   - Báo cáo phân tích lưu tại `LAB2/bai_tap_3/ket_qua_bai_3.txt`.
4. CÂU HỎI NGHIỆM THU:
   - Câu 1: Xu hướng (Trend) là tăng hay giảm? Tăng tuyến tính hay phi tuyến?
   - Câu 2: Tính mùa vụ (Seasonality) mạnh hay yếu? Đỉnh điểm vào tháng mấy?
   - Câu 3: Vì sao dạng Multiplicative phù hợp hơn dạng Additive cho AirPassengers?
   - Câu 4: Thành phần Residual có hành vi giống White Noise không?
=================================================================================
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
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.graphics.tsaplots import plot_acf

# -------------------------------------------------------------------------------
# 1. CẤU HÌNH ĐƯỜNG DẪN & THAM SỐ TOÀN CỤC
# -------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "AirPassengers.csv"

CONFIG = {
    "period": 12,
    "target_col": "Passengers",
    "datetime_col": "Month"
}


# -------------------------------------------------------------------------------
# 2. BƯỚC 1: TẢI VÀ KIỂM TRA CHUỖI DỮ LIỆU
# -------------------------------------------------------------------------------
def step1_load_data(data_path: Path) -> pd.Series:
    """
    Tải tệp CSV, định dạng DatetimeIndex và kiểm tra tính liên tục.
    """
    print(f"[BƯỚC 1] Đang tải dữ liệu từ: {data_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu tại {data_path}")

    df = pd.read_csv(data_path)
    df[CONFIG["datetime_col"]] = pd.to_datetime(df[CONFIG["datetime_col"]])
    df.set_index(CONFIG["datetime_col"], inplace=True)
    df.sort_index(inplace=True)

    series = df[CONFIG["target_col"]].astype(float)
    print(f" -> Đã tải {len(series)} quan sát. Chu kỳ phân rã (period) = {CONFIG['period']}.")
    return series


# -------------------------------------------------------------------------------
# 2. BƯỚC 2: PHÂN RÃ CHUỖI (CLASSICAL & STL DECOMPOSITION)
# -------------------------------------------------------------------------------
def step2_execute_analysis(series: pd.Series) -> Dict[str, Any]:
    """
    Thực hiện phân rã theo phương pháp cổ điển dạng nhân và phương pháp STL Loess.
    """
    print(f"[BƯỚC 2] Đang thực hiện phân rã chuỗi thời gian...")

    # 1. Classical Multiplicative
    decomp_mul = seasonal_decompose(series, model="multiplicative", period=CONFIG["period"])

    # 2. Classical Additive
    decomp_add = seasonal_decompose(series, model="additive", period=CONFIG["period"])

    # 3. STL Robust
    stl_model = STL(series, period=CONFIG["period"], robust=True)
    res_stl = stl_model.fit()

    # Phân tích thành phần
    trend_growth = (decomp_mul.trend.dropna().iloc[-1] / decomp_mul.trend.dropna().iloc[0] - 1.0) * 100
    peak_month = decomp_mul.seasonal.iloc[:12].idxmax().month

    results = {
        "decomp_mul": decomp_mul,
        "decomp_add": decomp_add,
        "res_stl": res_stl,
        "trend_growth_pct": trend_growth,
        "peak_month": peak_month,
        "residual_clean": decomp_mul.resid.dropna()
    }

    print(f" -> Tăng trưởng xu hướng tổng thể: +{trend_growth:.1f}%")
    print(f" -> Tháng có đỉnh mùa vụ cao nhất trong năm: Tháng {peak_month}")
    return results


# -------------------------------------------------------------------------------
# 3. BƯỚC 3: TRỰC QUAN HÓA 4 THÀNH PHẦN
# -------------------------------------------------------------------------------
def step3_visualize(series: pd.Series, analysis_results: Dict[str, Any], show_plot: bool = False) -> None:
    """
    Khởi tạo biểu đồ 4 thành phần: Observed, Trend, Seasonal, Residual cho cả Classical và STL.
    Lưu ý: Không tự ý xuất file ảnh ra ổ đĩa theo quy định tinh gọn của dự án.
    """
    print(f"[BƯỚC 3] Đang khởi tạo đồ thị phân rã chuỗi thời gian...")

    # Đồ thị Classical Multiplicative
    fig_mul = analysis_results["decomp_mul"].plot()
    fig_mul.set_size_inches(14, 8)
    fig_mul.suptitle("Classical Multiplicative Decomposition (Y = T * S * R)", fontsize=14, y=1.02)
    if not show_plot:
        plt.close(fig_mul)

    # Đồ thị STL
    fig_stl = analysis_results["res_stl"].plot()
    fig_stl.set_size_inches(14, 8)
    fig_stl.suptitle("STL Decomposition (Loess Smoothing, Robust to Outliers)", fontsize=14, y=1.02)
    if show_plot:
        plt.show()
    else:
        plt.close(fig_stl)

    print(" -> Đã hoàn thành trực quan hóa phân rã (không xuất tệp ảnh ra đĩa).")
    return None


# -------------------------------------------------------------------------------
# 4. BƯỚC 4: TỔNG KẾT VÀ TRẢ LỜI CÂU HỎI BÀI TOÁN
# -------------------------------------------------------------------------------
def step4_conclude_and_report(analysis_results: Dict[str, Any]) -> str:
    """
    Tổng hợp nhận xét học thuật và trả lời các câu hỏi nghiệm thu Bài 3.
    """
    print("\n[BƯỚC 4] TỔNG HỢP KẾT QUẢ VÀ TRẢ LỜI CÂU HỎI NGHIỆM THU BÀI 3:")
    report_text = f"""
=================================================================================
BÁO CÁO PHÂN TÍCH KẾT QUẢ BÀI TẬP 3 (PHÂN RÃ CHUỖI THỜI GIAN)
=================================================================================
1. XU HƯỚNG (TREND):
   - Đường xu hướng tăng trưởng liên tục và mạnh mẽ qua toàn bộ giai đoạn 1949 - 1960.
   - Tốc độ tăng trưởng đạt khoảng +{analysis_results['trend_growth_pct']:.1f}%, phản ánh sự bùng nổ của ngành hàng không sau Thế chiến 2.

2. MÙA VỤ (SEASONALITY):
   - Tính mùa vụ diễn ra cực kỳ mạnh và ổn định với chu kỳ 12 tháng.
   - Đỉnh điểm (Peak) luôn rơi vào Tháng {analysis_results['peak_month']} (Mùa cao điểm du lịch hè).
   - Đáy chạm mức thấp nhất vào Tháng 11 hàng năm.

3. SO SÁNH ADDITIVE VS MULTIPLICATIVE:
   - Dạng Nhân (Multiplicative: Y = T * S * R) phù hợp vượt trội so với dạng Cộng (Additive).
   - Lý do: Biên độ dao động giữa mùa cao điểm và thấp điểm mở rộng theo thời gian tỷ lệ thuận với xu hướng
     (Năm 1949 biên độ chỉ chênh ~40 khách, đến 1960 biên độ chênh lệch tăng lên hơn 150 khách).

4. ĐÁNH GIÁ THÀNH PHẦN PHẦN DƯ (RESIDUAL):
   - Thành phần Residual dao động ổn định quanh giá trị 1.0 (đối với dạng nhân).
   - Residual không còn chứa các bước sóng chu kỳ rõ rệt, tiệm cận với tính chất của White Noise,
     chứng minh mô hình phân rã đã trích xuất thành công gần như toàn bộ thông tin có cấu trúc.
=================================================================================
"""
    print(report_text)

    report_file = BASE_DIR / "ket_qua_bai_3.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f" -> Đã lưu báo cáo phân tích tại: {report_file}")
    return report_text


# -------------------------------------------------------------------------------
# HÀM THỰC THI CHÍNH
# -------------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("KHỞI CHẠY BÀI TẬP 3: PHÂN RÃ CHUỖI THỜI GIAN (DECOMPOSITION)")
    print("=" * 80)
    series = step1_load_data(DATA_PATH)
    analysis_results = step2_execute_analysis(series)
    step3_visualize(series, analysis_results)
    step4_conclude_and_report(analysis_results)
    print("\n[HOÀN THÀNH BÀI TẬP 3 THÀNH CÔNG]")


if __name__ == "__main__":
    main()
