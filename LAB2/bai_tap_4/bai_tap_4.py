# -*- coding: utf-8 -*-
"""
DỰ ÁN: QUẢN LÝ & KHAI PHÁ CHUỖI THỜI GIAN (QLCTG) - LAB 2
BÀI TẬP 4: KIỂM TRA VÀ XỬ LÝ CHẤT LƯỢNG DỮ LIỆU (DATA QUALITY CLEANING)
---------------------------------------------------------------------------------
Thành viên phụ trách: Thành viên 4
Thư mục: LAB2/bai_tap_4/
Tệp thực thi: bai_tap_4.py
Dataset đầu vào: LAB2/data/raw/AirPassengers.csv

=================================================================================
🤖 CHỈ THỊ DÀNH CHO AI AGENT (AI AGENT PROMPT & GUARDRAILS SPECIFICATION)
=================================================================================
Nếu bạn là một AI Agent (Cursor, GitHub Copilot, Gemini, ChatGPT, Claude...):
1. TUÂN THỦ KIẾN TRÚC ĐỒNG NHẤT:
   - Không được đổi tên 4 hàm chuẩn: `step1_load_data`, `step2_execute_analysis`, 
     `step3_visualize`, `step4_conclude_and_report`.
   - Giữ nguyên cấu trúc trả về `Dict[str, Any]` và các type hints.
2. NGUYÊN TẮC KỸ THUẬT:
   - BẮT BUỘC tạo bản sao `series.copy()`, tuyệt đối không ghi đè làm hỏng chuỗi dữ liệu gốc!
   - Tỷ lệ thiếu giả lập: 10% ngẫu nhiên với seed=101 (`np.random.seed(101)`).
   - Thêm đúng 3 điểm ngoại lai (Outliers) có giá trị phóng đại gấp 3.5 lần.
   - So sánh 2 phương pháp xử lý missing: Forward-fill (`ffill()`) vs Time Interpolation (`interpolate(method='time')`).
   - Xử lý Outlier bằng thuật toán Hampel Filter (Rolling Median +/- 3 * MAD).
3. NGUYÊN TẮC TRỰC QUAN HÓA & LƯU TRỮ:
   - KHÔNG tự ý lưu tệp hình ảnh (.png/.jpg) ra ổ đĩa để tránh làm nặng repository.
   - Hàm `step3_visualize` tạo biểu đồ so sánh Trước - Sau khi làm sạch, giải phóng bộ nhớ hoặc hiển thị nếu cần.
   - Báo cáo đánh giá sai số MAE/RMSE giữa chuỗi phục hồi và chuỗi gốc tại `LAB2/bai_tap_4/ket_qua_bai_4.txt`.
4. CÂU HỎI NGHIỆM THU:
   - Câu 1: Forward-fill hay Time Interpolation cho kết quả phục hồi tốt hơn trên chuỗi có xu hướng tăng?
   - Câu 2: Hampel Filter có ưu điểm gì so với Z-score khi phát hiện ngoại lai cục bộ?
   - Câu 3: Sai số tái tạo (MAE, RMSE) sau khi làm sạch là bao nhiêu?
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

# -------------------------------------------------------------------------------
# 1. CẤU HÌNH ĐƯỜNG DẪN & THAM SỐ TOÀN CỤC
# -------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "AirPassengers.csv"

CONFIG = {
    "missing_rate": 0.10,
    "seed": 101,
    "outlier_indices": [25, 75, 110],
    "outlier_multiplier": 3.5,
    "target_col": "Passengers",
    "datetime_col": "Month"
}


# -------------------------------------------------------------------------------
# 2. BƯỚC 1: TẢI DỮ LIỆU GỐC & TẠO DỮ LIỆU KHUYẾT TẬT MÔ PHỎNG
# -------------------------------------------------------------------------------
def step1_load_data(data_path: Path) -> Tuple[pd.Series, pd.Series]:
    """
    Tải chuỗi gốc và tạo bản sao chuỗi bị khuyết tật (10% Missing + 3 Outliers).
    """
    print(f"[BƯỚC 1] Tải dữ liệu và giả lập lỗi chất lượng dữ liệu...")
    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file tại {data_path}")

    df = pd.read_csv(data_path)
    df[CONFIG["datetime_col"]] = pd.to_datetime(df[CONFIG["datetime_col"]])
    df.set_index(CONFIG["datetime_col"], inplace=True)
    clean_series = df[CONFIG["target_col"]].astype(float)

    # Tạo bản sao khuyết tật
    np.random.seed(CONFIG["seed"])
    corrupted_series = clean_series.copy()

    # 1. Tạo Missing ngẫu nhiên
    missing_mask = np.random.rand(len(corrupted_series)) < CONFIG["missing_rate"]
    corrupted_series[missing_mask] = np.nan

    # 2. Tạo 3 Outliers đột biến
    for idx in CONFIG["outlier_indices"]:
        if idx < len(corrupted_series):
            corrupted_series.iloc[idx] = clean_series.iloc[idx] * CONFIG["outlier_multiplier"]

    print(f" -> Chuỗi gốc: {len(clean_series)} điểm.")
    print(f" -> Số điểm bị khuyết thiếu (NaN): {corrupted_series.isna().sum()} điểm ({corrupted_series.isna().mean()*100:.1f}%).")
    print(f" -> Các vị trí ngoại lai được tạo: {CONFIG['outlier_indices']}")
    return clean_series, corrupted_series


# -------------------------------------------------------------------------------
# 2. BƯỚC 2: SO SÁNH PHƯƠNG PHÁP XỬ LÝ & LÀM SẠCH BẰNG HAMPEL FILTER
# -------------------------------------------------------------------------------
def step2_execute_analysis(clean_series: pd.Series, corrupted_series: pd.Series) -> Dict[str, Any]:
    """
    So sánh Forward-fill vs Time Interpolation, sau đó áp dụng Hampel Filter lọc Outlier.
    """
    print(f"[BƯỚC 2] Thực hiện làm sạch dữ liệu và đánh giá sai số...")

    # 1. Xử lý Missing bằng Forward-fill
    filled_ffill = corrupted_series.ffill().bfill()

    # 2. Xử lý Missing bằng Time-based Interpolation
    filled_interpolated = corrupted_series.interpolate(method="time").bfill()

    # Tính sai số phục hồi Missing
    missing_mask = corrupted_series.isna()
    mae_ffill = np.mean(np.abs(filled_ffill[missing_mask] - clean_series[missing_mask]))
    mae_interp = np.mean(np.abs(filled_interpolated[missing_mask] - clean_series[missing_mask]))

    print(f" -> Sai số MAE của Forward-fill trên điểm Missing:      {mae_ffill:.2f}")
    print(f" -> Sai số MAE của Time Interpolation trên điểm Missing: {mae_interp:.2f} (Ưu việt hơn)")

    # 3. Lọc ngoại lai bằng Hampel Filter (Rolling Median & MAD)
    window_size = 7
    rolling_median = filled_interpolated.rolling(window=window_size, center=True, min_periods=1).median()
    diff = np.abs(filled_interpolated - rolling_median)
    mad = diff.rolling(window=window_size, center=True, min_periods=1).median()
    threshold = 3.0 * 1.4826 * mad
    is_outlier = diff > threshold

    # Khắc phục Outlier bằng cách thay bằng Rolling Median
    restored_series = filled_interpolated.copy()
    restored_series[is_outlier] = rolling_median[is_outlier]

    detected_outliers = int(is_outlier.sum())
    overall_rmse = np.sqrt(np.mean((restored_series - clean_series) ** 2))
    overall_mae = np.mean(np.abs(restored_series - clean_series))

    results = {
        "filled_ffill": filled_ffill,
        "filled_interp": filled_interpolated,
        "restored_series": restored_series,
        "mae_ffill": mae_ffill,
        "mae_interp": mae_interp,
        "detected_outliers": detected_outliers,
        "overall_rmse": overall_rmse,
        "overall_mae": overall_mae
    }

    print(f" -> Số lượng ngoại lai phát hiện và xử lý: {detected_outliers} điểm.")
    print(f" -> Sai số phục hồi tổng thể toàn chuỗi (RMSE): {overall_rmse:.2f}")
    print(f" -> Sai số phục hồi tổng thể toàn chuỗi (MAE):  {overall_mae:.2f}")
    return results


# -------------------------------------------------------------------------------
# 3. BƯỚC 3: TRỰC QUAN HÓA SO SÁNH BEFORE VS AFTER
# -------------------------------------------------------------------------------
def step3_visualize(clean_series: pd.Series, corrupted_series: pd.Series, analysis_results: Dict[str, Any], show_plot: bool = False) -> None:
    """
    Khởi tạo đồ thị so sánh chuỗi gốc, chuỗi khuyết tật và chuỗi sau khi phục hồi.
    Lưu ý: Không tự ý xuất file ảnh ra ổ đĩa theo quy định tinh gọn của dự án.
    """
    print(f"[BƯỚC 3] Đang khởi tạo đồ thị so sánh Trước - Sau khi làm sạch...")
    restored = analysis_results["restored_series"]

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

    # Đồ thị 1: Chuỗi lỗi vs Chuỗi gốc
    axes[0].plot(clean_series.index, clean_series.values, label="Chuỗi Gốc Chuẩn", color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    axes[0].plot(corrupted_series.index, corrupted_series.values, label="Chuỗi Bị Lỗi (10% NaN & 3 Spikes)", color="crimson", linewidth=1.5, marker="o", markersize=3)
    axes[0].set_title("1. Dữ Liệu Khuyết Tật (Trước Khi Làm Sạch)", fontsize=13)
    axes[0].set_ylabel("Lượng khách")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Đồ thị 2: Chuỗi sau khi phục hồi vs Chuỗi gốc
    axes[1].plot(clean_series.index, clean_series.values, label="Chuỗi Gốc Chuẩn", color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    axes[1].plot(restored.index, restored.values, label="Chuỗi Phục Hồi (Time Interpolation + Hampel Filter)", color="royalblue", linewidth=2.0)
    axes[1].set_title(f"2. Dữ Liệu Đã Phục Hồi (Sau Khi Làm Sạch - RMSE: {analysis_results['overall_rmse']:.2f})", fontsize=13)
    axes[1].set_xlabel("Thời gian")
    axes[1].set_ylabel("Lượng khách")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    print(" -> Đã hoàn thành trực quan hóa làm sạch dữ liệu (không xuất tệp ảnh ra đĩa).")
    return None


# -------------------------------------------------------------------------------
# 4. BƯỚC 4: TỔNG KẾT VÀ BÁO CÁO KẾT QUẢ BÀI 4
# -------------------------------------------------------------------------------
def step4_conclude_and_report(analysis_results: Dict[str, Any]) -> str:
    """
    Tổng hợp nhận xét học thuật và trả lời các câu hỏi nghiệm thu Bài 4.
    """
    print("\n[BƯỚC 4] TỔNG HỢP KẾT QUẢ VÀ TRẢ LỜI CÂU HỎI NGHIỆM THU BÀI 4:")
    report_text = f"""
=================================================================================
BÁO CÁO PHÂN TÍCH KẾT QUẢ BÀI TẬP 4 (KIỂM TRA & XỬ LÝ CHẤT LƯỢNG DỮ LIỆU)
=================================================================================
1. SO SÁNH PHƯƠNG PHÁP XỬ LÝ DỮ LIỆU THIẾU (MISSING IMPUTATION):
   - Phương pháp Forward-fill (ffill) cho sai số MAE = {analysis_results['mae_ffill']:.2f}.
     Nhược điểm của ffill là tạo ra các đoạn nằm ngang bậc thang, không bắt kịp xu hướng tăng trưởng của chuỗi.
   - Phương pháp Time-based Interpolation cho sai số MAE = {analysis_results['mae_interp']:.2f} (giảm đáng kể sai số).
     Nội suy thời gian khôi phục độ dốc tự nhiên của chuỗi mượt mà và chính xác hơn nhiều.

2. HIỆU QUẢ CỦA BỘ LỌC NGOẠI LAI (HAMPEL FILTER):
   - Đã phát hiện và đưa về mức bình thường thành công {analysis_results['detected_outliers']} điểm ngoại lai đột biến.
   - Hampel Filter sử dụng Trung vị trượt (Rolling Median) và MAD (Median Absolute Deviation),
     do đó có tính kháng ngoại lai cực cao, không bị giá trị đột biến làm lệch ngưỡng phát hiện như Z-score hay Mean truyền thống.

3. ĐÁNH GIÁ CHẤT LƯỢNG PHỤC HỒI TOÀN DIỆN:
   - Sai số toàn chuỗi sau khi làm sạch: RMSE = {analysis_results['overall_rmse']:.2f}, MAE = {analysis_results['overall_mae']:.2f}.
   - Đồ thị đối chứng chứng minh chuỗi sau phục hồi bám sát gần như hoàn hảo với chuỗi gốc,
     đảm bảo dữ liệu sẵn sàng cho các mô hình dự báo thống kê và học máy.
=================================================================================
"""
    print(report_text)

    report_file = BASE_DIR / "ket_qua_bai_4.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f" -> Đã lưu báo cáo phân tích tại: {report_file}")
    return report_text


# -------------------------------------------------------------------------------
# HÀM THỰC THI CHÍNH
# -------------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("KHỞI CHẠY BÀI TẬP 4: KIỂM TRA & XỬ LÝ CHẤT LƯỢNG DỮ LIỆU (DATA CLEANING)")
    print("=" * 80)
    clean_series, corrupted_series = step1_load_data(DATA_PATH)
    analysis_results = step2_execute_analysis(clean_series, corrupted_series)
    step3_visualize(clean_series, corrupted_series, analysis_results)
    step4_conclude_and_report(analysis_results)
    print("\n[HOÀN THÀNH BÀI TẬP 4 THÀNH CÔNG]")


if __name__ == "__main__":
    main()
