"""
================================================================================
LAB 2 - BÀI TẬP 5: TRỰC QUAN HÓA CHUYÊN SÂU CHUỖI THỜI GIAN (TIME SERIES EDA & VISUALIZATION)
NHÓM 7 - MÔN: PHÂN TÍCH CHUỖI THỜI GIAN
================================================================================

AI_AGENT_INSTRUCTIONS:
----------------------
Nếu bạn là một AI Agent (Cursor, GitHub Copilot, Claude, ChatGPT...) hỗ trợ thành viên nhóm hoàn thiện hoặc mở rộng file này,
hãy TUÂN THỦ NGHIÊM NGẶT các quy tắc kiến trúc sau:
1. NGUYÊN TẮC BẢO TOÀN KIẾN TRÚC ĐỒNG BỘ:
   - File bắt buộc phải chia thành 4 hàm tuần tự:
     + step1_load_data(data_path: Path) -> pd.Series
     + step2_execute_analysis(series: pd.Series, window: int = 12) -> Dict[str, Any]
     + step3_visualize(analysis_results: Dict[str, Any], show_plot: bool = False) -> None
     + step4_conclude_and_report(analysis_results: Dict[str, Any], output_path: Path) -> str
   - Hàm main() điều phối toàn bộ luồng thực thi, in log rõ ràng và lưu file kết quả.
2. NGUYÊN TẮC TRỰC QUAN HÓA & LƯU TRỮ:
   - KHÔNG tự ý lưu tệp hình ảnh (.png/.jpg) ra ổ đĩa để tránh làm nặng repository.
   - Phân tích đầy đủ 3 góc nhìn trực quan:
     (a) Rolling Statistics (Mean & Band ±2 Std): Đánh giá xu hướng và tính biến thiên phương sai (heteroskedasticity).
     (b) Monthly Seasonal Plot (Từng năm qua 12 tháng): Quan sát pha sóng và sự dịch chuyển tăng trưởng liên năm.
     (c) Seasonality Heatmap (Year x Month): Thể hiện mật độ hành khách qua bảng màu heatmap trực quan.
   - Hàm `step3_visualize` tạo biểu đồ và giải phóng bộ nhớ (hoặc hiển thị nếu show_plot=True).
3. TÍNH TƯƠNG THÍCH MÔI TRƯỜNG WINDOWS:
   - Luôn sử dụng encoding='utf-8' cho mọi thao tác đọc/ghi file và luồng I/O console.
================================================================================
"""

import sys
import os
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Thiết lập bảng mã UTF-8 cho console Windows nhằm ngăn ngừa UnicodeEncodeError
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def step1_load_data(data_path: Path) -> pd.Series:
    """
    Bước 1: Nạp và chuẩn hóa dữ liệu chuỗi thời gian AirPassengers.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"[LỖI] Không tìm thấy tệp dữ liệu tại: {data_path.resolve()}")
    
    df = pd.read_csv(data_path)
    
    # Chuẩn hóa tên cột
    date_col = [c for c in df.columns if 'date' in c.lower() or 'month' in c.lower()][0]
    val_col = [c for c in df.columns if c != date_col][0]
    
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    df.set_index(date_col, inplace=True)
    
    # Thiết lập chu kỳ theo tháng
    series = df[val_col].asfreq('MS')
    series.name = "Passengers"
    
    return series


def step2_execute_analysis(series: pd.Series, window: int = 12) -> Dict[str, Any]:
    """
    Bước 2: Tính toán các chỉ số thống kê trượt và tái cấu trúc dữ liệu theo Mùa vụ (Year x Month).
    """
    # 1. Thống kê trượt (Rolling Statistics)
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper_band = rolling_mean + 2 * rolling_std
    lower_band = rolling_mean - 2 * rolling_std
    
    # 2. Tái cấu trúc ma trận Pivot (Năm x Tháng)
    df_pivot = series.to_frame(name="Passengers").copy()
    df_pivot.index.name = "Date"
    df_pivot["Year"] = df_pivot.index.year
    df_pivot["Month_Num"] = df_pivot.index.month
    
    pivot_table = df_pivot.pivot(index="Year", columns="Month_Num", values="Passengers")
    
    # 3. Tính toán các chỉ số mùa vụ
    # Điểm trung bình theo tháng trên toàn bộ các năm
    monthly_avg = df_pivot.groupby("Month_Num")["Passengers"].mean()
    peak_month = int(monthly_avg.idxmax())
    trough_month = int(monthly_avg.idxmin())
    
    # Đánh giá sự mở rộng của phương sai theo thời gian (Heteroskedasticity)
    std_first_year = series.iloc[:12].std()
    std_last_year = series.iloc[-12:].std()
    std_ratio = std_last_year / std_first_year
    
    return {
        "series": series,
        "rolling_mean": rolling_mean,
        "rolling_std": rolling_std,
        "upper_band": upper_band,
        "lower_band": lower_band,
        "pivot_table": pivot_table,
        "monthly_avg": monthly_avg,
        "peak_month": peak_month,
        "trough_month": trough_month,
        "std_first_year": std_first_year,
        "std_last_year": std_last_year,
        "std_ratio": std_ratio,
        "years": sorted(df_pivot["Year"].unique().tolist())
    }


def step3_visualize(analysis_results: Dict[str, Any], show_plot: bool = False) -> None:
    """
    Bước 3: Trực quan hóa dữ liệu gồm Rolling Statistics, Seasonal Lines và Heatmap.
    Lưu ý: Không tự ý xuất file ảnh ra ổ đĩa theo quy định tinh gọn của dự án.
    """
    series = analysis_results["series"]
    rmean = analysis_results["rolling_mean"]
    upper_band = analysis_results["upper_band"]
    lower_band = analysis_results["lower_band"]
    pivot_table = analysis_results["pivot_table"]
    years = analysis_results["years"]
    
    # =========================================================================
    # ĐỒ THỊ 1: ROLLING STATISTICS & MONTHLY SEASONAL LINES
    # =========================================================================
    fig1, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Subplot 1: Rolling Mean & Bands
    axes[0].plot(series.index, series.values, label='Chuỗi gốc (AirPassengers)', color='#2b5c8f', linewidth=1.5, alpha=0.85)
    axes[0].plot(rmean.index, rmean.values, label='12-Month Rolling Mean (Trend dài hạn)', color='#d95f02', linewidth=2.5)
    axes[0].fill_between(series.index, lower_band, upper_band, color='#fdc086', alpha=0.35, label='Dải biên độ dao động ±2 Rolling Std')
    axes[0].set_title("1. Phân tích Thống kê Trượt 12 Tháng & Dải Biến thiên Độ lệch chuẩn (±2σ)", fontsize=13, fontweight='bold', pad=10)
    axes[0].set_xlabel("Thời gian (Năm)", fontsize=11)
    axes[0].set_ylabel("Số lượng hành khách (nghìn người)", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")
    
    # Subplot 2: Monthly Seasonal Plot (Từng năm qua 12 tháng)
    colors = plt.cm.viridis(np.linspace(0, 1, len(years)))
    for i, yr in enumerate(years):
        row_data = pivot_table.loc[yr]
        axes[1].plot(row_data.index, row_data.values, marker='o', markersize=4, label=str(yr), color=colors[i], linewidth=1.6, alpha=0.85)
    
    axes[1].set_title("2. Biểu đồ Mùa vụ Theo Tháng Qua Các Năm (1949 - 1960)", fontsize=13, fontweight='bold', pad=10)
    axes[1].set_xlabel("Tháng trong năm (1 đến 12)", fontsize=11)
    axes[1].set_ylabel("Số lượng hành khách (nghìn người)", fontsize=11)
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xticklabels([f"T{m}" for m in range(1, 13)])
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(title="Năm", bbox_to_anchor=(1.01, 1), loc="upper left", ncol=1, frameon=True)
    
    plt.tight_layout()
    if not show_plot:
        plt.close(fig1)
    
    # =========================================================================
    # ĐỒ THỊ 2: SEASONALITY HEATMAP (BẢN ĐỒ NHIỆT NĂM X THÁNG)
    # =========================================================================
    fig2 = plt.figure(figsize=(13, 7))
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt="g",
        cmap="YlGnBu",
        cbar_kws={'label': 'Lượng hành khách (Nghìn người)'},
        linewidths=0.5,
        linecolor='white'
    )
    plt.title("3. Bản đồ nhiệt Mùa vụ (Seasonality Heatmap): Năm vs Tháng", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Tháng trong năm", fontsize=12)
    plt.ylabel("Năm quan sát", fontsize=12)
    plt.xticks(ticks=[i + 0.5 for i in range(12)], labels=[f"Tháng {m}" for m in range(1, 13)], rotation=0)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    if show_plot:
        plt.show()
    else:
        plt.close(fig2)
    
    print(" -> Đã hoàn thành trực quan hóa Rolling Statistics và Heatmap (không xuất tệp ảnh ra đĩa).")
    return None


def step4_conclude_and_report(analysis_results: Dict[str, Any], output_path: Path) -> str:
    """
    Bước 4: Tổng hợp nhận định chuyên sâu và xuất báo cáo nghiệm thu Bài tập 5.
    """
    peak_m = analysis_results["peak_month"]
    trough_m = analysis_results["trough_month"]
    std_init = analysis_results["std_first_year"]
    std_end = analysis_results["std_last_year"]
    ratio = analysis_results["std_ratio"]
    
    report_text = f"""=================================================================================
BÁO CÁO PHÂN TÍCH KẾT QUẢ BÀI TẬP 5 (TRỰC QUAN HÓA CHUYÊN SÂU CHUỖI THỜI GIAN)
=================================================================================
1. PHÂN TÍCH XU HƯỚNG VÀ DẢI BIẾN THIÊN (ROLLING STATISTICS):
   - Đường trung bình trượt 12 tháng (Rolling Mean) làm mịn hoàn toàn các dao động sóng ngắn,
     cho thấy rõ ràng xu hướng tăng trưởng tuyến tính dốc mạnh qua từng năm.
   - Dải biên độ dao động ±2 Rolling Std mở rộng rõ rệt theo thời gian:
     + Độ lệch chuẩn nội năm 1949: {std_init:.2f}
     + Độ lệch chuẩn nội năm 1960: {std_end:.2f} (Gấp {ratio:.2f} lần so với 1949).
   - Kết luận phương sai: Chuỗi vi phạm giả định phương sai đồng nhất (Heteroskedasticity).
     Biên độ dao động tăng tỷ lệ thuận với mức độ lớn của chuỗi, minh chứng dạng mô hình nhân (Multiplicative)
     hoặc phép biến đổi Box-Cox / Log-transform là tối cần thiết khi xây dựng mô hình dự báo.

2. QUY LUẬT MÙA VỤ THEO THÁNG (MONTHLY SEASONALITY):
   - Biểu đồ Seasonal Plot cho thấy các đường cong qua từng năm có hình dáng (pha dao động)
     đồng dạng gần như tuyệt đối, chứng minh tính mùa vụ mang tính quy luật tự nhiên rất cao.
   - Đỉnh điểm hành khách (Peak): Tháng {peak_m} (Tháng 7 - mùa nghỉ hè quốc tế) luôn đạt lưu lượng cao nhất.
   - Đáy thấp nhất (Trough): Tháng {trough_m} (Tháng 11 - giai đoạn chuyển giao mùa thấp điểm du lịch).
   - Mỗi năm đường cong mùa vụ tịnh tiến đều lên phía trên mà không làm đảo lộn cấu trúc sóng.

3. ĐÁNH GIÁ MẬT ĐỘ QUA HEATMAP (YEAR X MONTH):
   - Bản đồ nhiệt thể hiện rõ sự chuyển màu từ tông vàng nhạt (năm 1949, lưu lượng ~100-150)
     sang tông xanh lục đậm (năm 1960, lưu lượng vượt ngưỡng 500-600).
   - "Dải màu nóng" (nồng độ hành khách cao nhất) tập trung liên tục vào cột Tháng 7 và Tháng 8 qua mọi năm.
   - Sự kết hợp trực quan giữa Heatmap và Rolling Statistics cung cấp bức tranh toàn diện
     về cả 2 chiều không gian (thời điểm trong năm) và thời gian (xu hướng liên năm).
=================================================================================
"""
    # Ghi báo cáo ra file text
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
        
    return report_text


def main():
    """
    Hàm thực thi chính của Bài tập 5.
    """
    print("=" * 80)
    print("KHỞI CHẠY BÀI TẬP 5: TRỰC QUAN HÓA CHUYÊN SÂU CHUỖI THỜI GIAN")
    print("=" * 80)
    
    current_dir = Path(__file__).resolve().parent
    data_path = current_dir.parent / "data" / "raw" / "AirPassengers.csv"
    report_file = current_dir / "ket_qua_bai_5.txt"
    
    # Bước 1
    print(f"[BƯỚC 1] Đang tải dữ liệu từ: {data_path}")
    series = step1_load_data(data_path)
    print(f" -> Đã nạp {len(series)} quan sát từ {series.index.min().strftime('%Y-%m')} đến {series.index.max().strftime('%Y-%m')}.")
    
    # Bước 2
    print("[BƯỚC 2] Tính toán các chỉ số thống kê trượt và ma trận Mùa vụ...")
    analysis_results = step2_execute_analysis(series, window=12)
    print(f" -> Tháng đỉnh điểm cao nhất: Tháng {analysis_results['peak_month']}")
    print(f" -> Tháng đáy thấp nhất: Tháng {analysis_results['trough_month']}")
    print(f" -> Tỷ lệ tăng phương sai (1960 vs 1949): {analysis_results['std_ratio']:.2f} lần.")
    
    # Bước 3
    print("[BƯỚC 3] Đang khởi tạo đồ thị trực quan chuyên sâu...")
    step3_visualize(analysis_results)
        
    # Bước 4
    print("\n[BƯỚC 4] TỔNG HỢP KẾT QUẢ VÀ TRẢ LỜI CÂU HỎI NGHIỆM THU BÀI 5:\n")
    report = step4_conclude_and_report(analysis_results, report_file)
    print(report)
    print(f" -> Đã lưu báo cáo phân tích tại: {report_file}")
    print("\n[HOÀN THÀNH BÀI TẬP 5 THÀNH CÔNG]")


if __name__ == "__main__":
    main()
