"""
================================================================================
LAB 2 - BAI TAP 5: TRUC QUAN HOA CHUYEN SAU CHUOI THOI GIAN (TIME SERIES EDA & VISUALIZATION)
NHOM 7 - MON: QUAN LY & KHAI PHA CHUOI THOI GIAN (QLCTG) - UTH
Thanh vien phu trach: Bui Tran Ngoc Khai (CV06) - Nhanh: ngoc-khai-bt5
================================================================================

AI_AGENT_INSTRUCTIONS:
----------------------
Nếu bạn là một AI Agent (Cursor, GitHub Copilot, Claude, ChatGPT...) hỗ trợ thành viên nhóm hoàn thiện hoặc mở rộng file này,
hãy TUÂN THỦ NGHIÊM NGẶT các quy tắc kiến trúc sau:
1. NGUYÊN TẮC BẢO TOÀN KIẾN TRÚC ĐỒNG BỘ:
   - File bắt buộc phải chia thành 4 hàm tuần tự:
     + step1_load_data(data_path: Path) -> pd.Series
     + step2_execute_analysis(series: pd.Series, window: int = 12) -> Dict[str, Any]
+ step3_visualize(series: pd.Series, analysis_results: Dict[str, Any], show_plot: bool = False, figures_dir: Path = None) -> None
      + step4_conclude_and_report(analysis_results: Dict[str, Any], output_path: Path) -> str
    - Hàm main() điều phối toàn bộ luồng thực thi, in log rõ ràng và lưu file kết quả.
2. NGUYÊN TẮC TRỰC QUAN HÓA & LƯU TRỮ:
   - Theo lựa chọn của thành viên CV06, bài này ĐƯỢC PHÉP lưu 3 biểu đồ PNG vào
     `LAB2/bai_tap_5/figures/` để nộp kèm báo cáo kỹ thuật:
     bai5_1_rolling_statistics.png, bai5_2_seasonal_plot.png, bai5_3_seasonality_heatmap.png (150 dpi).
     Chạy với cờ `--no-figures` để tắt lưu ảnh nếu cần giữ repo tinh gọn.
   - Phân tích đầy đủ 3 góc nhìn trực quan:
     (a) Rolling Statistics (Mean & Band ±2 Std): Đánh giá xu hướng và tính biến thiên phương sai (heteroskedasticity).
     (b) Monthly Seasonal Plot (Từng năm qua 12 tháng): Quan sát pha sóng và sự dịch chuyển tăng trưởng liên năm.
     (c) Seasonality Heatmap (Year x Month): Thể hiện mật độ hành khách qua bảng màu heatmap trực quan.
   - Hàm `step3_visualize` tạo biểu đồ và giải phóng bộ nhớ (hoặc hiển thị nếu show_plot=True).
3. TÍNH TƯƠNG THÍCH MÔI TRƯỜNG WINDOWS:
   - Luôn sử dụng encoding='utf-8' cho mọi thao tác đọc/ghi file và luồng I/O console.
4. CAC CAU HOI NGHIEM THU & LY THUYET BAT BUOC TRA LOI:
   - Cau 1: 12-Month Rolling Mean giup kiem tra dieu gi ve du lieu? (cau ly thuyet 9)
   - Cau 2: Tai sao truc quan hoa la buoc quan trong trong phan tich chuoi thoi gian? (cau ly thuyet 12)
   - Cau 3: Cac thang nao dat diem mua vu cao nhat va thap nhat? Minh chung dinh luong?
================================================================================
"""

import sys
import os
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Thiết lập bảng mã UTF-8 cho console Windows nhằm ngăn ngừa UnicodeEncodeError
if sys.platform == "win32":
    try:
        os.system("chcp 65001 >nul 2>&1")
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Cấu hình font chữ hỗ trợ tiếng Việt đầy đủ và đẹp trên Windows / Linux / macOS
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "Tahoma", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False


def step1_load_data(data_path: Path) -> pd.Series:
    """
    Bước 1: Nạp và kiểm định định dạng dữ liệu chuỗi thời gian AirPassengers.

    Mục tiêu kiểm định:
        - Tồn tại tệp dữ liệu đầu vào.
        - Chuỗi thời gian liên tục, tần suất tháng đầy đủ 'MS' (Month Start).
        - Không có giá trị thiếu (NaN) trước khi trực quan hóa.
    """
    print(f"[BƯỚC 1] Đang tải dữ liệu từ: {data_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"[LỖI] Không tìm thấy tệp dữ liệu tại: {data_path.resolve()}")

    df = pd.read_csv(data_path)

    # Chuẩn hóa tên cột một cách bền vững (không hard-code tên cột)
    date_col = [c for c in df.columns if 'date' in c.lower() or 'month' in c.lower()][0]
    val_col = [c for c in df.columns if c != date_col][0]

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    df.set_index(date_col, inplace=True)

    # Thiết lập chu kỳ theo tháng (Month Start)
    series = df[val_col].asfreq('MS')
    series.name = "Passengers"

    # --- Kiểm định chất lượng chuỗi thời gian ---
    n_missing = int(series.isna().sum())
    if n_missing > 0:
        print(f" -> [CẢNH BÁO] Phát hiện {n_missing} giá trị thiếu sau khi thiết lập tần suất 'MS'.")

    print(f" -> Số quan sát: {len(series)} | Tần suất: {series.index.freqstr} | Thiếu giá trị: {n_missing}")
    print(f" -> Khoảng thời gian: {series.index.min():%Y-%m} đến {series.index.max():%Y-%m}")
    print(f" -> Giá trị nhỏ nhất: {series.min():.0f} (tháng {series.idxmin():%Y-%m}) | "
          f"Giá trị lớn nhất: {series.max():.0f} (tháng {series.idxmax():%Y-%m})")
    return series


def step2_execute_analysis(series: pd.Series, window: int = 12) -> Dict[str, Any]:
    """
    Bước 2: Tính toán các chỉ số thống kê trượt và tái cấu trúc dữ liệu theo Mùa vụ (Year x Month).

    Các phép tính cốt lõi:
        1. Thống kê trượt (Rolling Statistics): Rolling Mean (window=12), Rolling Std,
           dải ±2σ và tỉ lệ phủ trong dải (đánh giá tính chuẩn & phương sai).
        2. Ma trận Pivot (Year x Month) phục vụ Seasonal Plot và Heatmap.
        3. Thống kê mùa vụ: TB theo tháng, tháng đỉnh/đáy, biên độ mùa vụ, tổng năm,
           và minh chứng heteroskedasticity qua Std(1949) vs Std(1960).
    """
    print(f"[BƯỚC 2] Tính toán các chỉ số thống kê trượt (window={window}) và ma trận Mùa vụ...")
    # 1. Thống kê trượt (Rolling Statistics)
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper_band = rolling_mean + 2 * rolling_std
    lower_band = rolling_mean - 2 * rolling_std

    # Chỉ số phủ trong dải ±2σ (xấp xỉ quy tắc 95% nếu dư phân phối chuẩn)
    valid = (~rolling_mean.isna()).sum()
    inside_band = int(((series >= lower_band) & (series <= upper_band)).sum()) if valid else 0
    coverage_pct = float(inside_band / valid * 100.0) if valid else float("nan")

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

    # Chuỗi tổng theo năm (đánh giá xu hướng dài hạn định lượng)
    annual_sum = series.resample("YE").sum()
    yoy_growth_pct = annual_sum.pct_change() * 100.0
    total_growth_factor = float(annual_sum.iloc[-1] / annual_sum.iloc[0]) if len(annual_sum) >= 2 else float("nan")
    total_growth_pct = float((total_growth_factor - 1.0) * 100.0) if not np.isnan(total_growth_factor) else float("nan")

    # Biên độ dao động nội năm: max(tháng) - min(tháng) theo từng năm
    amplitude_by_year = pivot_table.max(axis=1) - pivot_table.min(axis=1)

    # Đánh giá sự mở rộng của phương sai theo thời gian (Heteroskedasticity)
    std_first_year = float(series.iloc[:12].std())
    std_last_year = float(series.iloc[-12:].std())
    std_ratio = float(std_last_year / std_first_year) if std_first_year != 0 else float("nan")

    print(f" -> Tháng TB cao nhất: T{peak_month} ({monthly_avg.loc[peak_month]:.1f}) | "
          f"Thấp nhất: T{trough_month} ({monthly_avg.loc[trough_month]:.1f})")
    print(f" -> Std(1949)={std_first_year:.2f} | Std(1960)={std_last_year:.2f} | Tỉ lệ={std_ratio:.2f}×")
    print(f" -> Tổng năm: {int(annual_sum.iloc[0])} (1949) -> {int(annual_sum.iloc[-1])} (1960) "
          f"[{total_growth_factor:.2f}× / +{total_growth_pct:.1f}%]")
    print(f" -> Biên độ mùa vụ tăng {int(amplitude_by_year.iloc[0])} -> {int(amplitude_by_year.iloc[-1])} "
          f"(gấp {float(amplitude_by_year.iloc[-1]/amplitude_by_year.iloc[0]):.2f}×)")
    print(f" -> Điểm trong dải ±2σ: {inside_band}/{valid} ({coverage_pct:.1f}%)")

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
        "years": sorted(df_pivot["Year"].unique().tolist()),
        "annual_sum": annual_sum,
        "yoy_growth_pct": yoy_growth_pct,
        "total_growth_factor": total_growth_factor,
        "total_growth_pct": total_growth_pct,
        "amplitude_by_year": amplitude_by_year,
        "coverage_pct": coverage_pct,
    }


def step3_visualize(series: pd.Series, analysis_results: Dict[str, Any], show_plot: bool = False,
                    figures_dir: Path = None) -> None:
    """
    Bước 3: Trực quan hóa dữ liệu gồm Rolling Statistics, Seasonal Lines và Heatmap.

    Mặc định KHÔNG xuất tệp ảnh ra đĩa (quy định tinh gọn repo của Nhóm 7).
    Nếu truyền `figures_dir` (thư mục đích), hàm sẽ lưu 3 biểu đồ PNG ở độ phân giải 150 dpi
    để nộp kèm báo cáo kỹ thuật:
        1. bai5_1_rolling_statistics.png   -> Line chart + Rolling Mean + dải ±2σ
        2. bai5_2_seasonal_plot.png        -> Seasonal Plot 12 đường cong (đỉnh T7, đáy T11)
        3. bai5_3_seasonality_heatmap.png  -> Seasonality Heatmap (Năm x Tháng)
    """
    series = analysis_results["series"]
    rmean = analysis_results["rolling_mean"]
    upper_band = analysis_results["upper_band"]
    lower_band = analysis_results["lower_band"]
    pivot_table = analysis_results["pivot_table"]
    years = analysis_results["years"]

    print("[BƯỚC 3] Đang khởi tạo đồ thị trực quan chuyên sâu...")
    # =========================================================================
    # ĐỒ THỊ 1: BIỂU ĐỒ ĐƯỜNG TOÀN BỘ CHUỖI + ROLLING MEAN & DẢI ±2σ
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(14, 6))

    ax1.plot(series.index, series.values, label='Chuỗi gốc (AirPassengers)', color='#2b5c8f', linewidth=1.5, alpha=0.85)
    ax1.plot(rmean.index, rmean.values, label='12-Month Rolling Mean (Trend dài hạn)', color='#d95f02', linewidth=2.5)
    ax1.plot(analysis_results["rolling_std"].index, analysis_results["rolling_std"].values,
             label='12-Month Rolling Std (σ)', color='#2e7d32', linewidth=1.2, linestyle=':')
    ax1.fill_between(series.index, lower_band, upper_band, color='#fdc086', alpha=0.35, label='Dải biên độ dao động ±2 Rolling Std')
    ax1.set_title("1. Biểu đồ đường toàn bộ chuỗi AirPassengers (1949-1960)\n"
                  "kèm 12-Month Rolling Mean và dải biên độ ±2 Rolling Std",
                  fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel("Thời gian (Năm)", fontsize=11)
    ax1.set_ylabel("Số lượng hành khách (nghìn người)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")

    plt.tight_layout()
    if figures_dir is not None:
        figures_dir.mkdir(parents=True, exist_ok=True)
        out1 = figures_dir / "bai5_1_rolling_statistics.png"
        with open(out1, "wb") as f:
            fig1.savefig(f, format="png", dpi=150, bbox_inches="tight")
        print(f" -> Đã lưu biểu đồ 1 (Rolling Statistics): {out1}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig1)

    # =========================================================================
    # ĐỒ THỊ 2: SEASONAL PLOT THEO THÁNG (12 ĐƯỜNG CONG QUA CÁC NĂM)
    # =========================================================================
    fig_seasonal, ax2 = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(years)))
    for i, yr in enumerate(years):
        row_data = pivot_table.loc[yr]
        ax2.plot(row_data.index, row_data.values, marker='o', markersize=4, label=str(yr),
                 color=colors[i], linewidth=1.6, alpha=0.85)

    ax2.set_title("2. Biểu đồ Mùa vụ (Seasonal Plot) theo tháng qua các năm 1949-1960\n"
                  f"(Đỉnh mùa vụ: T{analysis_results['peak_month']} | Đáy: T{analysis_results['trough_month']})",
                  fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel("Tháng trong năm (1 đến 12)", fontsize=11)
    ax2.set_ylabel("Số lượng hành khách (nghìn người)", fontsize=11)
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels([f"T{m}" for m in range(1, 13)])
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(title="Năm", bbox_to_anchor=(1.01, 1), loc="upper left", ncol=1, frameon=True)

    plt.tight_layout()
    if figures_dir is not None:
        out2 = figures_dir / "bai5_2_seasonal_plot.png"
        with open(out2, "wb") as f:
            fig_seasonal.savefig(f, format="png", dpi=150, bbox_inches="tight")
        print(f" -> Đã lưu biểu đồ 2 (Seasonal Plot): {out2}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig_seasonal)

    # =========================================================================
    # ĐỒ THỊ 3: SEASONALITY HEATMAP (BẢN ĐỒ NHIỆT NĂM X THÁNG)
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
    if figures_dir is not None:
        out3 = figures_dir / "bai5_3_seasonality_heatmap.png"
        with open(out3, "wb") as f:
            fig2.savefig(f, format="png", dpi=150, bbox_inches="tight")
        print(f" -> Đã lưu biểu đồ 3 (Seasonality Heatmap): {out3}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig2)

    print(" -> Đã hoàn thành trực quan hóa Rolling Statistics và Heatmap.")
    return None


def _month_name(m: int) -> str:
    """Trả về tên tháng tiếng Việt."""
    names = {1: "Tháng 1", 2: "Tháng 2", 3: "Tháng 3", 4: "Tháng 4", 5: "Tháng 5", 6: "Tháng 6",
             7: "Tháng 7", 8: "Tháng 8", 9: "Tháng 9", 10: "Tháng 10", 11: "Tháng 11", 12: "Tháng 12"}
    return names.get(int(m), f"Tháng {int(m)}")


def step4_conclude_and_report(analysis_results: Dict[str, Any], output_path: Path) -> str:
    """
    Bước 4: Tổng hợp nhận định chuyên sâu, xuất báo cáo nghiệm thu Bài tập 5
            và trả lời đầy đủ 5 câu nghiệm thu đề bài + câu lý thuyết 9 & 12.

    Cấu trúc báo cáo 5 phần:
        1. Phân tích xu hướng & dải biến thiên (±2σ)
        2. Quy luật mùa vụ (Seasonal Plot)
        3. Đánh giá mật độ qua Heatmap
        4. Trả lời câu hỏi nghiệm thu (5 câu chuẩn đề bài)
        5. Trả lời câu hỏi lý thuyết 9 & 12 + định hướng mô hình hóa
    """
    peak_m = analysis_results["peak_month"]
    trough_m = analysis_results["trough_month"]
    std_init = float(analysis_results["std_first_year"])
    std_end = float(analysis_results["std_last_year"])
    ratio = float(analysis_results["std_ratio"])
    coverage = float(analysis_results["coverage_pct"])
    monthly_avg = analysis_results["monthly_avg"]
    amp = analysis_results["amplitude_by_year"]
    annual_sum = analysis_results["annual_sum"]
    yoy = analysis_results["yoy_growth_pct"]
    gfac = float(analysis_results["total_growth_factor"])
    gpct = float(analysis_results["total_growth_pct"])
    amp_growth = float(amp.iloc[-1] / amp.iloc[0]) if float(amp.iloc[0]) != 0 else float("nan")
    yoy_line = "; ".join(f"{int(ts.year)}: {val:+.1f}%" for ts, val in yoy.dropna().items())

    report_text = f"""=================================================================================
BÁO CÁO PHÂN TÍCH KẾT QUẢ BÀI TẬP 5 (TRỰC QUAN HÓA CHUYÊN SÂU CHUỖI THỜI GIAN)
Thành viên thực hiện: Bùi Trần Ngọc Khải (CV06) - Môn QLCTG (UTH), Nhóm 7
Dữ liệu: AirPassengers.csv | 144 quan sát tháng | 1949-01 -> 1960-12
=================================================================================
1. PHÂN TÍCH XU HƯỚNG VÀ DẢI BIẾN THIÊN (ROLLING STATISTICS, window=12):
   - Đường trung bình trượt 12 tháng (Rolling Mean) làm mịn hoàn toàn các dao động
     mùa vụ nội năm ngắn hạn, thể hiện rõ xu hướng tăng trưởng dốc và liên tục qua từng năm.
   - Tổng lượng hành khách hàng năm: {int(annual_sum.iloc[0])} (1949) -> {int(annual_sum.iloc[-1])} (1960),
     tương đương tăng gấp {gfac:.2f} lần (+{gpct:.1f}%). Tăng trưởng YoY: {yoy_line}.
   - Dải biên độ dao động ±2 Rolling Std mở rộng rõ rệt theo thời gian:
     + Độ lệch chuẩn nội năm 1949: {std_init:.2f}
     + Độ lệch chuẩn nội năm 1960: {std_end:.2f} (Gấp {ratio:.2f} lần so với năm 1949).
     + Biên độ mùa vụ nội năm (Max - Min): {int(amp.iloc[0])} (1949) -> {int(amp.iloc[-1])} (1960),
       gấp {amp_growth:.2f} lần - biên độ tăng tỷ lệ thuận với độ lớn của chuỗi thời gian.
   - Tỷ lệ số điểm nằm trọn trong dải ±2 Std: {coverage:.1f}% (~95%, phù hợp quy tắc thực nghiệm thống kê).
   - Kết luận phương sai: Chuỗi vi phạm giả định phương sai đồng nhất
     (Heteroskedasticity). Mô hình dạng nhân (Multiplicative) hoặc phép biến đổi
     Box-Cox / Log-transform là tối cần thiết khi xây dựng mô hình dự báo.

2. QUY LUẬT MÙA VỤ THEO THÁNG (MONTHLY SEASONAL PLOT):
   - Biểu đồ Seasonal Plot cho thấy các đường cong qua từng năm có hình dạng
     (pha dao động) đồng dạng gần như tuyệt đối, chứng minh tính mùa vụ mang
     quy luật tự nhiên rất cao và ổn định.
   - Đỉnh điểm hành khách (Peak): Tháng {peak_m} (TB {monthly_avg.loc[peak_m]:.1f} nghìn người;
     kỷ lục {int(analysis_results['series'].max())} nghìn người tại {analysis_results['series'].idxmax():%Y-%m}).
   - Đáy thấp nhất (Trough): Tháng {trough_m} (TB {monthly_avg.loc[trough_m]:.1f} nghìn người;
     thấp nhất lịch sử {int(analysis_results['series'].min())} nghìn người tại {analysis_results['series'].idxmin():%Y-%m}).
   - Mỗi năm đường cong mùa vụ tịnh tiến đều lên phía trên mà không làm đảo lộn
     cấu trúc sóng -> Xu hướng dài hạn và mùa vụ chu kỳ 12 tháng cùng song song tồn tại.

3. ĐÁNH GIÁ MẬT ĐỘ QUA BẢN ĐỒ NHIỆT (SEASONALITY HEATMAP: NĂM x THÁNG):
   - Bản đồ nhiệt thể hiện rõ sự chuyển dịch tông màu từ nhạt (giai đoạn 1949: ~100-150 nghìn khách)
     sang tông đậm (giai đoạn 1960: vượt ngưỡng 500-622 nghìn khách).
   - "Dải màu nóng" (mật độ cao nhất) tập trung liên tục vào cột Tháng 7 và Tháng 8 qua mọi năm
     (TB T7 = {monthly_avg.loc[7]:.1f}, TB T8 = {monthly_avg.loc[8]:.1f} nghìn người).
   - Cột Tháng 11 luôn là "vùng lạnh nhất" xuyên suốt 12 năm quan sát (TB {monthly_avg.loc[11]:.1f} nghìn người).
   - Sự kết hợp trực quan giữa Heatmap và Rolling Statistics cung cấp góc nhìn
     toàn diện theo cả 2 chiều: thời điểm trong năm (seasonality) và xu hướng liên năm (trend).

4. TRẢ LỜI CÂU HỎI NGHIỆM THU BÀI 5 (5 CÂU CHUẨN ĐỀ BÀI):
   Câu 1 - Line chart toàn bộ chuỗi cho thấy gì?
     -> Chuỗi tăng trưởng liên tục từ 1949 đến 1960 (tổng năm tăng gấp {gfac:.2f} lần),
        dao động hình cánh quạt mở rộng dần, có 12 đỉnh sóng mùa vụ lặp lại đều đặn mỗi năm.
   Câu 2 - Rolling Mean (window=12) và Rolling Std có vai trò gì?
     -> Rolling Mean làm mịn chu kỳ 12 tháng để làm nổi bật Trend dài hạn; Rolling Std
        và dải ±2 Std kiểm tra tính thuần nhất phương sai. Dải mở rộng dần
        (Std tăng gấp {ratio:.2f} lần) chứng minh hiện tượng phương sai thay đổi (Heteroskedasticity).
   Câu 3 - Seasonal Plot theo tháng rút ra điều gì?
     -> 12 đường cong đồng dạng, tịnh tiến đều đặn lên trên qua từng năm; đỉnh Tháng {peak_m}
        và đáy Tháng {trough_m} ổn định qua mọi năm -> Mùa vụ chu kỳ 12 tháng rất mạnh và bền vững.
   Câu 4 - Heatmap Seasonality (Month x Year) phát hiện gì?
     -> Ma trận {len(analysis_results['years'])}x12 làm nổi bật "vùng nóng" Tháng 7 - Tháng 8 và "vùng lạnh" Tháng 11,
        đồng thời thể hiện gradient tăng dần theo trục năm – bằng chứng trực quan
        cho cả Trend và Seasonality.
   Câu 5 - Kết luận: xu hướng chung, mùa vụ nổi bật, bất thường?
     -> Xu hướng: Tăng trưởng mạnh mẽ, không có điểm gãy cấu trúc lớn. Mùa vụ: Chu kỳ 12 tháng
        cực kỳ rõ nét (đỉnh du lịch hè T7 - đáy cuối thu T11). Bất thường: Không xuất hiện ngoại lai
        cực đoan nào vi phạm dải ±2 Std một cách hệ thống ({coverage:.1f}% nằm trong dải).

5. TRẢ LỜI CÂU HỎI LÝ THUYẾT 9 & 12 + ĐỊNH HƯỚNG MÔ HÌNH HÓA:
   Câu 9 - Rolling mean được dùng để kiểm tra điều gì trong dữ liệu?
     -> Rolling mean làm mượt các dao động ngắn hạn để làm nổi bật ĐƯỜNG XU HƯỚNG
        DÀI HẠN (Trend) và kiểm tra xem kỳ vọng của chuỗi có ổn định theo thời gian
        hay không (kiểm tra tính dừng về trung bình). Với AirPassengers, Rolling Mean
        12 tháng tăng đơn điệu -> Chuỗi KHÔNG DỪNG về trung bình (non-stationary mean),
        cần lấy sai phân (d >= 1) trước khi xây dựng mô hình ARIMA/SARIMA.
   Câu 12 - Vì sao trực quan hóa dữ liệu là bước quan trọng trong phân tích
     chuỗi thời gian?
     -> (a) Phát hiện tự nhiên Trend / Seasonality / gãy cấu trúc / ngoại lai mà bảng
        số liệu tĩnh khó nhận ra; (b) Lựa chọn dạng mô hình phù hợp (Additive vs Multiplicative -
        bài này rõ ràng là Multiplicative vì biên độ mùa vụ tăng gấp {amp_growth:.2f} lần);
        (c) Kiểm định các giả định mô hình (phương sai đồng nhất, phần dư phân phối chuẩn);
        (d) Truyền đạt insight nhanh chóng và trực quan cho người ra quyết định.
   Định hướng mô hình hóa:
     -> Chuỗi hội tụ đầy đủ điều kiện để mô hình hóa dự báo (Trend rõ + Seasonality chu kỳ 12 tháng
        bền vững, không phải nhiễu trắng). Đề xuất: SARIMA(p,d,q)(P,D,Q)[12] trên
        dữ liệu Log-transform, Prophet với chế độ seasonality_mode='multiplicative',
        hoặc XGBoost / LightGBM với bộ đặc trưng trễ (lag 1..12, rolling mean/std, tháng, năm).
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

    Mặc định lưu 3 biểu đồ PNG vào `LAB2/bai_tap_5/figures/` để nộp kèm báo cáo
    kỹ thuật (lựa chọn B của thành viên CV06).
    Truyền cờ `--no-figures` khi chạy nếu muốn giữ chế độ tinh gọn repo cũ
    (không lưu ảnh ra đĩa).
    """
    print("=" * 80)
    print("KHỞI CHẠY BÀI TẬP 5: TRỰC QUAN HÓA CHUYÊN SÂU CHUỖI THỜI GIAN")
    print("=" * 80)

    save_figures = "--no-figures" not in sys.argv[1:]

    current_dir = Path(__file__).resolve().parent
    data_path = current_dir.parent / "data" / "raw" / "AirPassengers.csv"
    report_file = current_dir / "ket_qua_bai_5.txt"
    figures_dir = current_dir / "figures" if save_figures else None

    # Bước 1
    series = step1_load_data(data_path)

    # Bước 2
    analysis_results = step2_execute_analysis(series, window=12)

    # Bước 3
    step3_visualize(series, analysis_results, figures_dir=figures_dir)

    # Bước 4
    print("\n[BƯỚC 4] TỔNG HỢP KẾT QUẢ VÀ TRẢ LỜI CÂU HỎI NGHIỆM THU + LÝ THUYẾT 9, 12:\n")
    report = step4_conclude_and_report(analysis_results, report_file)
    print(report)
    print(f" -> Đã lưu báo cáo phân tích tại: {report_file}")
    if figures_dir is not None:
        print(f" -> Đã lưu 3 biểu đồ PNG tại thư mục: {figures_dir}")
        print("    (bai5_1_rolling_statistics.png, bai5_2_seasonal_plot.png, bai5_3_seasonality_heatmap.png)")
    print("\n[HOÀN THÀNH BÀI TẬP 5 THÀNH CÔNG]")


if __name__ == "__main__":
    main()
