# -*- coding: utf-8 -*-
"""CV06 — Bùi Trần Ngọc Khải: Tổng hợp Pipeline, trả lời câu hỏi lý luận & xuất báo cáo tổng kết.

Đầu vào : outputs/tables/*.csv, outputs/figures/*.png (từ CV01 đến CV05)
Đầu ra  : docs/BAO_CAO_LAB3_NHOM7.md
          outputs/tables/final_model_comparison.csv
          outputs/figures/cv06_summary_dashboard.png
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List

# Cấu hình encoding UTF-8 cho console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------
# 1. CẤU HÌNH ĐƯỜNG DẪN HỆ THỐNG
# ---------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
DOCS_DIR = PROJECT_ROOT / "docs"

DOCS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_TABLES = [
    "cv01_data_summary.csv",
    "cv02_train_history.csv",
    "cv03_evaluation_metrics.csv",
    "cv03_feature_importance.csv",
    "cv04_multistep_metrics.csv",
    "cv05_best_hyperparameters.csv",
    "cv05_benchmark_summary.csv"
]

EXPECTED_FIGURES = [
    "cv01_raw_and_lags.png",
    "cv02_learning_curve.png",
    "cv03_actual_vs_predicted.png",
    "cv03_feature_importance.png",
    "cv04_rolling_features_impact.png",
    "cv04_multistep_forecast.png",
    "cv05_benchmark_comparison.png"
]


# ---------------------------------------------------------------------------------
# 2. KIỂM TOÁN TIẾN ĐỘ THỰC HIỆN CỦA CÁC THÀNH VIÊN
# ---------------------------------------------------------------------------------
def step1_audit_team_deliverables() -> Dict[str, List[str]]:
    """Kiểm tra các bảng số liệu và đồ thị trong outputs/."""
    print("[CV06 - BƯỚC 1] Đang kiểm toán tiến độ hoàn thành các đầu ra của nhóm...")
    found_tables = [t for t in EXPECTED_TABLES if (TABLES_DIR / t).exists()]
    missing_tables = [t for t in EXPECTED_TABLES if not (TABLES_DIR / t).exists()]
    
    found_figs = [f for f in EXPECTED_FIGURES if (FIGURES_DIR / f).exists()]
    missing_figs = [f for f in EXPECTED_FIGURES if not (FIGURES_DIR / f).exists()]
    
    print(f" -> Đã có {len(found_tables)}/{len(EXPECTED_TABLES)} bảng số liệu.")
    print(f" -> Đã có {len(found_figs)}/{len(EXPECTED_FIGURES)} biểu đồ trực quan.")
    
    return {
        "found_tables": found_tables,
        "missing_tables": missing_tables,
        "found_figs": found_figs,
        "missing_figs": missing_figs
    }


# ---------------------------------------------------------------------------------
# 3. HỢP NHẤT TOÀN BỘ CHỈ SỐ VÀO BẢNG FINAL COMPARISON
# ---------------------------------------------------------------------------------
def step2_consolidate_comparison_table() -> pd.DataFrame:
    """Hợp nhất các bảng chỉ số đánh giá và xuất file final_model_comparison.csv."""
    print("[CV06 - BƯỚC 2] Hợp nhất bảng số liệu so sánh toàn diện...")
    final_table_path = TABLES_DIR / "final_model_comparison.csv"
    # TODO: Hợp nhất bảng dữ liệu
    df_final = pd.DataFrame()
    return df_final


# ---------------------------------------------------------------------------------
# 4. SOẠN THẢO BÁO CÁO KHOA HỌC VÀ TRẢ LỜI 3 CÂU HỎI LÝ LUẬN
# ---------------------------------------------------------------------------------
def step3_compile_final_report(audit_info: Dict[str, List[str]], report_path: Path) -> Path:
    """Biên soạn báo cáo Markdown toàn diện và lưu ra docs/BAO_CAO_LAB3_NHOM7.md."""
    print(f"[CV06 - BƯỚC 3] Soạn thảo báo cáo kỹ thuật tại: {report_path}")
    # TODO: Ghi báo cáo Markdown hoàn chỉnh
    return report_path


# ---------------------------------------------------------------------------------
# HÀM THỰC THI CHÍNH (MAIN ENTRYPOINT)
# ---------------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("KHỞI CHẠY TIẾN TRÌNH CV06: TỔNG HỢP PIPELINE & XUẤT BÁO CÁO (BÙI TRẦN NGỌC KHẢI)")
    print("=" * 80)
    
    # 1. Kiểm toán tiến độ
    audit_info = step1_audit_team_deliverables()
    
    # 2. Hợp nhất bảng so sánh
    df_final = step2_consolidate_comparison_table()
    
    # 3. Soạn thảo báo cáo
    report_file = DOCS_DIR / "BAO_CAO_LAB3_NHOM7.md"
    step3_compile_final_report(audit_info, report_file)
    
    print("[CV06 HOÀN TẤT: Báo cáo tổng hợp toàn bài lab đã sẵn sàng nộp]")


if __name__ == "__main__":
    main()
