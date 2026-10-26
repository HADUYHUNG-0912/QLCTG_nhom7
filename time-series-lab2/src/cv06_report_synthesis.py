# -*- coding: utf-8 -*-
"""
===================================================================================
DỰ ÁN: QUẢN LÝ & KHAI PHÁ CHUỖI THỜI GIAN (QLCTG) - LAB 3: MACHINE LEARNING & XGBOOST
NHIỆM VỤ CV06: TỔNG HỢP PIPELINE, TRẢ LỜI CÂU HỎI LÝ LUẬN & XUẤT BÁO CÁO TỔNG KẾT
===================================================================================

[THÔNG TIN DÀNH CHO THÀNH VIÊN & AI AGENT]:
- Mã công việc: CV06
- Thành viên phụ trách: Bùi Trần Ngọc Khải
- Nhánh Git khuyến nghị: feat/cv06-report-theory
- Tệp thực thi: src/cv06_report_synthesis.py
- Dữ liệu đầu vào:
  + Tất cả các bảng chỉ số tại: outputs/tables/*.csv
  + Tất cả các hình ảnh trực quan tại: outputs/figures/*.png
- Sản phẩm đầu ra:
  + docs/BAO_CAO_LAB3_NHOM7.md (Báo cáo kỹ thuật toàn diện nộp Giảng viên)
  + outputs/tables/final_model_comparison.csv (Bảng so sánh tổng hợp tất cả các kịch bản)
  + outputs/figures/cv06_summary_dashboard.png (Bảng điều khiển tổng hợp kết quả đồ thị)

[QUY TẮC KỸ THUẬT BẮT BUỘC CHO AI AGENT]:
1. KIỂM TOÁN TÍNH ĐẦY ĐỦ CỦA DỰ ÁN (PROJECT READINESS AUDIT):
   - Tự động kiểm tra sự tồn tại của các artifacts từ CV01 đến CV05 trước khi tổng hợp.
2. TRẢ LỜI ĐẦY ĐỦ VÀ CHẶT CHẼ 3 CÂU HỎI LÝ LUẬN CỦA ĐỀ BÀI:
   - Câu 1: So sánh XGBoost vs Linear Regression trong bài toán chuỗi thời gian:
     + Quan hệ phi tuyến & tương tác bậc cao giữa các lags.
     + Hiện tượng đa cộng tuyến (Multicollinearity).
     + Khả năng ngoại suy xu hướng ngoài miền dữ liệu huấn luyện.
   - Câu 2: Tác động khi tăng số lượng bậc trễ lags p:
     + Lợi ích tiếp nhận tín hiệu dài hạn (chu kỳ ngày 24h, tuần 168h).
     + Rủi ro quá khớp (Overfitting), chiều dữ liệu và mất mát quan sát đầu chuỗi.
   - Câu 3: Khi nào XGBoost phù hợp hơn ARIMA/SARIMA:
     + Chuỗi phi tuyến, cấu trúc mùa vụ đa chu kỳ (Intra-day, Weekly, Annual).
     + Có nhiều biến ngoại sinh và dữ liệu kích thước lớn.
3. TỰ ĐỘNG XUẤT BÁO CÁO MARKDOWN ĐẦY ĐỦ ĐỒ THỊ VÀ BẢNG SỐ LIỆU.
===================================================================================
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
# 1. CẤU HÌNH ĐƯỜNG DẪN HỆ THỐNG (ĐỒNG BỘ CHO MỌI AGENT & THÀNH VIÊN)
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
def step2_consolidate_comparison_table() -> pd.DataFrame:
    """Hợp nhất các bảng kết quả thực nghiệm hiện có."""
    print("[CV06 - BƯỚC 2] Hợp nhất bảng số liệu...")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    final_table_path = TABLES_DIR / "final_model_comparison.csv"

    input_files = [
        "cv03_evaluation_metrics.csv",
        "cv04_multistep_metrics.csv",
        "cv05_benchmark_summary.csv",
    ]

    frames = []

    for filename in input_files:
        path = TABLES_DIR / filename

        if not path.exists():
            print(f"  [THIẾU] {filename}")
            continue

        try:
            df = pd.read_csv(path)
            if df.empty:
                print(f"  [CẢNH BÁO] {filename} không có dữ liệu.")
                continue

            df.insert(0, "source_file", filename)
            frames.append(df)
            print(f"  [OK] Đã đọc {filename}: {len(df)} dòng")

        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            print(f"  [LỖI] Không đọc được {filename}: {exc}")

    if frames:
        df_final = pd.concat(frames, ignore_index=True, sort=False)
    else:
        df_final = pd.DataFrame(
            columns=["source_file", "status", "message"]
        )
        df_final.loc[0] = [
            "",
            "incomplete",
            "Chưa có bảng kết quả CV03-CV05 để tổng hợp."
        ]

    df_final.to_csv(final_table_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Đã xuất bảng tổng hợp: {final_table_path}")

    return df_final

# ---------------------------------------------------------------------------------
# 3.5. KIỂM TOÁN CÁC ĐẦU RA CỦA NHÓM
# ---------------------------------------------------------------------------------
def step1_audit_team_deliverables() -> Dict[str, List[str]]:
    """Kiểm tra sự tồn tại của các bảng và biểu đồ CV01-CV05."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    found_tables = [f for f in EXPECTED_TABLES if (TABLES_DIR / f).exists()]
    missing_tables = [f for f in EXPECTED_TABLES if not (TABLES_DIR / f).exists()]
    found_figs = [f for f in EXPECTED_FIGURES if (FIGURES_DIR / f).exists()]
    missing_figs = [f for f in EXPECTED_FIGURES if not (FIGURES_DIR / f).exists()]

    print(f"[CV06] Bảng có sẵn: {len(found_tables)}/{len(EXPECTED_TABLES)}")
    print(f"[CV06] Biểu đồ có sẵn: {len(found_figs)}/{len(EXPECTED_FIGURES)}")

    return {
        "found_tables": found_tables,
        "missing_tables": missing_tables,
        "found_figs": found_figs,
        "missing_figs": missing_figs,
    }


# ---------------------------------------------------------------------------------
# 4. TẠO BÁO CÁO MARKDOWN
# ---------------------------------------------------------------------------------
def step3_compile_final_report(
    audit_info: Dict[str, List[str]],
    report_path: Path,
    df_final: pd.DataFrame,
) -> Path:
    """Xuất báo cáo lý thuyết và trạng thái pipeline."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    table_markdown = (
        df_final.to_markdown(index=False)
        if not df_final.empty
        else "_Chưa có dữ liệu tổng hợp._"
    )

    report = f"""# BÁO CÁO TỔNG KẾT LAB 3 – NHÓM 7

## 1. Mục tiêu
CV06 kiểm toán đầu ra CV01-CV05, tổng hợp các bảng kết quả và xây dựng báo cáo kỹ thuật cho bài toán dự báo chuỗi thời gian bằng Machine Learning và XGBoost.

## 2. Kiểm toán pipeline

### 2.1. Bảng dữ liệu
- Đã tìm thấy: {len(audit_info['found_tables'])}/{len(EXPECTED_TABLES)}
- Thiếu: {', '.join(audit_info['missing_tables']) if audit_info['missing_tables'] else 'Không'}

### 2.2. Biểu đồ
- Đã tìm thấy: {len(audit_info['found_figs'])}/{len(EXPECTED_FIGURES)}
- Thiếu: {', '.join(audit_info['missing_figs']) if audit_info['missing_figs'] else 'Không'}

Các tệp được đánh dấu thiếu cần được tạo từ công đoạn phụ trách tương ứng. Không suy diễn kết quả khi chưa có dữ liệu thực nghiệm.

## 3. Câu hỏi lý thuyết

### 3.1. XGBoost và Linear Regression

Linear Regression mô hình hóa mối quan hệ tuyến tính giữa các biến đầu vào và biến mục tiêu. Khi sử dụng các giá trị trễ (lags), mô hình dễ diễn giải và thường có chi phí tính toán thấp. Tuy nhiên, mô hình không tự biểu diễn tốt các quan hệ phi tuyến nếu không bổ sung đặc trưng phù hợp.

XGBoost kết hợp nhiều cây quyết định theo phương pháp boosting, có thể học các quan hệ phi tuyến và tương tác giữa các lags. Điều này có lợi khi hành vi chuỗi phức tạp. Tuy vậy, XGBoost không mặc nhiên tốt hơn Linear Regression; cần so sánh trên cùng tập kiểm tra và cùng chỉ số đánh giá.

Các lags thường tương quan với nhau, gây đa cộng tuyến trong Linear Regression. Điều này có thể làm hệ số hồi quy thiếu ổn định, dù khả năng dự báo không nhất thiết luôn giảm. XGBoost ít nhạy với đa cộng tuyến theo cách của hồi quy tuyến tính, nhưng các đặc trưng dư thừa vẫn có thể làm tăng độ phức tạp.

Về ngoại suy, Linear Regression có thể tiếp tục xu hướng tuyến tính theo phương trình đã học, nhưng không đảm bảo xu hướng đó đúng ngoài miền dữ liệu. XGBoost dựa trên cây thường dự báo theo các giá trị lá đã học và không giỏi ngoại suy xu hướng liên tục. Vì vậy, cần cân nhắc đặc điểm chuỗi và đánh giá ngoài mẫu.

### 3.2. Tác động của việc tăng số lượng lags p

Tăng số lượng lags giúp mô hình tiếp cận lịch sử dài hơn. Với dữ liệu theo giờ, lag 24 có thể cung cấp thông tin từ cùng giờ ngày trước, còn lag 168 có thể phản ánh cùng giờ của tuần trước. Những đặc trưng này hữu ích khi dữ liệu có chu kỳ ngày hoặc tuần.

Tuy nhiên, nhiều lags làm tăng số đặc trưng, chi phí tính toán và nguy cơ quá khớp. Việc tạo lags cũng khiến các quan sát đầu chuỗi không đủ lịch sử để sử dụng, làm giảm số mẫu huấn luyện. Do đó, cần thử nghiệm các cấu hình lags khác nhau trên cùng cách chia dữ liệu theo thời gian và chọn cấu hình dựa trên kết quả kiểm tra.

### 3.3. Khi nào XGBoost phù hợp hơn ARIMA/SARIMA?

ARIMA/SARIMA phù hợp với nhiều chuỗi thời gian có cấu trúc tự hồi quy và mùa vụ có thể mô hình hóa bằng phương pháp thống kê. XGBoost đáng cân nhắc khi dữ liệu có quan hệ phi tuyến, nhiều biến ngoại sinh hoặc tương tác phức tạp giữa các đặc trưng.

XGBoost có thể sử dụng các đặc trưng cho nhiều chu kỳ như trong ngày, theo tuần và theo năm, nhưng các chu kỳ này cần được mã hóa thành lags hoặc đặc trưng thời gian. Với dữ liệu lớn và nhiều biến đầu vào, XGBoost có thể khai thác các quan hệ phức tạp; đổi lại, nó cần thiết kế đặc trưng và kiểm soát rò rỉ dữ liệu.

Không thể kết luận XGBoost luôn tốt hơn ARIMA/SARIMA nếu chưa có kết quả so sánh trên cùng tập kiểm tra. Lựa chọn cuối cùng phải dựa trên sai số dự báo và yêu cầu thực tế.

## 4. Bảng kết quả tổng hợp

{table_markdown}

## 5. Kết luận

CV06 cung cấp bước kiểm tra tính đầy đủ của pipeline và hợp nhất các bảng kết quả hiện có. Kết luận về mô hình tốt nhất chỉ nên đưa ra sau khi các công đoạn trước hoàn tất và dữ liệu đánh giá được kiểm chứng. Những tệp còn thiếu cần được bổ sung trước khi xác nhận toàn bộ pipeline đã sẵn sàng.

## 6. Hình tổng hợp
Biểu đồ tổng hợp sẽ được tạo nếu có dữ liệu đầu vào phù hợp và thư viện trực quan hóa được cài đặt.
"""

    report_path.write_text(report, encoding="utf-8")
    print(f"[CV06] Đã xuất báo cáo: {report_path}")
    return report_path


# ---------------------------------------------------------------------------------
# 5. TẠO BIỂU ĐỒ TỔNG HỢP
# ---------------------------------------------------------------------------------
def step4_create_dashboard(audit_info: Dict[str, List[str]]) -> None:
    """Tạo biểu đồ trạng thái đầu ra CV01-CV05."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        labels = ["Tables", "Figures"]
        completed = [
            len(audit_info["found_tables"]),
            len(audit_info["found_figs"]),
        ]
        total = [len(EXPECTED_TABLES), len(EXPECTED_FIGURES)]

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(labels))
        ax.bar(x, completed, label="Available")
        ax.bar(x, np.array(total) - np.array(completed),
               bottom=completed, label="Missing")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Number of artifacts")
        ax.set_title("CV06 Project Readiness Dashboard")
        ax.legend()
        fig.tight_layout()

        output_path = FIGURES_DIR / "cv06_summary_dashboard.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"[CV06] Đã xuất biểu đồ: {output_path}")

    except ImportError:
        print("[CV06] Chưa có matplotlib; bỏ qua việc tạo dashboard.")


# ---------------------------------------------------------------------------------
# 6. CHƯƠNG TRÌNH CHÍNH
# ---------------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("CV06 - TỔNG HỢP PIPELINE VÀ BÁO CÁO LAB 3")
    print("=" * 70)

    audit_info = step1_audit_team_deliverables()
    df_final = step2_consolidate_comparison_table()

    report_path = DOCS_DIR / "BAO_CAO_LAB3_NHOM7.md"
    step3_compile_final_report(audit_info, report_path, df_final)
    step4_create_dashboard(audit_info)

    print("[CV06] Hoàn tất tiến trình tạo báo cáo.")


if __name__ == "__main__":
    main()

