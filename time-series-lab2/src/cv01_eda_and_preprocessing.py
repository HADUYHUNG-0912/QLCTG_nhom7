# =============================================================================
# AI_AGENT_INSTRUCTIONS — ĐỌC TRƯỚC KHI THỰC HIỆN BẤT KỲ THAY ĐỔI NÀO
# =============================================================================
# MODULE     : CV01 — EDA, Tiền xử lý dữ liệu & Phân chia tập Train/Test 80/20
# PHỤ TRÁCH  : Hiệp
# MÔ TẢ TASK : Nạp a10.csv (204 tháng: 07/1991 – 06/2008), thiết lập
#              DatetimeIndex tần suất tháng (freq='MS'), kiểm toán chất lượng
#              dữ liệu (0% missing, 0 duplicate), phân tích khám phá EDA
#              (trend, mùa vụ), phân chia TUẦN TỰ 80% Train / 20% Test.
#
# DELIVERABLES (ĐẦU RA):
#   • data/processed/train.csv           – 163 tháng (1991-07 → 2005-01)
#   • data/processed/test.csv            – 41 tháng  (2005-02 → 2008-06)
#   • outputs/figures/cv01_eda_timeseries.png
#   • outputs/figures/cv01_monthly_seasonality_boxplot.png
#   • outputs/tables/cv01_data_summary.csv
#   • outputs/tables/cv01_split_info.csv
#
# RÀNG BUỘC BẮT BUỘC (HARD CONSTRAINTS):
#   [1] KHÔNG dùng shuffle=True ở bất kỳ bước nào — bảo tồn thứ tự thời gian.
#   [2] KHÔNG fit scaler / KHÔNG dùng thông tin Test để tính train stats.
#   [3] KHÔNG ghi đè data/raw/a10.csv — chỉ ĐỌC file gốc.
#   [4] Lưu hình ảnh với dpi=150 và bbox_inches='tight'.
#   [5] Ghi log print() đầy đủ để xác nhận kết quả.
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------------------------
# 0. THIẾT LẬP ĐƯỜNG DẪN GỐC (ROOT PATHS)
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA    = os.path.join(ROOT, "data", "raw",       "a10.csv")
PROC_DIR    = os.path.join(ROOT, "data", "processed")
FIG_DIR     = os.path.join(ROOT, "outputs", "figures")
TABLE_DIR   = os.path.join(ROOT, "outputs", "tables")

for d in [PROC_DIR, FIG_DIR, TABLE_DIR]:
    os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. NẠP & THIẾT LẬP DatetimeIndex
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """Đọc a10.csv, ép kiểu ngày tháng và thiết lập DatetimeIndex."""
    # TODO: pd.read_csv() → parse_dates=['date'] → set_index('date') → asfreq('MS')
    raise NotImplementedError("Thành viên 1 cài đặt tại đây")


# ---------------------------------------------------------------------------
# 2. KIỂM TOÁN CHẤT LƯỢNG DỮ LIỆU (DATA AUDIT)
# ---------------------------------------------------------------------------
def audit_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kiểm tra:
      - Số lượng bản ghi: phải đúng 204 tháng.
      - Missing values: phải = 0.
      - Duplicate rows: phải = 0.
      - Phạm vi ngày: 1991-07 → 2008-06.
    In kết quả kiểm toán và lưu outputs/tables/cv01_data_summary.csv.
    """
    # TODO: df.info(), df.isna().sum(), df.duplicated().sum(), df.describe()
    raise NotImplementedError("Thành viên 1 cài đặt tại đây")


# ---------------------------------------------------------------------------
# 3. PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA)
# ---------------------------------------------------------------------------
def plot_eda(df: pd.DataFrame) -> None:
    """
    Vẽ 2 biểu đồ:
      (a) Line plot chuỗi thời gian gốc: nhận diện Trend tăng dần
          và phương sai mở rộng → đặc trưng mô hình nhân tính (Multiplicative).
      (b) Boxplot 12 tháng trong năm → phân tích chu kỳ mùa vụ:
          Đỉnh tháng 1 (Jan), đáy tháng 2–3 (Feb–Mar).
    Lưu: outputs/figures/cv01_eda_timeseries.png
         outputs/figures/cv01_monthly_seasonality_boxplot.png
    """
    # TODO: fig, ax = plt.subplots(...)
    #       ax.plot(df.index, df['value'])
    #       ax.set_title(...); ax.set_xlabel('Năm'); ax.set_ylabel('Doanh số (AUD triệu)')
    #       fig.savefig(os.path.join(FIG_DIR, 'cv01_eda_timeseries.png'), dpi=150, bbox_inches='tight')
    raise NotImplementedError("Thành viên 1 cài đặt tại đây")


# ---------------------------------------------------------------------------
# 4. PHÂN CHIA DỮ LIỆU THEO THỜI GIAN (TEMPORAL SPLIT 80/20)
# ---------------------------------------------------------------------------
def split_train_test(df: pd.DataFrame, train_ratio: float = 0.80):
    """
    Phân chia TUẦN TỰ (không shuffle):
      Train: 80% đầu = 163 tháng (1991-07 → 2005-01)
      Test : 20% cuối = 41 tháng  (2005-02 → 2008-06)

    Args:
        df          : DataFrame đã có DatetimeIndex.
        train_ratio : Tỷ lệ tập Train (mặc định 0.80).

    Returns:
        train (pd.DataFrame), test (pd.DataFrame)
    """
    # TODO: n = len(df); train_size = int(n * train_ratio)
    #       train = df.iloc[:train_size]; test = df.iloc[train_size:]
    raise NotImplementedError("Thành viên 1 cài đặt tại đây")


# ---------------------------------------------------------------------------
# 5. LƯU TẬP TRAIN & TEST
# ---------------------------------------------------------------------------
def save_splits(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """
    Lưu:
      data/processed/train.csv
      data/processed/test.csv
      outputs/tables/cv01_split_info.csv  (thống kê phân chia)
    """
    # TODO: train.to_csv(...); test.to_csv(...)
    raise NotImplementedError("Thành viên 1 cài đặt tại đây")


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("CV01 — EDA, Tiền xử lý & Phân chia dữ liệu (Thành viên 1)")
    print("=" * 70)

    # Bước 1: Nạp dữ liệu
    df = load_data(RAW_DATA)
    print(f"[OK] Đã nạp {len(df)} quan sát từ {df.index[0].date()} đến {df.index[-1].date()}")

    # Bước 2: Kiểm toán chất lượng
    df = audit_data(df)

    # Bước 3: EDA
    plot_eda(df)
    print("[OK] Đã lưu biểu đồ EDA.")

    # Bước 4 & 5: Phân chia và lưu
    train, test = split_train_test(df)
    save_splits(train, test)
    print(f"[OK] Train: {len(train)} tháng | Test: {len(test)} tháng")
    print("[DONE] CV01 hoàn tất. Chạy CV02, CV03, CV04, CV05 song song tiếp theo.")

