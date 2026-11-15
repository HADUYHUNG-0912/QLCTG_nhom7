
import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# 1. Thiết lập đường dẫn
ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
PROC_DIR = os.path.join(ROOT, "data", "processed")
TABLE_DIR = os.path.join(ROOT, "outputs", "tables")
FIG_DIR = os.path.join(ROOT, "outputs", "figures")

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# 2. Đọc dữ liệu Train
train = pd.read_csv(
    os.path.join(PROC_DIR, "train.csv"),
    index_col="timestamp",
    parse_dates=True
)["load"]

# 3. Kiểm định ADF chuỗi gốc
adf_result = adfuller(train.dropna(), autolag="AIC")

# 4. Sai phân bậc 1
train_diff = train.diff().dropna()

# 5. Kiểm định ADF sau sai phân
adf_diff_result = adfuller(train_diff, autolag="AIC")

# 6. Lưu chuỗi sau sai phân
train_diff.to_frame(name="load").to_csv(
    os.path.join(PROC_DIR, "train_differenced.csv"),
    index_label="timestamp"
)

# 7. Lưu bảng kết quả ADF
rows = [
    {
        "Chuoi": "Goc",
        "ADF_stat": adf_result[0],
        "p_value": adf_result[1],
        "is_stationary": adf_result[1] < 0.05
    },
    {
        "Chuoi": "Sai phan d=1",
        "ADF_stat": adf_diff_result[0],
        "p_value": adf_diff_result[1],
        "is_stationary": adf_diff_result[1] < 0.05
    }
]

pd.DataFrame(rows).to_csv(
    os.path.join(TABLE_DIR, "cv02_adf_test_summary.csv"),
    index=False
)

# 8. Vẽ biểu đồ
fig, axes = plt.subplots(2, 1, figsize=(12, 7))

axes[0].plot(train.index, train.values, color="blue")
axes[0].set_title("Train goc - Truoc sai phan")
axes[0].grid(True)

axes[1].plot(train_diff.index, train_diff.values, color="orange")
axes[1].set_title("Train sau sai phan d=1")
axes[1].grid(True)

plt.tight_layout()
plt.savefig(
    os.path.join(
        FIG_DIR, "cv02_adf_before_after_differencing.png"
    ),
    dpi=300
)
plt.close()

print("CV02 hoan thanh!")
print("ADF goc p-value:", adf_result[1])
print("ADF sau sai phan p-value:", adf_diff_result[1])
