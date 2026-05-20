import json
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# =========================
# 讀取原始資料
# =========================
df = pd.read_csv("data/train.csv")

# =========================
# 刪除無用欄位
# =========================
df = df.drop(columns=["id"])

# =========================
# Mulching_Used 轉成 1 / 0
# =========================
le = LabelEncoder()
df["Mulching_Used"] = le.fit_transform(df["Mulching_Used"])

# =========================
# Irrigation_Need 轉成 0 / 1 / 2
# =========================
irrigation_map = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

df["Irrigation_Need"] = df["Irrigation_Need"].map(irrigation_map)

# =========================
# One-Hot Encoding 類別欄位
# =========================
df = pd.get_dummies(
    df,
    columns=[
        "Soil_Type",
        "Crop_Type",
        "Season",
        "Irrigation_Type",
        "Water_Source",
        "Region",
        "Crop_Growth_Stage"
    ]
)

# =========================
# True / False 轉成 1 / 0
# =========================
bool_cols = df.select_dtypes(include=["bool"]).columns
df[bool_cols] = df[bool_cols].astype(int)

# =========================
# Irrigation_Need 移到最後一欄
# =========================
cols = [c for c in df.columns if c != "Irrigation_Need"]
cols.append("Irrigation_Need")
df = df[cols]

# =========================
# 數值欄位標準化
# =========================
num_cols = [
    "Soil_pH",
    "Soil_Moisture",
    "Organic_Carbon",
    "Electrical_Conductivity",
    "Temperature_C",
    "Humidity",
    "Rainfall_mm",
    "Sunlight_Hours",
    "Wind_Speed_kmh",
    "Field_Area_hectare",
    "Previous_Irrigation_mm"
]

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# =========================
# 儲存 scaler
# =========================
joblib.dump(scaler, "data/scaler.pkl")

# =========================
# 儲存數值欄位名稱
# =========================
with open("data/num_cols.json", "w", encoding="utf-8") as f:
    json.dump(num_cols, f, ensure_ascii=False, indent=4)

# =========================
# 儲存模型輸入欄位順序
# =========================
feature_columns = [c for c in df.columns if c != "Irrigation_Need"]

with open("data/feature_columns.json", "w", encoding="utf-8") as f:
    json.dump(feature_columns, f, ensure_ascii=False, indent=4)

# =========================
# 輸出 clean_train.csv
# =========================
df.to_csv("data/clean_train.csv", index=False)

print("前處理檔案產生完成！")
print("已產生：")
print("- data/clean_train.csv")
print("- data/scaler.pkl")
print("- data/feature_columns.json")
print("- data/num_cols.json")