from flask import Flask, render_template, request
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# 讀取前處理需要的檔案
scaler = joblib.load(DATA_DIR / "scaler.pkl")

with open(DATA_DIR / "feature_columns.json", "r", encoding="utf-8") as f:
    feature_columns = json.load(f)

with open(DATA_DIR / "num_cols.json", "r", encoding="utf-8") as f:
    num_cols = json.load(f)

with open(DATA_DIR / "rf_top10_features.json", "r", encoding="utf-8") as f:
    rf_top10_features = json.load(f)


# 載入 Logistic Regression
lr_model = joblib.load(MODEL_DIR / "Lrmodel.pkl")

# 修正不同 scikit-learn 版本造成的 multi_class 屬性問題
if not hasattr(lr_model, "multi_class"):
    lr_model.multi_class = "auto"

# 載入其他模型
sklearn_models = {
    "lr": lr_model,
    "rf": joblib.load(MODEL_DIR / "Rfmodel.pkl"),
    "rf_new": joblib.load(MODEL_DIR / "Rfmodelnew.pkl"),
    "xgb": joblib.load(MODEL_DIR / "Xgbmodel.pkl"),
    "hgbc": joblib.load(MODEL_DIR / "Hgbcmodel.pkl"),
}

model_display_names = {
    "lr": "Logistic Regression",
    "rf": "Random Forest",
    "rf_new": "Random Forest New",
    "xgb": "XGBoost",
    "hgbc": "HistGradientBoosting",
    "nn": "Keras / TensorFlow MLP",
}

label_map = {
    0: "Low",
    1: "Medium",
    2: "High",
}


def build_input_dataframe(form_data):
    """
    將使用者在網頁輸入的原始農業參數，
    轉換成模型訓練時使用的 clean_train.csv 特徵格式。
    """

    # 建立一筆全 0 的資料，欄位順序完全依照 feature_columns.json
    input_df = pd.DataFrame(0, index=[0], columns=feature_columns)

    # 1. 數值欄位：先收集原始數值，再用 scaler 標準化
    raw_num_data = {}

    for col in num_cols:
        raw_num_data[col] = float(form_data.get(col))

    raw_num_df = pd.DataFrame([raw_num_data], columns=num_cols)
    scaled_values = scaler.transform(raw_num_df)

    input_df.loc[0, num_cols] = scaled_values[0]

    # 2. Mulching_Used：Yes / No 轉 1 / 0
    mulching_value = form_data.get("Mulching_Used")
    input_df.loc[0, "Mulching_Used"] = 1 if mulching_value == "Yes" else 0

    # 3. 類別欄位：手動轉成 One-Hot Encoding
    categorical_fields = [
        "Soil_Type",
        "Crop_Type",
        "Season",
        "Irrigation_Type",
        "Water_Source",
        "Region",
        "Crop_Growth_Stage",
    ]

    for field in categorical_fields:
        value = form_data.get(field)
        one_hot_col = f"{field}_{value}"

        if one_hot_col in input_df.columns:
            input_df.loc[0, one_hot_col] = 1

    return input_df


def predict_with_sklearn_model(model_type, input_df):
    """
    根據使用者選擇的非神經網路模型進行預測。
    """

    model = sklearn_models[model_type]

    # Random Forest New 只使用前 10 個特徵
    if model_type == "rf_new":
        model_input = input_df[rf_top10_features]
    else:
        # 其他非神經網路模型使用完整 42 個特徵
        model_input = input_df[feature_columns]

    pred = model.predict(model_input)[0]
    pred = int(pred)

    result = label_map[pred]

    probability_text = ""

    # 如果模型支援 predict_proba，就顯示三類機率
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(model_input)[0]
        probability_text = (
            f"Low：{probs[0] * 100:.2f}%｜"
            f"Medium：{probs[1] * 100:.2f}%｜"
            f"High：{probs[2] * 100:.2f}%"
        )

    return result, probability_text


def get_result_explanation(result):
    """
    根據預測結果給使用者簡單解釋。
    """

    if result == "Low":
        return "預測結果為 Low，代表此條件下灌溉需求較低。"
    elif result == "Medium":
        return "預測結果為 Medium，代表此條件下灌溉需求為中等。"
    elif result == "High":
        return "預測結果為 High，代表此條件下灌溉需求較高。"
    else:
        return "無法判斷預測結果。"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/preprocessing")
def preprocessing():
    return render_template("preprocessing.html")


@app.route("/sklearn-models")
def sklearn_models_page():
    return render_template("sklearn_models.html")


@app.route("/neural-network")
def neural_network():
    return render_template("neural_network.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        model_group = request.form.get("model_group")
        model_type = request.form.get("model_type")

        try:
            input_df = build_input_dataframe(request.form)

            if model_group == "sklearn":
                result, probability_text = predict_with_sklearn_model(model_type, input_df)

                explanation = get_result_explanation(result)
                selected_model_name = model_display_names.get(model_type, model_type)

                return render_template(
                    "predict.html",
                    result=result,
                    explanation=explanation,
                    probability_text=probability_text,
                    selected_model_name=selected_model_name,
                )

            elif model_group == "tensorflow":
                result = "尚未串接"
                explanation = "神經網路模型尚未正式串接，下一步會加入 Keras / TensorFlow MLP 預測功能。"
                selected_model_name = "Keras / TensorFlow MLP"

                return render_template(
                    "predict.html",
                    result=result,
                    explanation=explanation,
                    probability_text="",
                    selected_model_name=selected_model_name,
                )

        except Exception as e:
            return render_template(
                "predict.html",
                result="預測失敗",
                explanation=f"錯誤訊息：{str(e)}",
                probability_text="",
                selected_model_name="",
            )

    return render_template("predict.html")


if __name__ == "__main__":
    app.run(debug=True)