# irrigation_web

# Smart Irrigation Need Prediction System

本專案為機器學習概論期末專題，主題為智慧灌溉需求預測。  
系統使用 Kaggle 農業灌溉資料集，透過資料前處理、機器學習模型訓練、神經網路模型訓練與 Flask 網頁整合，讓使用者可以輸入農業相關參數，並選擇不同模型預測灌溉需求等級。

---

## 1. 專案目標

本專案希望建立一個網頁式智慧灌溉需求預測系統，使用者可以輸入土壤、氣候、作物與灌溉條件，系統會根據所選模型預測灌溉需求：

- Low：低灌溉需求
- Medium：中等灌溉需求
- High：高灌溉需求

目前系統已整合：

- 資料前處理流程說明
- 非神經網路模型比較
- 神經網路模型結果展示
- 使用者互動式預測頁面
- 多模型選擇功能
- Flask 後端模型預測功能

---

## 2. 使用技術

### Web

- Flask
- HTML
- CSS
- JavaScript

### Data Processing

- pandas
- numpy
- scikit-learn
- joblib

### Machine Learning

- Logistic Regression
- Random Forest
- Random Forest New
- XGBoost
- HistGradientBoosting

### Deep Learning

- TensorFlow
- Keras
- MLP, Multi-Layer Perceptron

---

## 3. 專案資料夾結構

```text
IRRIGATION_WEB/
│
├── app.py
├── make_preprocessing_files.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── scaler.pkl
│   ├── feature_columns.json
│   ├── num_cols.json
│   └── rf_top10_features.json
│
├── models/
│   ├── Lrmodel.pkl
│   ├── Rfmodel.pkl
│   ├── Rfmodelnew.pkl
│   ├── Xgbmodel.pkl
│   ├── Hgbcmodel.pkl
│   └── best_irrigation_tf_model.keras
│
├── templates/
│   ├── index.html
│   ├── preprocessing.html
│   ├── sklearn_models.html
│   ├── neural_network.html
│   └── predict.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   │
│   ├── js/
│   │   └── predict.js
│   │
│   └── images/
│       ├── scikit_learn/
│       └── tensorflow/
│
└── reference/