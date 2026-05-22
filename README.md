# Smart Irrigation Need Prediction System

本專案為機器學習概論期末專題，主題為智慧灌溉需求預測。  
系統使用 Kaggle 農業灌溉資料集，透過資料前處理、非神經網路模型訓練、神經網路模型訓練、模型診斷分析與 Flask 網頁整合，建立一套可互動操作的灌溉需求預測系統。

使用者可在網頁中輸入農業相關參數，並選擇不同模型進行灌溉需求預測，系統會輸出 Low、Medium、High 三種灌溉需求等級與模型預測機率。

---

## 0.其他組員本地端執行前需要準備的檔案

由於部分資料檔與模型檔案較大，或不適合直接上傳至 GitHub，因此其他組員從 GitHub clone 專案後，需另外手動放入以下檔案，才能完整執行網頁與模型預測功能。

---

### 需要自行準備的大型資料檔

請將原始資料放入 `data/` 資料夾：
data/train.csv

## 1. 專案目標

本專案希望建立一個網頁式智慧灌溉需求預測系統，讓使用者可以輸入土壤、氣候、作物、灌溉方式與地區等農業條件，並透過機器學習模型預測灌溉需求等級。

預測結果分為三類：

- Low：低灌溉需求
- Medium：中等灌溉需求
- High：高灌溉需求

目前系統已完成：

- 資料前處理流程說明
- 非神經網路模型訓練與比較
- 神經網路模型訓練與結果展示
- 模型診斷與多數決預測分析
- 使用者互動式預測頁面
- 多模型選擇功能
- Flask 後端模型即時預測功能
- 多數決預測下載功能

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

### Model Diagnostics

- Overfitting Gap Analysis
- Prediction Confidence / Residual Distribution
- ROC / AUC Visualization
- Model Prediction Consistency
- Majority Voting

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
│   ├── diagnostics.html
│   └── predict.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   │
│   ├── js/
│   │   └── predict.js
│   │
│   ├── images/
│   │   ├── scikit_learn/
│   │   ├── tensorflow/
│   │   └── diagnostics/
│   │       ├── tensorflow_metrics.png
│   │       ├── sklearn_overfitting.png
│   │       ├── tensorflow_residual_roc.png
│   │       └── sklearn_residual_roc.png
│   │
│   └── files/
│       └── final_predictions_voting.csv
│


## Keras / TensorFlow 神經網路模型相容性說明

本專案的神經網路模型使用 Keras / TensorFlow 儲存。由於不同版本的 Keras / TensorFlow 對模型設定參數的支援不同，原始模型檔：
models/best_irrigation_tf_model.keras