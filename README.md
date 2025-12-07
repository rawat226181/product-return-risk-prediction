# 🛒 Product Return Risk Prediction — Retail (Hackathon)

**Project goal:** build a binary classification system to predict whether an order (or order line) will be returned/cancelled, using features available at order time.  
This repo contains EDA, feature engineering, model training, hyperparameter tuning, final model artifacts, and a Streamlit demo for predictions.

---

## 🚩 Quick links
- Dataset used: **Synthetic Dataset for E-Commerce Return Analysis** (Kaggle) — recommended.  
  (https://www.kaggle.com/datasets/sayalikhot21/synthetic-dataset-for-e-commerce-return-analysis)  
- Demo (local): ` `

---

## 📁 Project structure

product-return-risk/ ← repo root
│
├── data/
│ ├── raw/
│ │ └── <raw csv files from dataset>
│ └── processed/
│ └── processed_returns.csv
│
├── notebooks/
│ ├── 1_EDA.ipynb
│ ├── 2_Feature_Engineering.ipynb
│ ├── 3_Model_Training.ipynb
│ ├── 4_Model_Comparison.ipynb
│ ├── 5_Hyperparameter_Tuning.ipynb
│ ├── 6_Final_Model_Evaluation.ipynb
│ └── 8_Prediction_Pipeline_Test.ipynb
│
├── models/
│ ├── final_model.pkl
│ ├── scaler.pkl
│ ├── feature_columns.json
│ └── model_info.json
│
├── results/
│ ├── model_comparison.json
│ ├── hyperparameter_tuning_results.json
│ └── final_model_evaluation.json
│
├── app/
│ ├── prediction_pipeline.py
│ ├── pridict.py ← (script for CLI testing)
│ └── streamlit_app.py ← demo UI
│
├── requirements.txt
└── README.md



🔎 Key features 
1-product_id, product_category_name
2-price, payment_type, payment_installments
3-customer_id, customer_past_returns, customer_total_orders
4-order_purchase_timestamp, order_weekday, order_hour
5-shipping_limit_date (expected shipping window)
6-order_channel (web/mobile)

ARCHITECHURE PART-
                ┌────────────────────────┐
                │  Raw Retail Dataset     │
                └──────────┬─────────────┘
                           ▼
                ┌────────────────────────┐
                │  Data Cleaning & EDA    │
                │  - Missing values       │
                │  - Class balance        │
                └──────────┬─────────────┘
                           ▼
                ┌────────────────────────┐
                │ Feature Engineering     │
                │ - Encoding              │
                │ - Normalization         │
                │ - Customer history      │
                └──────────┬─────────────┘
                           ▼
                ┌────────────────────────┐
                │ Model Training          │
            │ Logistic / decision-tree / XGB     │
                └──────────┬─────────────┘
                           ▼
                ┌────────────────────────┐
                │ Model Evaluation        │
                │ AUC, F1, Recall         │
                └──────────┬─────────────┘
                           ▼
                ┌────────────────────────┐
                │ Save Final Model        │
                │ export .pkl + schema    │
                └──────────┬─────────────┘
                           ▼
                ┌────────────────────────┐
                │  UI / Endpoint (Sprint 3)│
                │- User enters order details │
                │- Get return-risk score │
                └──────────────────────────┘


🧭 Approach & pipeline
EDA & label construction (notebooks/1_EDA.ipynb)
Understand missingness, class balance, basic feature distributions.
Feature engineering (notebooks/2_Feature_Engineering.ipynb)
Encode categoricals, create customer history features, normalize numeric features.
Baseline + Model training (notebooks/3_Model_Training.ipynb)
Baseline: Logistic Regression
Tree-based: Random Forest, Gradient Boosting (scikit-learn)
Model comparison & tuning (notebooks/4_Model_Comparison.ipynb, 5_Hyperparameter_Tuning.ipynb)
RandomizedSearchCV for RF & GB; compare AUC, F1, Precision, Recall.
Final evaluation & packaging (notebooks/6_Final_Model_Evaluation.ipynb, 7_Save_Final_Model.ipynb)
Save final_model.pkl, scaler.pkl, and feature_columns.json.
Prediction pipeline & demo (app/pridict.py, app/prediction_pipeline.py, app/streamlit_app.py)



We evaluate with:
ROC AUC (primary: good for imbalanced classes)
Precision / Recall / F1 (report all)
Confusion Matrix
Class distribution / baseline model (random or majority class)


UI DESIGN PART:

┌──────────────────────────────────────────────────────┐
│              🛍️ Retail Return Risk App               │
│──────────────────────────────────────────────────────│
│  🔡 Enter Order Details                              │
│   • Product Category [Dropdown]                       │
│   • Price [Input]                                     │
│   • Quantity [Input]                                  │
│   • Payment Type [Dropdown]                           │
│   • Installments [Input]                              │
│   • Order Channel [Dropdown]                          │
│   • Freight Value [Input]                             │
│   • Customer Region [Dropdown]                        │
│   • Purchase Date [Date Picker]                       │
│                                                      │
│        [ Predict Return Risk ]                        │
│──────────────────────────────────────────────────────│
│  📊 Prediction Results                                │
│   • Probability of Return: 0.78                       │
│   • Predicted Class: RETURNED (1)                     │
│   • Risk Level: 🔴 High Risk                          │
│                                                      │
└──────────────────────────────────────────────────────┘

