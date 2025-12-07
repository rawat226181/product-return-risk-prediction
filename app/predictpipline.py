import pandas as pd
import numpy as np
import pickle
import json
import os

# ======================================================
#       RESOLVE PROJECT ROOT DIRECTORY
# ======================================================

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(ROOT, "models", "final_model.pkl")
SCALER_PATH = os.path.join(ROOT, "models", "scaler.pkl")
FEATURE_COLS_PATH = os.path.join(ROOT, "models", "feature_columns.json")

# ======================================================
#                 LOAD MODEL + SCALER
# ======================================================

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

with open(FEATURE_COLS_PATH, "r") as f:
    feature_columns = json.load(f)


# ======================================================
#           PREPROCESSING FOR INPUT DATA
# ======================================================

def preprocess_input(input_dict):
    """
    Converts raw input dictionary → scaled feature vector
    """
    df = pd.DataFrame([input_dict])

    # Ensure same column order as training
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Scale the data
    X_scaled = scaler.transform(df)

    return X_scaled


# ======================================================
#               SINGLE PREDICTION
# ======================================================

def predict_single(input_dict):
    """
    Takes a dict → returns (probability, class)
    """
    X = preprocess_input(input_dict)

    prob = model.predict_proba(X)[0][1]      # probability of class = 1
    pred = int(prob > 0.5)                   # convert to class label

    return prob, pred


# ======================================================
#               CSV BATCH PREDICTION
# ======================================================

def predict_from_csv(csv_path):
    """
    Takes CSV file path → returns DataFrame with predictions
    """
    df = pd.read_csv(csv_path)

    # Align columns
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Scale
    X_scaled = scaler.transform(df)

    probabilities = model.predict_proba(X_scaled)[:, 1]
    predictions = (probabilities > 0.5).astype(int)

    df["Return_Probability"] = probabilities
    df["Predicted_Return"] = predictions

    return df
