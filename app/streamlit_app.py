import streamlit as st
import pandas as pd
import json
import numpy as np
import pickle
import os
from prediction_pipeline import predict_single, predict_from_csv

# ======================================================
# Streamlit Setup
# ======================================================

st.set_page_config(page_title="Product Return Risk Predictor", layout="centered")

st.title("🛒 Product Return Risk Prediction App")
st.write("Predict whether a product is likely to be returned based on customer & order features.")

# Sidebar
menu = st.sidebar.radio("Select Mode", ["Single Prediction", "Batch Prediction (CSV)"])


# ======================================================
# SINGLE PREDICTION UI
# ======================================================
if menu == "Single Prediction":

    st.subheader("🔍 Predict Return Probability for One Order")

    # User Inputs
    Product_Price = st.number_input("Product Price", min_value=1.0, max_value=50000.0, value=500.0)
    Order_Quantity = st.number_input("Order Quantity", 1, 50, 1)
    Days_to_Return = st.number_input("Days to Return", 0, 60, 10)
    User_Age = st.number_input("User Age", 15, 80, 30)
    Discount_Applied = st.slider("Discount Applied", 0.0, 1.0, 0.10)
    Order_Year = st.number_input("Order Year", 2019, 2025, 2023)
    Order_Month = st.number_input("Order Month", 1, 12, 7)
    Order_DayOfWeek = st.number_input("Order Day of Week (0 = Monday)", 0, 6, 3)
    Order_Day = st.number_input("Order Day", 1, 31, 15)

    # Additional derived features
    High_Discount = int(Discount_Applied > 0.3)
    High_Price = int(Product_Price > 1000)
    Bulk_Order = int(Order_Quantity > 3)

    sample = {
        "Product_Price": Product_Price,
        "Order_Quantity": Order_Quantity,
        "Days_to_Return": Days_to_Return,
        "User_Age": User_Age,
        "Discount_Applied": Discount_Applied,
        "Order_Year": Order_Year,
        "Order_Month": Order_Month,
        "Order_DayOfWeek": Order_DayOfWeek,
        "Order_Day": Order_Day,
        "High_Discount": High_Discount,
        "High_Price": High_Price,
        "Bulk_Order": Bulk_Order
    }

    if st.button("Predict Return Risk"):
        prob, pred = predict_single(sample)

        st.success(f"📊 Return Probability: **{prob:.2f}**")
        st.info(f"🎯 Prediction: **{'Returned' if pred == 1 else 'Not Returned'}**")


# ======================================================
# CSV BATCH PREDICTION
# ======================================================
if menu == "Batch Prediction (CSV)":

    st.subheader("📂 Upload a CSV File to Predict Return Risk")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df_output = predict_from_csv(uploaded_file)
        st.dataframe(df_output.head())

        csv_bytes = df_output.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Predictions CSV",
            data=csv_bytes,
            file_name="return_predictions.csv",
            mime="text/csv"
        )
