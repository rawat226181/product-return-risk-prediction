from prediction_pipeline import predict_single

# Sample input (you can modify anytime)
sample = {
    "Product_Price": 1500,
    "Order_Quantity": 1,
    "Days_to_Return": 6,
    "User_Age": 29,
    "Discount_Applied": 0.10,
    "Order_Year": 2024,
    "Order_Month": 6,
    "Order_DayOfWeek": 4,
    "Order_Day": 12,
    "High_Discount": 0,
    "High_Price": 1,
    "Bulk_Order": 0
}

prob, pred = predict_single(sample)

print("\n==============================")
print("     TEST PREDICTION")
print("==============================")
print(f"Probability of Return : {prob:.4f}")
print(f"Prediction            : {'Returned' if pred == 1 else 'Not Returned'}")
print("==============================\n")
