# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score

# import joblib


# df=pd.read_csv("/workspaces/codespaces-blank/Final_data")

# x1= df[['Impressions', 'Approved_Conversion_Rate', 'Cost_per_Approved_Conversion', 'Spent',
#         'Spent_x_Clicks', 'Conversion_Rate', 'Cost_per_Conversion', 'Clicks', 'interest']]
# y1 = df["ROAS"]

# # Replace inf/-inf with NaN
# x1 = x1.replace([np.inf, -np.inf], np.nan)

# # Drop rows with NaNs in x1 or y1
# x1 = x1.dropna()
# y1 = y1.loc[x1.index]  # Ensure alignment with x1



# # Split the data again
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.2, random_state=42)

# # Train Random Forest Regressor again
# rf_model_new = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_model_new.fit(X_train, y_train)

# # Evaluate performance
# from sklearn.metrics import mean_squared_error, r2_score
# y_pred = rf_model_new.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"New MSE: {mse}")
# print(f"New R²: {r2}")



# scores = cross_val_score(rf_model_new, x1, y1, cv=5, scoring='r2')
# print(f"Cross-validated R² scores: {scores}")
# print(f"Average CV R²: {scores.mean():.4f}")


# joblib.dump(rf_model_new, "rf_roas_model.pkl")




#-----------------------------------------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ------------------------------------
# Load the trained model pipeline
# ------------------------------------
model = joblib.load("rf_roas_model.pkl")

# ------------------------------------
# Streamlit UI Configuration
# ------------------------------------
st.set_page_config(page_title="ROAS Optimizer", layout="wide")
st.title("🎯 AI-Powered Ad Campaign ROAS Predictor")
st.markdown("""
    *Predict and optimize Return on Ad Spend based on campaign parameters*
    """)

# Sidebar for input
st.sidebar.header("🧾 Campaign Inputs")

impressions = st.sidebar.number_input("Impressions", min_value=0, value=10000)
clicks = st.sidebar.number_input("Clicks", min_value=1, value=100)
total_conversion = st.sidebar.number_input("Total Conversions", min_value=0, value=40)
approved_conversion = st.sidebar.number_input("Approved Conversions", min_value=1, value=35)
spent = st.sidebar.number_input("Ad Spend ($)", min_value=1.0, value=100.0)
interest = st.sidebar.number_input("Interest Score", min_value=0, value=1)

# ----------------------------------------
# Feature Engineering
# ----------------------------------------
input_df = pd.DataFrame({
    'Impressions': [impressions],
    'Clicks': [clicks],
    'Total_Conversion': [total_conversion],
    'Approved_Conversion': [approved_conversion],
    'Spent': [spent],
    'interest': [interest]
})

# Derived features
input_df["Approved_Conversion_Rate"] = input_df["Approved_Conversion"] / (input_df["Clicks"] + 1e-9)
input_df["Cost_per_Approved_Conversion"] = input_df["Spent"] / (input_df["Approved_Conversion"] + 1e-9)
input_df["Spent_x_Clicks"] = input_df["Spent"] * input_df["Clicks"]
input_df["Conversion_Rate"] = input_df["Total_Conversion"] / (input_df["Clicks"] + 1e-9)
input_df["Cost_per_Conversion"] = input_df["Spent"] / (input_df["Total_Conversion"] + 1e-9)

# Final model features
features = [
    'Impressions', 'Approved_Conversion_Rate', 'Cost_per_Approved_Conversion',
    'Spent', 'Spent_x_Clicks', 'Conversion_Rate', 'Cost_per_Conversion', 'Clicks', 'interest'
]

X_input = input_df[features]

# ----------------------------------------
# Prediction
# ----------------------------------------
st.header("🔍 Prediction")

if st.sidebar.button("Predict ROAS"):
    prediction = model.predict(X_input)[0]
    st.success(f"📊 Predicted ROAS: **{prediction:.3f}**")

    # Simple interpretation
    if prediction >= 1:
        st.markdown("✅ This campaign is likely **profitable**.")
    else:
        st.markdown("⚠️ This campaign may be **unprofitable** (ROAS < 1).")