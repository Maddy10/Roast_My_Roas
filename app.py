import streamlit as st
import pandas as pd
import numpy as np
import joblib

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

# Add feature explanation section
with st.expander("ℹ️ About the Input Features"):
    st.markdown("""
    This app predicts **Return on Ad Spend (ROAS)** based on your ad campaign metrics.
    
    #### 🔢 Input Features Explained:
    
    - **Impressions**: Number of times your ad was shown. More impressions = more exposure.
    - **Clicks**: How many users clicked on your ad. A higher number indicates better engagement.
    - **Total Conversions**: Total actions taken by users (e.g., purchases, signups).
    - **Approved Conversions**: Verified/legit conversions (e.g., filtered for fraud or quality).
    - **Spent**: Total money spent on the ad campaign.
    - **Interest Score**: A numeric code representing the targeted audience interest group.

    """)
# Sidebar for input
# Sidebar for input
st.sidebar.header("🧾 Campaign Inputs")

impressions = st.sidebar.number_input(
    "Impressions", min_value=0, value=10000,
    help="How many times your ad was shown to users."
)

clicks = st.sidebar.number_input(
    "Clicks", min_value=1, value=100,
    help="Number of users who clicked on your ad."
)

total_conversion = st.sidebar.number_input(
    "Total Conversions", min_value=0, value=40,
    help="Total number of user actions from the ad (e.g., purchases, sign-ups)."
)

approved_conversion = st.sidebar.number_input(
    "Approved Conversions", min_value=1, value=35,
    help="Subset of conversions that were verified or qualified."
)

spent = st.sidebar.number_input(
    "Ad Spend ($)", min_value=1.0, value=100.0,
    help="Total money spent on this ad campaign."
)

interest = st.sidebar.number_input(
    "Interest Score", min_value=0, value=1,
    help="Audience interest category or segment code targeted."
)

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

    # Interpretation with clearer thresholds and emoji
    if prediction >= 2:
        st.success("💸 Excellent! This campaign is **highly profitable** (ROAS ≥ 2).")
        st.markdown("Double or more return on ad spend – strong performance!")
    elif prediction >= 1:
        st.info("📈 Good! This campaign is **profitable** (ROAS ≥ 1).")
        st.markdown("You're making more than you spend – keep optimizing!")
    elif prediction >= 0.5:
        st.warning("⚠️ Caution: This campaign is **underperforming** (0.5 ≤ ROAS < 1).")
        st.markdown("You're spending more than you're earning – consider adjustments.")
    else:
        st.error("🔻 Warning: This campaign is **loss-making** (ROAS < 0.5).")
        st.markdown("Significant losses detected. Pause or revise strategy urgently.")
