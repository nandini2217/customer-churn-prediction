import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go

# ── PAGE CONFIG ────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ── LOAD MODEL ─────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load('models/xgboost_churn_model.pkl')
    with open('models/feature_columns.json') as f:
        cols = json.load(f)
    return model, cols

model, feature_cols = load_model()

# ── HEADER ─────────────────────────────────────────────
st.title("📊 Customer Churn Prediction")
st.markdown("### Predict whether a telecom customer will leave — powered by XGBoost")
st.markdown("---")

# ── SIDEBAR INPUTS ─────────────────────────────────────
st.sidebar.header("Customer Details")
st.sidebar.markdown("Fill in the customer profile below:")

tenure         = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges= st.sidebar.slider("Monthly Charges ($)", 18, 120, 65)
total_charges  = st.sidebar.number_input("Total Charges ($)",
                    min_value=0.0, max_value=9000.0,
                    value=float(tenure * monthly_charges))

st.sidebar.markdown("---")
gender         = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior         = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner        = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
dependents     = st.sidebar.selectbox("Has Dependents", ["No", "Yes"])
phone_service  = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
internet       = st.sidebar.selectbox("Internet Service",
                    ["Fiber optic", "DSL", "No"])
contract       = st.sidebar.selectbox("Contract Type",
                    ["Month-to-month", "One year", "Two year"])
payment        = st.sidebar.selectbox("Payment Method",
                    ["Electronic check", "Mailed check",
                     "Bank transfer (automatic)",
                     "Credit card (automatic)"])
paperless      = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

# ── BUILD INPUT ROW ────────────────────────────────────
def build_input():
    from sklearn.preprocessing import StandardScaler
    import warnings
    warnings.filterwarnings('ignore')

    row = {col: 0 for col in feature_cols}

    # Numeric — apply same scaling as training
    # Using approximate means and stds from the dataset
    row['tenure']         = (tenure - 32.37) / 24.56
    row['MonthlyCharges'] = (monthly_charges - 64.76) / 30.09
    row['TotalCharges']   = (total_charges - 2283.30) / 2266.77

    # Binary
    row['gender']           = 1 if gender == "Male" else 0
    row['SeniorCitizen']    = 1 if senior == "Yes" else 0
    row['Partner']          = 1 if partner == "Yes" else 0
    row['Dependents']       = 1 if dependents == "Yes" else 0
    row['PhoneService']     = 1 if phone_service == "Yes" else 0
    row['PaperlessBilling'] = 1 if paperless == "Yes" else 0
    row['MultipleLines']    = 0
    row['OnlineSecurity']   = 0
    row['OnlineBackup']     = 0
    row['DeviceProtection'] = 0
    row['TechSupport']      = 0
    row['StreamingTV']      = 0
    row['StreamingMovies']  = 0

    # Internet service one-hot
    if internet == "Fiber optic":
        row['InternetService_Fiber optic'] = 1
    elif internet == "No":
        row['InternetService_No'] = 1

    # Contract one-hot
    if contract == "One year":
        row['Contract_One year'] = 1
    elif contract == "Two year":
        row['Contract_Two year'] = 1

    # Payment one-hot
    if payment == "Credit card (automatic)":
        row['PaymentMethod_Credit card (automatic)'] = 1
    elif payment == "Electronic check":
        row['PaymentMethod_Electronic check'] = 1
    elif payment == "Mailed check":
        row['PaymentMethod_Mailed check'] = 1

    return pd.DataFrame([row])[feature_cols]

# ── PREDICTION ─────────────────────────────────────────
input_df   = build_input()
churn_prob = model.predict_proba(input_df)[0][1]
churn_pred = model.predict(input_df)[0]

# ── RESULTS ────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    color = "#C00000" if churn_prob > 0.5 else "#70AD47"
    st.markdown(f"""
    <div style='background:{color};padding:20px;border-radius:10px;text-align:center'>
        <h2 style='color:white;margin:0'>
            {"🚨 HIGH RISK" if churn_pred == 1 else "✅ LOW RISK"}
        </h2>
        <p style='color:white;font-size:18px;margin:5px 0'>
            {"Customer likely to CHURN" if churn_pred == 1 else "Customer likely to STAY"}
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(churn_prob * 100, 1),
        title={'text': "Churn Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar':  {'color': "#C00000" if churn_prob > 0.5 else "#70AD47"},
            'steps': [
                {'range': [0, 30],  'color': "#e8f5e9"},
                {'range': [30, 60], 'color': "#fff3e0"},
                {'range': [60, 100],'color': "#ffeaea"},
            ],
        }
    ))
    fig.update_layout(height=250, margin=dict(t=40, b=0, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)

with col3:
    risk = "High" if churn_prob > 0.6 else "Medium" if churn_prob > 0.3 else "Low"
    st.markdown("### Key Risk Factors")
    if contract == "Month-to-month":
        st.error("⚠️ Month-to-month contract — highest churn risk")
    if internet == "Fiber optic":
        st.warning("⚠️ Fiber optic — elevated churn segment")
    if churn_prob < 0.3:
        st.success("✅ Long contract or low charges — low risk profile")
    st.info(f"Overall Risk Level: **{risk}**")

# ── RECOMMENDATIONS ────────────────────────────────────
st.markdown("---")
st.markdown("### 💡 Retention Recommendations")

r1, r2, r3 = st.columns(3)

with r1:
    st.markdown("#### Contract Strategy")
    if contract == "Month-to-month":
        st.write("Offer 20% discount to upgrade to a 1-year contract. "
                 "Churn rate drops from 42.7% to 11.3%.")
    else:
        st.write("Customer is on a long-term contract — low churn risk. "
                 "Focus retention resources elsewhere.")

with r2:
    st.markdown("#### Pricing Strategy")
    if monthly_charges > 80:
        st.write(f"Monthly charge of ${monthly_charges} is above average. "
                 "Consider a loyalty discount or service bundle to reduce effective cost.")
    else:
        st.write("Monthly charges are within normal range. "
                 "No immediate pricing intervention needed.")

with r3:
    st.markdown("#### Service Strategy")
    if internet == "Fiber optic":
        st.write("Fiber optic customers churn at 41.9%. "
                 "Proactively check service quality and offer a tech support call.")
    else:
        st.write("Service type is not a primary risk factor for this customer.")

# ── MODEL INFO ─────────────────────────────────────────
st.markdown("---")
with st.expander("ℹ️ About this model"):
    st.write("""
    **Model:** XGBoost Classifier
    **Training data:** 5,634 telecom customers
    **Test accuracy:** 75.80%
    **ROC-AUC:** 0.836 (good discriminating power)
    **Top churn drivers:** Contract type, Internet service, Tenure, Monthly charges
    **Business impact:** Retaining top 200 high-risk customers saves ~₹12L annually
    """)