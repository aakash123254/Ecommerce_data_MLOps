import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# ======================
# 📦 Paths
# ======================
MODEL_PATH = "artifacts/model/model.pkl"
FEATURE_PATH = "data/features/features.csv"

# ======================
# 🔹 Load model and feature structure
# ======================
@st.cache_data
def load_model():
    model = joblib.load(MODEL_PATH)
    return model

@st.cache_data
def load_feature_structure():
    df = pd.read_csv(FEATURE_PATH)
    expected_columns = df.drop("Sales", axis=1).columns.tolist()
    return expected_columns

# ======================
# 🔹 Preprocess input
# ======================
def preprocess_input(input_data: dict, expected_columns):
    df = pd.DataFrame([input_data])
    
    # Convert InvoiceDate
    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
        df["InvoiceYear"] = df["InvoiceDate"].dt.year
        df["InvoiceMonth"] = df["InvoiceDate"].dt.month
        df["InvoiceDay"] = df["InvoiceDate"].dt.day
        df["InvoiceHour"] = df["InvoiceDate"].dt.hour
        df["InvoiceDayOfWeek"] = df["InvoiceDate"].dt.dayofweek
        df.drop("InvoiceDate", axis=1, inplace=True)
    
    # One-hot encoding for Country
    if "Country" in df.columns:
        df = pd.get_dummies(df, columns=["Country"], prefix="Country", drop_first=True)
    
    # Drop Sales column if exists
    df = df.drop("Sales", axis=1, errors="ignore")
    
    # Align columns with training data
    df = df.reindex(columns=expected_columns, fill_value=0)
    
    return df

# ======================
# 🔹 Predict
# ======================
def predict_sales(input_data: dict):
    model = load_model()
    expected_columns = load_feature_structure()
    df = preprocess_input(input_data, expected_columns)
    prediction = model.predict(df)[0]
    return prediction

# ======================
# 🚀 Streamlit UI
# ======================
st.set_page_config(page_title="E-commerce Sales Predictor", page_icon="📈")

st.title("📦 E-commerce Sales Prediction")
st.write("Enter details of the transaction to predict sales amount.")

with st.form("sales_form"):
    stock_code = st.text_input("Stock Code", value="84029G")
    description = st.text_input("Description", value="WHITE METAL LANTERN")
    quantity = st.number_input("Quantity", min_value=1, value=6)
    invoice_date = st.text_input("Invoice Date (YYYY-MM-DD HH:MM:SS)", value="2010-12-06 08:26:00")
    unit_price = st.number_input("Unit Price", min_value=0.01, value=3.39)
    customer_id = st.number_input("Customer ID", min_value=1, value=13085)
    country = st.selectbox("Country", ["United Kingdom", "France", "Germany", "EIRE", "Spain"])
    
    submitted = st.form_submit_button("Predict Sales")

if submitted:
    input_data = {
        "StockCode": stock_code,
        "Description": description,
        "Quantity": quantity,
        "InvoiceDate": invoice_date,
        "UnitPrice": unit_price,
        "CustomerID": customer_id,
        "Country": country
    }
    
    prediction = predict_sales(input_data)
    st.success(f"🔮 Predicted Sales: {prediction:.2f}")
























