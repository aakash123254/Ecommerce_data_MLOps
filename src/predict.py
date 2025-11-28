import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "artifacts/model/model.pkl"
FEATURE_PATH = "data/features/features.csv"


# ======================
# 📦 Load Model
# ======================
def load_model():
    print("📦 Loading trained model....")
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded succesfully.")
    return model


# ======================
# 📥 Load Expected Feature Columns
# ======================
def load_feature_structure():
    print("📊 Loading feature structure for alignment....")
    df = pd.read_csv(FEATURE_PATH)
    expected_columns = df.drop("Sales", axis=1).columns.tolist()
    print(f"📌 Expected feature count: {len(expected_columns)}")
    return expected_columns


# ======================
# 🛠 Preprocess Input
# ======================
def preprocess_input(input_data: dict, expected_columns):
    print("🛠 Preprocessing input....")

    df = pd.DataFrame([input_data])

    # Convert date fields
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

    # Create Sales if missing (not used in prediction)
    if "Quantity" in df.columns and "UnitPrice" in df.columns:
        df["Sales"] = df["Quantity"] * df["UnitPrice"]

    # Drop Sales to match training input
    df = df.drop("Sales", axis=1, errors="ignore")

    # Align with training features (VERY IMPORTANT)
    df = df.reindex(columns=expected_columns, fill_value=0)

    print("✅ Input preprocessing complete.")
    return df


# ======================
# 🔮 Predict Function
# ======================
def predict_sales(input_data: dict):
    model = load_model()
    expected_columns = load_feature_structure()

    processed_df = preprocess_input(input_data, expected_columns)

    prediction = model.predict(processed_df)[0]

    print(f"\n🔮 Predicted Sales: {prediction:.4f}")


# ======================
# 🚀 MAIN
# ======================
if __name__ == "__main__":

    example_input = {
        "StockCode": "84029G",
        "Description": "WHITE METAL LANTERN",
        "Quantity": 6,
        "InvoiceDate": "2010-12-06 08:26:00",
        "UnitPrice": 3.39,
        "CustomerID": 13085.0,
        "Country": "United Kingdom"
    }

    predict_sales(example_input)
