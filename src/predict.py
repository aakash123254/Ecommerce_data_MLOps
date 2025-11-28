import os 
import joblib 
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder


FEATURE_PATH = "data/features/features.csv"
MODEL_PATH = "artifacts/model/model.pkl"

# ============================================
# 📥 Load Saved Model
# ============================================

def load_model():
    print("📦 Loading trained model....")
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded succesfully.")
    return model 

# ============================================
# 📥 Load Training Feature Structure
# ============================================
def load_feature_structure():
    print("📊 Loading feature structure for alignment....")
    df = pd.read_csv(FEATURE_PATH,nrows=5) # Only header needed 
    columns = df.drop("Sales",axis=1).columns.tolist()
    print(f"📌 Expected feature count: {len(columns)}")
    return columns 

# ============================================
# 🧩 Preprocess a Single Input
# ============================================
def preprocess_input(data_dict,expected_columns):
    print("🛠 Preprocessing input....")
    
    df = pd.DataFrame([data_dict]) # Convert input to DF
    
    
    # --- One-Hot encode COUNTRY same as training ---
    if "Country" in df.columns:
        df = df.get_dummies(df,columns=["Country"],prefix="Country",drop_first=True)
        
    # --- Add missing columns ---
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0 #Missing dummy column get 0 
    
    # ---- Ensure correct column order ----
    df = df[expected_columns]
    
    print("✅ Input preprocessed successfully.")
    return df 

# ============================================
# 🤖 Make Prediction
# ============================================
def predict_sales(input_data):
    model = load_model()
    expected_columns = load_feature_structure()
    
    processed_df = preprocess_input(input_data,expected_columns)
    prediction = model.predict(processed_df)[0]
    
    print(f"\n🎯 Predicted Sales: {prediction:.2f}")

    return prediction

# ============================================
# 🚀 MAIN (Example)
# ============================================
if __name__ == "__main__":
    example_input = {
        "StockCode" : "12345",
        "Quantity" : 10,
        "UnitPrice" : 20.0,
        "CustomerID" : 17850,
        "InvoiceYear" : 2010,
        "InvoiceMonth" : 12,
        "InvoiceDay" : 1,
        "InvoiceHour" : 8,
        "InvoiceDayOfWeek" : 3,
        "Country" : "United Kingdom"
    }
    predict_sales(example_input)
    