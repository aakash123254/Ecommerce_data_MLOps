import os
import pandas as pd

# ==========================
# 📁 File Paths
# ==========================

PROCESSED_DATA_PATH = "data/processed/processed_data.csv"
FEATURE_DATA_PATH = "data/features/features.csv"

# ==========================
# 📥 Load Processed Data
# ==========================

def load_processed_data():
    print("📥 Loading processed data...")
    
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        print(f"✅ Data loaded successfully. Shape: {df.shape}")
        return df 
    except Exception as e:
        print(f"❌ Error loading processed data: {e}")
        return None 

# ==========================
# 🧩 Feature Engineering
# ==========================

def create_features(df):
    print("🛠 Creating features...")
    
    if df is None or df.empty:
        print("❌ No data available for feature engineering.")
        return None 
    
    # ---- Convert InvoiceDate to datetime ----
    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
        
        df["InvoiceYear"] = df["InvoiceDate"].dt.year 
        df["InvoiceMonth"] = df["InvoiceDate"].dt.month 
        df["InvoiceDay"] = df["InvoiceDate"].dt.day
        df["InvoiceHour"] = df["InvoiceDate"].dt.hour
        df["InvoiceDayOfWeek"] = df["InvoiceDate"].dt.dayofweek
        
    # ---- Create Sales (Regression Target) ----
    if "Quantity" in df.columns and "UnitPrice" in df.columns:
        df["Sales"] = df["Quantity"] * df["UnitPrice"]

    # ---- One-hot encode Country ----
    if "Country" in df.columns:
        df = pd.get_dummies(df, columns=["Country"], prefix="Country", drop_first=True)
    
    # ---- Drop non-essential columns ----
    drop_columns = ["InvoiceNo", "Description", "InvoiceDate"]
    df = df.drop(columns=[col for col in drop_columns if col in df.columns], errors="ignore")
    
    print(f"✅ Feature engineering completed. Shape: {df.shape}")
    print("🎯 REGRESSION TARGET CREATED: 'Sales'")
    return df

# ==========================
# 💾 Save Feature Data
# ==========================

def save_feature_data(df):
    print("💾 Saving features data...")
    
    os.makedirs(os.path.dirname(FEATURE_DATA_PATH), exist_ok=True)
    
    df.to_csv(FEATURE_DATA_PATH, index=False)
    print(f"✅ Features data saved to {FEATURE_DATA_PATH}")

# ==========================
# 🚀 Main Function
# ==========================

def main():
    df = load_processed_data()
    df_features = create_features(df) 
    
    if df_features is not None:
        save_feature_data(df_features)

if __name__ == "__main__":
    main()
