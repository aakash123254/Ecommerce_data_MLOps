import os 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import joblib 


# ================================================
# 📁 PATHS
# ================================================
RAW_DATA_PATH = "data/raw/Online_Retail.csv"
PROCESSED_DATA_PATH = "data/processed/processed_data.csv"
FEATURE_DATA_PATH = "data/features/features.csv"
MODEL_PATH = "artifacts/model/model.pkl"
METRIC_PATH = "artifacts/metrics/metrics.txt"

# ================================================
# 🧹 STEP 1 — DATA PREPROCESSING
# ================================================
def load_raw_data():
    print("📥 Loading raw data.....")

    try:
        df = pd.read_csv(RAW_DATA_PATH,encoding="ISO-8859-1")
        print(f"✅ Raw data loaded. Shape: {df.shape}")
        return df 
    except Exception as e:
        print(f"❌Failed to load raw data: {e}")
        return None 

def preprocess_data(df):
    print("🧹 Cleaning data....")
    
    # Remove missing InvoiceNo or CustomerID
    df = df.dropna(subset=["InvoiceNo","CustomerID"])
    
    # Remove negative quantities
    df = df[df["Quantity"]>0]
    
    # Remove negative price
    df = df[df["UnitPrice"]>0]
    
    # Add target column 
    df["Sales"] = df["Quantity"] * df["UnitPrice"]
    
    print(f"✅ Preprocessing complete. Shape: {df.shape}")
    return df 

def save_processed(df):
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH),exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH,index=False)
    print(f"💾 Processed data saved → {PROCESSED_DATA_PATH}")


# ================================================
# 🧩 STEP 2 — FEATURE ENGINEERING
# ================================================