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
