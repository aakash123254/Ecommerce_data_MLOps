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