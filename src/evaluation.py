import os 
import joblib 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.metrics import(
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, auc
)

# ==========================
# 📥 Load Features
# ==========================
def load_feature_data(path="data/features/features.csv"):
    print(f"📥 Loading features from: {path}")
    df = pd.read_csv(path)
    print(f"✅ Loaded. Shape: {df.shape}")
    return df 


# ==========================
# 📤 Load Trained Model
# ==========================
def load_model(path="artifacts/model/model.pkl"):
    print(f"📦 Loading model from: {path}")
    model = joblib.load(path)
    print("✅ Model loaded successfully.")
    return model


# ==========================
# 🧪 Evaluate Model
# ==========================
def evaluate(model,df):
    print("🧪 Running evaluation....")
    
    if "Is_Return" not in df.columns:
        raise ValueError("❌ ERROR: Target column 'Is_Return' not found in dataset!")
    
    X = df.drop("Is_return",axis=1)
    y_true = df["Is_Return"]
    
    preds = model.predict(X)
    
    acc = accuracy_score(y_true,preds)
    prec = precision_score(y_true,preds,zero_division=0)
    rec = recall_score(y_true,preds,zero_division=0)
    f1 = f1_score(y_true,preds,zero_division=0)
    
    print(f"🎯 Accuracy: {acc:.4f}")
    print(f"🎯 Precision: {prec:.4f}")
    print(f"🎯 Recall: {rec:.4f}")
    print(f"🎯 F1 Score: {f1:.4f}")
    
    print("\n📌 Classification Report")
    print(classification_report(y_true,preds,zero_division=0))
    
    print("\n 📌Confusion Matrix")
    print(confusion_matrix(y_true,preds))
    

