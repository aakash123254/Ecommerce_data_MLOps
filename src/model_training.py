import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ==========================
# 📥 Load Feature Data
# ==========================

def load_feature_data():
    path = "data/features/features.csv"
    print(f"📥 Loading features from: {path}")
    df = pd.read_csv(path)
    print(f"✅ Loaded. Shape: {df.shape}")
    return df

# ==========================
# 🧹 Clean / Select Valid Columns
# ==========================

def clean_data(df):
    print("🧹 Cleaning data...")

    drop_cols = [col for col in df.columns if df[col].dtype == "object"]

    if drop_cols:
        print(f"⚠️ Dropping non-numeric columns: {drop_cols}")
        df = df.drop(columns=drop_cols)

    print(f"✅ Cleaned data shape: {df.shape}")
    return df

# ==========================
# ✂️ Train/Test Split
# ==========================

def split_data(df):
    print("✂️ Splitting data into train/test...")

    X = df.drop("HighValue", axis=1)
    y = df["HighValue"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"🔹 Train Shape: {X_train.shape}")
    print(f"🔹 Test Shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test

# ==========================
# 🛠 Model Training
# ==========================

def train_model(X_train, y_train):
    print("🛠 Training Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    print("✅ Model training completed.")
    return model

# ==========================
# 📊 Evaluation
# ==========================

def evaluate_model(model, X_test, y_test):
    print("📊 Evaluating model...")

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print(f"🎯 Accuracy: {acc:.4f}")
    print(f"🎯 Precision: {prec:.4f}")
    print(f"🎯 Recall: {rec:.4f}")
    print(f"🎯 F1 Score: {f1:.4f}")

    print("\n📌 Classification Report")
    print(classification_report(y_test, preds))

    print("\n📌 Confusion Matrix")
    print(confusion_matrix(y_test, preds))

    return acc, prec, rec, f1

# ==========================
# 💾 Save Model
# ==========================

def save_model(model, path="artifacts/model/model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"💾 Model saved at: {path}")

# ==========================
# 🚀 MAIN
# ==========================

def main():
    print("🚀 Starting model training pipeline...")

    df = load_feature_data()
    df = clean_data(df)

    X_train, X_test, y_train, y_test = split_data(df)

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model)

    print("🎉 Pipeline completed successfully!")

if __name__ == "__main__":
    main()
