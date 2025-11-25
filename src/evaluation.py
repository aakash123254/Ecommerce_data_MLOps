import os
import joblib
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
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
# 🧪 Regression Evaluation
# ==========================
def evaluate(model, df):
    print("🧪 Running regression evaluation...")

    # ---- Ensure target exists ----
    if "Sales" not in df.columns:
        raise ValueError("❌ ERROR: Target column 'Sales' not found in dataset!")

    # ---- Split X and y ----
    X = df.drop("Sales", axis=1)
    y_true = df["Sales"]

    # ---- Predictions ----
    preds = model.predict(X)

    # ---- Regression Metrics ----
    mae = mean_absolute_error(y_true, preds)
    mse = mean_squared_error(y_true, preds)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, preds)

    print(f"📌 MAE  : {mae:.4f}")
    print(f"📌 MSE  : {mse:.4f}")
    print(f"📌 RMSE : {rmse:.4f}")
    print(f"📌 R²   : {r2:.4f}")

    return mae, mse, rmse, r2


# ==========================
# 🚀 MAIN
# ==========================
def main():
    print("🚀 Starting evaluation script....")

    df = load_feature_data()
    model = load_model()

    evaluate(model, df)

    print("🎉 Evaluation completed successfully!")


if __name__ == "__main__":
    main()
