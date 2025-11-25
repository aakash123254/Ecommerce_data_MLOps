import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
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

    # Drop non-numeric columns (Regression model needs numeric only)
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

    # 🎯 For regression our target is SALES
    X = df.drop("Sales", axis=1)
    y = df["Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"🔹 Train Shape: {X_train.shape}")
    print(f"🔹 Test Shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test

# ==========================
# 🛠 Model Training
# ==========================

def train_model(X_train, y_train):
    print("🛠 Training Random Forest Regressor...")

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("✅ Model training completed.")
    return model

# ==========================
# 📊 Evaluation
# ==========================

def evaluate_model(model, X_test, y_test):
    print("📊 Evaluating regression model...")

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, preds)

    print(f"📌 MAE  : {mae:.4f}")
    print(f"📌 MSE  : {mse:.4f}")
    print(f"📌 RMSE : {rmse:.4f}")
    print(f"📌 R² Score : {r2:.4f}")

    return mae, mse, rmse, r2

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
    print("🚀 Starting regression model training pipeline...")

    df = load_feature_data()
    df = clean_data(df)

    X_train, X_test, y_train, y_test = split_data(df)

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model)

    print("🎉 Regression pipeline completed successfully!")

if __name__ == "__main__":
    main()
