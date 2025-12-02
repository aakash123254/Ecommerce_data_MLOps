import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


# ================================================
# 📁 PATHS
# ================================================
RAW_DATA_PATH = "data/raw/raw_data.xlsx"
PROCESSED_DATA_PATH = "data/processed/processed_data.csv"
FEATURE_DATA_PATH = "data/features/features.csv"
MODEL_PATH = "artifacts/model/model.pkl"
METRIC_PATH = "artifacts/metrics/metrics.txt"


# ================================================
# 🧹 STEP 1 — LOAD + PREPROCESS RAW DATA
# ================================================
def load_raw_data():
    print("📥 Loading raw data...")

    try:
        df = pd.read_excel(RAW_DATA_PATH)
        print(f"✅ Raw data loaded. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Failed to load raw data: {e}")
        return None


def preprocess_data(df):
    print("🧹 Cleaning data...")

    # Remove missing InvoiceNo or CustomerID
    df = df.dropna(subset=["InvoiceNo", "CustomerID"])

    # Remove negative quantities
    df = df[df["Quantity"] > 0]

    # Remove negative prices
    df = df[df["UnitPrice"] > 0]

    # Add Sales Column
    df["Sales"] = df["Quantity"] * df["UnitPrice"]

    print(f"✅ Preprocessing complete. Shape: {df.shape}")
    return df


def save_processed(df):
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"💾 Processed data saved → {PROCESSED_DATA_PATH}")


# ================================================
# 🧩 STEP 2 — FEATURE ENGINEERING
# ================================================
def create_features(df):
    print("🧩 Creating features...")

    # Fix InvoiceDate conversion
    if not pd.api.types.is_datetime64_any_dtype(df["InvoiceDate"]):
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    # Date features
    df["InvoiceYear"] = df["InvoiceDate"].dt.year
    df["InvoiceMonth"] = df["InvoiceDate"].dt.month
    df["InvoiceDay"] = df["InvoiceDate"].dt.day
    df["InvoiceHour"] = df["InvoiceDate"].dt.hour
    df["InvoiceDayOfWeek"] = df["InvoiceDate"].dt.dayofweek

    # One-hot encode Country
    df = pd.get_dummies(df, columns=["Country"], prefix="Country", drop_first=True)

    # Drop non-numeric & unnecessary columns
    drop_cols = ["InvoiceNo", "Description", "InvoiceDate", "StockCode"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    print(f"✅ Features created. Shape: {df.shape}")
    return df



def save_features(df):
    os.makedirs(os.path.dirname(FEATURE_DATA_PATH), exist_ok=True)
    df.to_csv(FEATURE_DATA_PATH, index=False)
    print(f"💾 Feature data saved → {FEATURE_DATA_PATH}")


# ================================================
# 🤖 STEP 3 — MODEL TRAINING
# ================================================
def train_model(X_train, y_train):
    print("🤖 Training model...")

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("✅ Training complete.")
    return model


def save_model(model):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"💾 Model saved → {MODEL_PATH}")


# ================================================
# 📊 STEP 4 — MODEL EVALUATION
# ================================================
def evaluate_model(model, X_test, y_test):
    print("📊 Evaluating model...")

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, preds)

    print(f"📌 MAE : {mae:.4f}")
    print(f"📌 MSE : {mse:.4f}")
    print(f"📌 RMSE: {rmse:.4f}")
    print(f"📌 R2  : {r2:.4f}")

    # Save metrics
    os.makedirs(os.path.dirname(METRIC_PATH), exist_ok=True)
    with open(METRIC_PATH, "w") as f:
        f.write(f"MAE: {mae}\nMSE: {mse}\nRMSE: {rmse}\nR2: {r2}\n")

    print(f"💾 Metrics saved → {METRIC_PATH}")

    return mae, mse, rmse, r2


# ================================================
# 🚀 MAIN EXECUTION PIPELINE
# ================================================
def main():
    # Step 1 — Raw → Processed
    df_raw = load_raw_data()
    if df_raw is None:
        print("❌ Pipeline stopped. Raw data not found.")
        return

    df_processed = preprocess_data(df_raw)
    save_processed(df_processed)

    # Step 2 — Processed → Features
    df_features = create_features(df_processed)
    save_features(df_features)

    # Step 3 — Train/Test Split
    print("✂ Splitting dataset...")
    X = df_features.drop("Sales", axis=1)
    y = df_features["Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"📊 Train: {X_train.shape}, Test: {X_test.shape}")

    # Train model
    model = train_model(X_train, y_train)
    save_model(model)

    # Step 4 — Evaluate
    evaluate_model(model, X_test, y_test)

    print("\n🎉 Pipeline execution completed successfully!")


if __name__ == "__main__":
    main()
