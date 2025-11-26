import os 
import joblib 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score 

FEATURE_DATA_PATH = "data/features/features.csv"
MODEL_PATH = "artifacts/model/model.pkl"

# -------------------------------
# 📥 Load Features
# -------------------------------
def load_features():
    print("📥 Loading features...")
    df = pd.read_csv(FEATURE_DATA_PATH)
    print(f"✅ Loaded. Shape: {df.shape}")
    return df 

# -------------------------------
# 🎯 Prepare Train-Test Data
# -------------------------------
def split_data(df):
    print("✂ Splitting into train/test.....")
    
    if "Sales" not in df.columns:
        raise ValueError("❌ ERROR: 'Sales' column not found for regression!")
    
    X = df.drop("Sales",axis=1)
    y = df["Sales"]
    
    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.2,random_state=42
    )
    
    print(f"📊 Train shape: {X_train.shape},Test shape: {X_test.shape}")
    
    return X_train,X_test,y_train,y_test 

# -------------------------------
# 🤖 Train the Model
# -------------------------------
def train_model(X_train,y_train):
    print("🤖 Training RandomForestRegressor.....")
    
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train,y_train)
    print("✅ Model training complete.")
    
    return model 

# -------------------------------
# 🧪 Evaluate Model
# -------------------------------
def evaluate_model(model,X_test,y_test):
    print("🧪 Evaluating model....")
    
    preds = model.predict(X_test)
    
    mae = mean_absolute_error(y_test,preds)
    mse = mean_squared_error(y_test,preds)
    rmse = mse ** 0.5 
    r2 = r2_score(y_test,preds)
    
    print(f"📌 MAE : {mae:.4f}")
    print(f"📌 MSE : {mse:.4f}")
    print(f"📌 RMSE:{rmse:.4f}")
    print(f"📌 R2 :{r2:.2f}")
    

# -------------------------------
# 💾 Save Model
# -------------------------------
def save_model(model):
    print("💾 Saving model.....")
    os.makedirs(os.path.dirname(MODEL_PATH),exist_ok=True)
    joblib.dump(model,MODEL_PATH)
    print(f"✅ Model saved at: {MODEL_PATH}")
    

# -------------------------------
# 🚀 MAIN
# -------------------------------
def main():
    df = load_features()
    X_train,X_test,y_train,y_test = split_data(df)
    model = train_model(X_train,y_train)
    evaluate_model(model,X_test,y_test)
    save_model(model)
    print("🎉 Training pipeline completed!")

if __name__ == "__main__":
    main()