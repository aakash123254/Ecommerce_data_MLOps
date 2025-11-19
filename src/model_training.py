import os 
import pandas as pd 
import numpy as np 
import joblib 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from utils.logger import logging 

def load_feature_data(path="artifacts/feature_engineering/feature_data.csv"):
    logging.info("📥 Loading feature engineered data....")
    df = pd.read_csv(path)
    logging.info(f"✅ Data Loaded. Shape: {df.shape}") 
    return df 

def split_data(df):
      logging.info("✂️ Splitting data into train and test sets....")
      
      X = df.drop("Is_Return",axis=1)
      y = df["Is_Return"]
      
      X_train, X_test, y_train, y_test = train_test_split(
          X,y,test_size=0.2, random_state=42, stratify=y
      )
      
      logging.info(f"🔹 Train Shape: {X_train.shape}")
      logging.info(f"🔹 Test Shape: {X_test.shape}")
      
      return X_train,X_test, y_train, y_test

def train_model(X_train,y_train):
    logging.info("🛠 Training Random Forest model.....")
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        class_weight="balanced"
    )
    
    model.fit(X_train,y_train)
    
    logging.info("✅ Model training completed.")
    
    return model 

def evaluate_model(model,X_test,y_test):
    logging.info("📊 Evaluate model....")
    
    preds = model.predict(X_test)
    
    acc = accuracy_score(y_test,preds)
    precision = precision_score(y_test,preds)
    recall = recall_score(y_test,preds)
    f1 = f1_score(y_test,preds)
    
    logging.info(f"🎯 Accuracy: {acc: .4f}")
    logging.info(f"🎯 Precision: {precision: .4f}")
    logging.info(f"🎯 Recall: {recall: .4f}")
    logging.info(f"🎯 F1-score: {f1:.4f}")
    
    logging.info("📌 Classification Report ")
    logging.info("\n"+ classification_report(y_test,preds))
    
    cm = confusion_matrix(y_test,preds)
    logging.info(f"📌 Confusion Matrix: \n{cm}")
    
    return acc,precision,recall,f1 

def save_model(model, path="artifacts/model/model.pkl"):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    joblib.dump(model,path)
    logging.info(f"💾 Model saved at: {path}")
    

def main():
    logging.info("🚀 Starting model training pipeline....")
    
    df = load_feature_data()
    
    X_train, X_test, y_train, y_test = split_data(df)
    
    model = train_model(X_train,y_train)
    
    evaluate_model(model,X_test,y_test)
    
    save_model(model)
    
    logging.info("🎉 Model training pipeline completed successfully!")
    

if __name__ == "__main__":
    main()
# ==========================
# 🛠 Feature Engineering