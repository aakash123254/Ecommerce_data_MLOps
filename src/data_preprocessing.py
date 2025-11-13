import os
import pandas as pd
from io import StringIO

# ==========================
# 📁 File Paths
# ==========================
RAW_DATA_PATH = "data/raw/raw_data.xlsx"
PROCESSED_DATA_PATH = "data/processed/processed_data.csv"

# ==========================
# 📥 Load Data
# ==========================
def load_data(file_path):
    print("📥 Loading raw data...")

    try:
        # --- If Excel file ---
        if file_path.endswith(".xlsx") or file_path.endswith(".xls"):
            df = pd.read_excel(file_path)

        # --- If CSV file ---
        else:
            with open(file_path, "r", encoding="ISO-8859-1") as f:
                content = f.read().replace("\x00", "")
            df = pd.read_csv(StringIO(content), sep=None, engine="python", on_bad_lines="skip")

        print(f"✅ Data loaded successfully. Shape: {df.shape}")
        print(f"📊 Columns: {list(df.columns)}")
        return df

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None


# ==========================
# 🧹 Clean Data
# ==========================
def clean_data(df):
    print("🧹 Cleaning data....")

    if df is None or df.empty:
        print("❌ No data to clean!")
        return None

    # Expected columns
    possible_columns = ['InvoiceNo', 'StockCode', 'Description', 'Quantity',
                        'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country']

    df.columns = [col.strip() for col in df.columns]  # clean whitespace

    # Keep only the known columns
    df = df[[col for col in df.columns if col in possible_columns]]

    # Drop missing or invalid values
    if 'CustomerID' in df.columns:
        df = df.dropna(subset=['CustomerID'])
    if 'Description' in df.columns:
        df = df[df['Description'].notna()]
    if 'Quantity' in df.columns:
        df = df[df['Quantity'] > 0]

    # Drop duplicates
    df = df.drop_duplicates()

    print(f"✅ Data cleaned successfully. Shape: {df.shape}")
    return df


# ==========================
# 💾 Save Processed Data
# ==========================
def save_data(df, path):
    print("💾 Saving processed data...")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Processed data saved to {path}")


# ==========================
# 🚀 Main
# ==========================
def main():
    df = load_data(RAW_DATA_PATH)
    df_cleaned = clean_data(df)
    if df_cleaned is not None:
        save_data(df_cleaned, PROCESSED_DATA_PATH)


if __name__ == "__main__":
    main()
