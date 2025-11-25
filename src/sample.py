import pandas as pd

df = pd.read_csv("data/features/features.csv")

print("\n================ COLUMNS ================\n")
print(df.columns.tolist())

print("\n========= ROW COUNTS PER COLUMN =========\n")
for col in df.columns:
    try:
        print(f"{col} → {df[col].notna().sum()} rows,  unique = {df[col].nunique()}")
    except:
        print(f"{col} → ERROR reading column")
