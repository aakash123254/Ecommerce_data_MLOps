import pandas as pd
df = pd.read_csv("data/features/features.csv")
print(df["Is_Return"].value_counts())
