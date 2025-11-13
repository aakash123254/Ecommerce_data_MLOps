import os
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 


# ==========================
# 📁 File Paths
# ==========================

PROCESSED_DATA_PATH = "data/processed/processed_data.csv"
EDA_OUTPUT_PATH = "reports/eda"

# Automatically create the folder for reports
os.makedirs(EDA_OUTPUT_PATH,exist_ok=True)

# ==========================
# 📊 Load Data
# ==========================
def load_data(file_path):
    print("📥 Loading processed data.....")
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Data loaded successfully. Shapr: {df.shape}")
        return df 
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None 
    

# ==========================
# 🔍 Basic EDA
# ==========================
def basic_eda(df):
    print("\n 📊 Basic Dataset Information: ")
    print(df.info())
    print("\n 📈 Summary Statistics: ")
    print(df.describe())
    
    #Save summary to text file
    summary_path = os.path.join(EDA_OUTPUT_PATH, "summary.txt")
    with open(summary_path, "w",encoding-"utf-8") as f:
        f.write("Dataset Info: \n")
        df.info(buf=f)
        f.write("\n\n Summary Statistics: \n")
        f.write(str(df.describe()))
    print(f"📝 Summary report saved to {summary_path}")

# ==========================
# 📈 Visualization
# ==========================
def visualize_data(df):
    print("\n 📊 Generating Visualization....")
    
    # 1️⃣ Top 10 most sold products
    plt.figure(figsize==(10,6))
    top_products = df['Description'].value_count().head(10)
    sns.barplot(x=top_products.values,y=top_products.index,palette="viridis")
    plt.title("Top 10 Most Sold Products")
    plt.xlabel("Count")
    plt.ylabel("Product")
    plt.tight_layout()
    plt.savefig(os.path.join())