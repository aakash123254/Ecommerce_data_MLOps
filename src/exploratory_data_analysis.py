import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

# Paths
INPUT_PATH = "data/processed/processed_data.csv"
EDA_OUTPUT_PATH = "reports/eda"

def create_reports_folder():
    """Create reports/eda folder if not exists."""
    if not os.path.exists(EDA_OUTPUT_PATH):
        os.makedirs(EDA_OUTPUT_PATH)
        print(f"📁 Created folder: {EDA_OUTPUT_PATH}")

def load_data():
    """Load processed data."""
    print("📥 Loading processed data.....")
    df = pd.read_csv(INPUT_PATH)
    print(f"✅ Data loaded successfully. Shape: {df.shape}")
    return df

def basic_eda(df):
    """Generate text summary of dataset."""
    print("\n 📊 Basic Dataset Information: ")
    info_path = os.path.join(EDA_OUTPUT_PATH, "info.txt")

    # Save df.info() output
    with open(info_path, "w", encoding="utf-8") as f:
        df.info(buf=f)

    print(df.info())

    # Summary statistics
    print("\n 📈 Summary Statistics:")
    summary = df.describe()
    print(summary)

    # Save summary
    summary_path = os.path.join(EDA_OUTPUT_PATH, "summary.txt")
    summary.to_string(open(summary_path, "w", encoding="utf-8"))

    print(f"📝 Summary report saved to {summary_path}")

def visualize_data(df):
    """Create visual charts."""
    print("\n 📊 Generating Visualization....")

    # 1. Top 10 Frequent Products
    top_products = df["Description"].value_counts().head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_products.values, y=top_products.index)
    plt.title("Top 10 Most Sold Products")
    plt.xlabel("Count")
    plt.ylabel("Product")
    prod_path = os.path.join(EDA_OUTPUT_PATH, "top_products.png")
    plt.savefig(prod_path)
    plt.close()

    # 2. Sales by Country
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    country_revenue = df.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=country_revenue.values, y=country_revenue.index)
    plt.title("Top Countries by Revenue")
    plt.xlabel("Revenue")
    plt.ylabel("Country")
    country_path = os.path.join(EDA_OUTPUT_PATH, "country_revenue.png")
    plt.savefig(country_path)
    plt.close()

    print(f"📊 Visualizations saved in {EDA_OUTPUT_PATH}")

def generate_pdf_report():
    """Generate PDF combining outputs."""
    pdf_path = os.path.join(EDA_OUTPUT_PATH, "EDA_Report.pdf")
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="EDA Report", ln=True, align="C")

    pdf.set_font("Arial", size=12)
    pdf.ln(5)
    pdf.cell(200, 10, txt="1. Top 10 Most Sold Products", ln=True)
    pdf.image(os.path.join(EDA_OUTPUT_PATH, "top_products.png"), w=180)

    pdf.ln(10)
    pdf.cell(200, 10, txt="2. Top Countries by Revenue", ln=True)
    pdf.image(os.path.join(EDA_OUTPUT_PATH, "country_revenue.png"), w=180)

    pdf.output(pdf_path)

    print(f"📄 PDF Report generated: {pdf_path}")

def main():
    """Main execution function."""
    create_reports_folder()
    df = load_data()
    basic_eda(df)
    visualize_data(df)
    generate_pdf_report()
    print("\n✅ EDA process completed successfully!")

if __name__ == "__main__":
    main()
