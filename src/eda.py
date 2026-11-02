
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# Create output directory
Path("outputs").mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv("data/raw/student_data.csv")

print("Dataset loaded successfully!")
print(df.head())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Correlation matrix
correlation = df.corr(numeric_only=True)

print("\nCorrelation Matrix:")
print(correlation)

# First graph: Distribution
plt.figure(figsize=(8, 5))

sns.histplot(df["final_marks"], kde=True)

plt.title("Distribution of Final Marks")
plt.savefig("outputs/final_marks_distribution.png")
plt.close()

# Second graph: Correlation Heatmap
plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Student Performance Correlation Heatmap")

plt.savefig(
    "outputs/correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nBoth graphs saved successfully!")
