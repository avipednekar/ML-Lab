# Exp1: Data Cleaning — Missing Values and Outliers
import pandas as pd
import numpy as np
from scipy import stats

# Load dataset
df = pd.read_csv("Housing.csv")

print("Original DataFrame:")
print(df.head())

# --- Handle Missing Values ---
# Fill missing numeric values with column mean
df_filled = df.fillna(df.mean(numeric_only=True))
print("\nAfter Mean Imputation:")
print(df_filled.head())

# --- Handle Outliers (Z-score method) ---
z_scores = np.abs(stats.zscore(df_filled.select_dtypes(include=[np.number])))
df_no_outliers = df_filled[(z_scores < 3).all(axis=1)]

print("\nAfter Removing Outliers (Z-score):")
print(df_no_outliers.head())
