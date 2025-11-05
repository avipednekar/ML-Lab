# Exp2: Feature Transformation and Scaling
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load dataset
df = pd.read_csv("Housing.csv")

print("Original DataFrame:")
print(df.head())

# --- Log Transformation (for skewed features like 'price') ---
df['price_log'] = np.log1p(df['price'])

# --- Min-Max Scaling for 'area' ---
min_max_scaler = MinMaxScaler()
df['area_minmax'] = min_max_scaler.fit_transform(df[['area']])

# --- Standard Scaling for 'bedrooms' ---
standard_scaler = StandardScaler()
df['bedrooms_std'] = standard_scaler.fit_transform(df[['bedrooms']])

# --- Derived Feature: price per area ---
df['price_per_sqft'] = df['price'] / df['area']

print("\nAfter Transformations:")
print(df[['area', 'area_minmax', 'bedrooms', 'bedrooms_std', 'price', 'price_log', 'price_per_sqft']].head())
