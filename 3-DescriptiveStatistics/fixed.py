import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Task 1: Summary Statistics by Categorical Variable (Mall_Customers.csv)
print("Task 1: Summary Statistics for Mall_Customers.csv Grouped by Genre")
print("=" * 60)

# Load data
df = pd.read_csv("Mall_Customers.csv")
df = df.rename(columns={'Annual Income (k$)': 'Income'}, inplace=False)

# Define numeric and categorical columns
numeric_cols = ['Age', 'Income', 'Spending Score (1-100)']
cat_col = 'Genre'

# Compute summary statistics grouped by Genre
grouped_stats = df.groupby(cat_col)[numeric_cols].agg(['mean', 'median', 'min', 'max', 'std'])
print("\nSummary Statistics Grouped by Genre:")
print(grouped_stats)

# Create a list of numeric values for each response to the categorical variable
le = LabelEncoder()
df['Genre_encoded'] = le.fit_transform(df[cat_col])
category_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("\nNumeric Encoding for Genre:")
print(category_mapping)
numeric_values = df['Genre_encoded'].tolist()
print("\nList of Numeric Values for Genre (first 10 shown):")
print(numeric_values[:10], "...")

# Task 2: Statistical Details for Iris Species (Iris.csv)
print("\nTask 2: Statistical Details for Iris.csv by Species")
print("=" * 60)

# Load data
iris = pd.read_csv("Iris.csv")
numeric_cols_iris = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

# Compute statistics for each species
for species in ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']:
    print(f"\nStatistics for {species}:")
    stats = iris[iris['Species'] == species][numeric_cols_iris].describe()
    print(stats)