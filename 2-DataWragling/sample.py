import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("demo.csv")  # Or "StudentsPerformanceTest.csv"

# 1. Handle Missing Values and Inconsistencies
# Convert 'na' to NaN
df = df.replace('na', np.nan)
# Check missing values
print("Missing Values:\n", df.isnull().sum())
# Impute numeric columns with mean
numeric_cols = ['math score', 'reading score', 'writing score', 'placement score']
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
# Verify data types
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
print("Data Types:\n", df.dtypes)

# 2. Detect and Handle Outliers
# IQR method for 'math score'
Q1 = df['math score'].quantile(0.25)
Q3 = df['math score'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['math score'] < lower_bound) | (df['math score'] > upper_bound)]
print("Outliers in math score:\n", outliers)
# Cap outliers
df['math score'] = df['math score'].clip(lower=lower_bound, upper=upper_bound)
# Visualize
plt.boxplot(df['math score'].dropna())
plt.title("Math Score After Outlier Handling")
plt.show()

# 3. Data Transformation
# Log transformation to reduce skewness
df['log_math'] = np.log10(df['math score'] + 1)  # Add 1 to handle zeros
# Visualize before and after
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
df['math score'].plot(kind='hist', ax=axes[0], title='Math Score')
df['log_math'].plot(kind='hist', ax=axes[1], title='Log Math Score')
plt.show()
# Skewness
print("Skewness Before:", df['math score'].skew())
print("Skewness After:", df['log_math'].skew())