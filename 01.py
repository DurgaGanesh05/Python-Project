import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


df = pd.read_csv('data.csv')
# # print(df.head())

# Summary
# print(df.describe())
# print(df.info())
# print(df.head(10))

# print("\n")


# Data Cleaning
# df = df.dropna()
# print("Missing values: ",df.isnull().sum().sum())
# print("\n")



# EDA

df['last_update'] = pd.to_datetime(df['last_update'], errors='coerce')
df['pollutant_avg'] = pd.to_numeric(df['pollutant_avg'], errors='coerce')
df = df.dropna(subset=['last_update', 'pollutant_avg'])
df['month'] = df['last_update'].dt.to_period('M').astype(str)
monthly_avg = df.groupby('month')['pollutant_avg'].mean().reset_index()

# print("\nCleaned Data Sample:")
# print(df.head())


# print("Actual column names in your dataset:")
# print(df.columns.tolist())



# 1. Trend Analysis of Air Quality Over Time
# Plot
plt.figure(figsize=(14, 6))
sns.set_style("whitegrid")
sns.lineplot(data=monthly_avg, x='month', y='pollutant_avg', marker='o', color='royalblue', linewidth=2.5)

# Labels and formatting
plt.title('📅 Monthly Average Air Quality Over Time', fontsize=16, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Average Pollutant Level (µg/m³)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

