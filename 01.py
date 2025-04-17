import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


df = pd.read_csv('data.csv')

print("Original Data: " ,df.head(10))
print("\n")

# Summary
print("Descriptive Stats: \n",df.describe())
print("\n")

print("Info\n" ,df.info())
print("\n")




# Data Cleaning
df = df.dropna()
print("Missing values: ",df.isnull().sum().sum())
print("\n")



# EDA
df['last_update'] = pd.to_datetime(df['last_update'], errors='coerce')
df['pollutant_avg'] = pd.to_numeric(df['pollutant_avg'], errors='coerce')
df = df.dropna(subset=['last_update', 'pollutant_avg', 'station'])
df['month'] = df['last_update'].dt.to_period('M').astype(str)
monthly_avg = df.groupby('month')['pollutant_avg'].mean().reset_index()


df['state'] = df['state'].str.strip().str.title()
state_avg = df.groupby('state')['pollutant_avg'].mean().reset_index().sort_values(by='pollutant_avg', ascending=False)


df['pollutant_avg'] = pd.to_numeric(df['pollutant_avg'], errors='coerce')
df_clean = df.dropna(subset=['pollutant_avg', 'pollutant_id'])


print("\nCleaned Data Sample:")
print(df.head())


print("\nActual column names in your dataset:")
print(df.columns.tolist())


# -----------------------------------------------------------------------------------------


# 1. Trend Analysis of Air Quality Over Time
# Plot
plt.figure(figsize=(14, 6))
sns.set_style("whitegrid")
sns.lineplot(data=monthly_avg, x='month', y='pollutant_avg', marker='o', color='royalblue', linewidth=2.5)

plt.title('Trend Analysis of Air Quality Over Time', fontsize=16, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Average Pollutant Level (µg/m³)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()




# 2. Neighborhood Comparison of Air Quality
# Plot
plt.figure(figsize=(14, 6))
sns.set_style("whitegrid")
sns.barplot(data=state_avg, x='state', y='pollutant_avg', palette='viridis')

plt.title('Neighborhood Comparison of Air Quality', fontsize=16, fontweight='bold')
plt.xlabel('State')
plt.ylabel('Average Pollutant Level (µg/m³)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.grid(axis='y')
plt.show()




# 3. Correlation Analysis Between Pollutants
print("\nCorrelation Analysis Between Pollutants:")
numeric_data = df.select_dtypes(include=['number'])
correlation_matrix = numeric_data.corr()
print(correlation_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Analysis Between Pollutants')
plt.show()



# 4. Outlier Detection in Air Quality Data
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_clean, x='pollutant_id', y='pollutant_avg', palette="Set2")

plt.title("🚨 Outlier Detection for Each Pollutant", fontsize=14)
plt.xlabel("Pollutant Type")
plt.ylabel("Average Pollutant Level")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

outlier_counts = {}

for pollutant in df_clean['pollutant_id'].unique():
    sub_df = df_clean[df_clean['pollutant_id'] == pollutant]
    Q1 = sub_df['pollutant_avg'].quantile(0.25)
    Q3 = sub_df['pollutant_avg'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = sub_df[(sub_df['pollutant_avg'] < lower) | (sub_df['pollutant_avg'] > upper)]
    outlier_counts[pollutant] = len(outliers)

print("\nOutliers detected per pollutant:")
for k, v in outlier_counts.items():
    print(f"{k}: {v} outliers")



# 5. Data Visualization for Public Awareness

# Count how often each pollutant is recorded
pollutant_freq = df['pollutant_id'].value_counts().reset_index()
pollutant_freq.columns = ['Pollutant', 'Frequency']

plt.figure(figsize=(10, 6))
sns.barplot(x='Pollutant', y='Frequency', data=pollutant_freq, palette='mako')

plt.title("Most Frequently Monitored Pollutants", fontsize=16)
plt.xlabel("Pollutant", fontsize=12)
plt.ylabel("Monitoring Frequency", fontsize=12)
plt.xticks(rotation=15)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()




# Weekend vs Weekday Pollution Levels

df['last_update'] = pd.to_datetime(df['last_update'], errors='coerce')
df = df.dropna(subset=['last_update'])

df['day_of_week'] = df['last_update'].dt.dayofweek  # 0 = Monday, ..., 6 = Sunday
df['day_type'] = df['day_of_week'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')

df['pollutant_avg'] = pd.to_numeric(df['pollutant_avg'], errors='coerce')
df = df.dropna(subset=['pollutant_avg'])

daytype_avg = df.groupby('day_type')['pollutant_avg'].mean().reset_index()

plt.figure(figsize=(8, 5))
sns.barplot(x='day_type', y='pollutant_avg', data=daytype_avg, palette='coolwarm')

plt.title('Average Air Pollution: Weekend vs Weekday', fontsize=16)
plt.xlabel('Day Type', fontsize=12)
plt.ylabel('Mean Pollutant Level', fontsize=12)
plt.tight_layout()
plt.show()



# Pollution Spikes
df['last_update'] = pd.to_datetime(df['last_update'], errors='coerce')
df = df.dropna(subset=['last_update'])

df['pollutant_avg'] = pd.to_numeric(df['pollutant_avg'], errors='coerce')
df = df.dropna(subset=['pollutant_avg'])

top_spikes = df.sort_values(by='pollutant_avg', ascending=False).head(5)
print("Top 5 Pollution Spikes:")
print(top_spikes[['last_update', 'state', 'pollutant_avg']])

plt.figure(figsize=(10, 5))
sns.histplot(df['pollutant_avg'], bins=30, kde=True, color='crimson')

plt.title('Distribution of Pollution Levels (Spikes Highlighted)', fontsize=14)
plt.xlabel('Pollutant Level')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()


