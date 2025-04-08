import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

df = pd.read_csv('data.csv')
# print(df.head())

# Summary
# print(df.describe())
# print(df.info())
# print(df.head(10))
# print(df.nunique())

print("\n")

# Data Cleaning
df = df.dropna()
print("Missing values: ",df.isnull().sum().sum())

df = df.drop(['longitude'], axis=1)
df = df.drop(['latitude'], axis=1)

print(df.head(10))