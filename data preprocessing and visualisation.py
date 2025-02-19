#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("user_data.csv")

# Display first few rows
print("🔹 Dataset Preview:")
print(df.head())

# ---------------- DATA PREPROCESSING ----------------

# Check for missing values
print("\n🔹 Checking for Missing Values:")
print(df.isnull().sum())

# Fill missing values (if any) with mean for numerical data
df.fillna(df.mean(numeric_only=True), inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Encode categorical variable 'Gender' (Male -> 1, Female -> 0)
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

print("\n✅ Data Preprocessing Completed!\n")

# ---------------- DATA VISUALIZATION ----------------

sns.set(style="whitegrid")

# 1. Histogram - Age Distribution
plt.figure(figsize=(6,4))
sns.histplot(df["Age"], bins=5, kde=True, color="blue")
plt.title("Age Distribution")
plt.show()

# 2. Boxplot - Estimated Salary Distribution
plt.figure(figsize=(6,4))
sns.boxplot(x=df["EstimatedSalary"], color="green")
plt.title("Estimated Salary Distribution")
plt.show()

# 3. Countplot - Gender Distribution
plt.figure(figsize=(6,4))
sns.countplot(x=df["Gender"], palette="pastel")
plt.xticks([0, 1], ["Female", "Male"])
plt.title("Gender Distribution")
plt.show()

# 4. Scatter Plot - Age vs Salary
plt.figure(figsize=(6,4))
sns.scatterplot(x=df["Age"], y=df["EstimatedSalary"], hue=df["Gender"], palette="coolwarm")
plt.title("Age vs Estimated Salary")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.show()

# 5. Correlation Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


# In[ ]:




