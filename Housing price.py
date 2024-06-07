# -*- coding: utf-8 -*-
"""
Created on Tue May 14 22:30:00 2024

@author: Mohamed
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load the dataset
df = pd.read_csv("C:/Users/Mohamed/Desktop/data/housing_price_dataset.csv")

# Display basic information and statistics
df.info()
print(df.describe())
print(df.isnull().sum())

# Plot correlations between price and other features
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.scatter(df["Price"], df["Bedrooms"])
plt.title("Correlation between Price and Bedrooms")

plt.subplot(2, 2, 2)
plt.scatter(df["Price"], df["Bathrooms"])
plt.title("Correlation between Price and Bathrooms")

plt.subplot(2, 2, 3)
plt.scatter(df["Price"], df["YearBuilt"])
plt.title("Correlation between Price and YearBuilt")

plt.subplot(2, 2, 4)
plt.scatter(df["Price"], df["SquareFeet"])
plt.title("Correlation between Price and SquareFeet")

plt.tight_layout()
plt.show()

# Process 'SquareFeet' to 'Area' in square meters
df.rename(columns={"SquareFeet": "Area"}, inplace=True)
df["Area"] = df["Area"] * 0.09290304
print(df.describe())

plt.scatter(df["Price"], df["Area"])
plt.title("Correlation between Price and Area")
plt.show()

df.drop_duplicates(inplace=True)

label_encoder = LabelEncoder()
df['Neighborhood'] = label_encoder.fit_transform(df['Neighborhood'])
df.head()
# RURAL=0,SUBURB=1,URBAN=2

plt.scatter(df["Price"], df["Neighborhood"])
plt.title("Correlation between Price and Neighborhood")
plt.show()

# Train data
features = ['Bedrooms', 'Bathrooms', 'Area','Neighborhood','YearBuilt']
x = df[features]
y = df['Price']

# Adding polynomial features
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_poly, y, train_size=0.6, random_state=42)
# Models
Al1 = LinearRegression()
Al2 = DecisionTreeRegressor(random_state=42)
Al3 = RandomForestRegressor(n_estimators=40, random_state=42)

# Cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

scores_Al1 = cross_val_score(Al1, x_train, y_train, cv=kfold, scoring='r2')
scores_Al2 = cross_val_score(Al2, x_train, y_train, cv=kfold, scoring='r2')
scores_Al3 = cross_val_score(Al3, x_train, y_train, cv=kfold, scoring='r2')

print("Mean R2 Score for AL1 (Linear Regression):", np.mean(scores_Al1))
print("Mean R2 Score for AL2 (Decision Tree):", np.mean(scores_Al2))
print("Mean R2 Score for AL3 (Random Forest):", np.mean(scores_Al3))

# Fit and evaluate models on the test set
Al1.fit(x_train, y_train)
Al2.fit(x_train, y_train)
Al3.fit(x_train, y_train)

y_pred_Al1 = Al1.predict(x_test)
y_pred_Al2 = Al2.predict(x_test)
y_pred_Al3 = Al3.predict(x_test)

print("Test R2 Score for AL1 (Linear Regression):", r2_score(y_test, y_pred_Al1))
print("Test R2 Score for AL2 (Decision Tree):", r2_score(y_test, y_pred_Al2))
print("Test R2 Score for AL3 (Random Forest):", r2_score(y_test, y_pred_Al3))
