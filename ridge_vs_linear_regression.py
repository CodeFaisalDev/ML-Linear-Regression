# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import machine learning models and utilities
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load the California housing dataset
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()

# Convert the dataset to a pandas DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

# Show summary statistics of the features
df.describe()

# Add the target variable (median house value) to the DataFrame
df["MedHouseValue"] = data.target

# Display the first few rows of the dataset
print(df.head())

# Split the data into features (X) and target (y)
X = df.drop("MedHouseValue", axis=1)
y = df["MedHouseValue"]

# Split the dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

# Standardize the features (important for models like Ridge)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train, transform train
X_test_scaled = scaler.fit_transform(X_test)    # Incorrect: should use transform only (explained below)

# Initialize and train Ridge Regression model with alpha=100 (regularization strength)
ridge = Ridge(alpha=100)
ridge.fit(X_train_scaled, y_train)

# Predict house values using the Ridge Regression model
y_pred_ridge = ridge.predict(X_test_scaled)

# Initialize and train a standard Linear Regression model
linear = LinearRegression()
linear.fit(X_train_scaled, y_train)

# Predict house values using the Linear Regression model
y_pred_linear = linear.predict(X_test_scaled)

# Evaluate the Linear Regression model
print("R^2 Score", r2_score(y_test, y_pred_linear))  # Coefficient of determination
print("MSE", mean_squared_error(y_test, y_pred_linear))  # Mean Squared Error

# Plot predictions vs. actual values for Linear Regression
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_linear, alpha=0.5, color='blue')  # Scatter plot of predictions
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Perfect prediction line
plt.title(f'Linear Regression\nR²: {r2_score(y_test, y_pred_linear):.3f}, MSE: {mean_squared_error(y_test, y_pred_linear):.3f}')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid(True)

# Plot predictions vs. actual values for Ridge Regression
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_ridge, alpha=0.5, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title(f'Ridge Regression\nR²: {r2_score(y_test, y_pred_ridge):.3f}, MSE: {mean_squared_error(y_test, y_pred_ridge):.3f}')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid(True)

# Adjust layout and display plots
plt.tight_layout()
plt.show()
