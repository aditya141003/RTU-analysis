#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

######## Step 1: Load the dataset

file_path = r"C:\Project@GAVATAR\Energy_consumption.csv"
data = pd.read_csv(file_path)


# In[5]:


######## Step 2: Data Preprocessing


# Convert categorical columns to numeric using one-hot encoding
categorical_cols = ['DayOfWeek', 'Holiday', 'HVACUsage', 'LightingUsage']
## encoder = OneHotEncoder(drop='first', sparse_output=False)


encoder = OneHotEncoder(drop='first', sparse=False)

encoded_cats = encoder.fit_transform(data[categorical_cols])

# Create a DataFrame from the encoded columns
encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols))

# Drop the original categorical columns and concatenate the encoded ones
data = data.drop(categorical_cols + ['Timestamp'], axis=1)
data = pd.concat([data, encoded_df], axis=1)

# Check for any non-numeric columns
print("Data types after preprocessing:")
print(data.dtypes)

# Separate features and target
X = data.drop('EnergyConsumption', axis=1)
y = data['EnergyConsumption']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Output to check
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


# Step 3: Data Analysis

get_ipython().system('pip install xgboost')

# 1. Statistical Summary of the Dataset
print("Statistical Summary of the Dataset:")
print(data.describe())

# 2. Correlation Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()

# 3. Distribution of Target Variable (Energy Consumption)
plt.figure(figsize=(10, 6))
sns.histplot(y, bins=30, kde=True)
plt.title("Distribution of Energy Consumption")
plt.xlabel("Energy Consumption")
plt.ylabel("Frequency")
plt.show()

# 4. Pairplot of Features
# This can help visualize the relationships between features
sns.pairplot(data, diag_kind='kde')
plt.show()

# 5. Boxplot of Categorical Features vs. Energy Consumption
for col in categorical_cols:
    if col in data.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=col, y='EnergyConsumption', data=pd.concat([data[col], y], axis=1))
        plt.title(f"Boxplot of {col} vs Energy Consumption")
        plt.show()
    else:
        print(f"Column {col} not found in the data.")


# 6. Check for Missing Values
print("Missing Values in the Dataset:")
print(data.isnull().sum())

# 7. Feature Importance (Optional but recommended before model creation)
# This can be done using an initial model, such as a Random Forest
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Get feature importances
importances = model.feature_importances_
feature_names = X.columns

# Create a DataFrame for visualization
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title("Feature Importance")
plt.show()

import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score


# In[10]:


# Step 4: Model Creation


dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'rmse'
}

model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtest, "Test")], early_stopping_rounds=10)

y_pred = model.predict(dtest)

rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")

# Feature Importance (optional)
xgb.plot_importance(model)
plt.show()

import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Step 4: Model Creation
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Define XGBoost parameters
params = {
    'max_depth': 6,
    'eta': 0.1,
    'objective': 'reg:squarederror'
}

# Train the model
model = xgb.train(params, dtrain, num_boost_round=100)

# Predict on the test set
y_pred = model.predict(dtest)

# Evaluate the model performance
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Step 4: Model Creation - Root Mean Squared Error (RMSE): {rmse}")

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score


# In[11]:


# Step 5: Model Validation

# Evaluate the model performance on the test set
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Step 5: Model Validation - Mean Absolute Error (MAE): {mae}")
print(f"Step 5: Model Validation - R-squared (R2): {r2}")

# Hyperparameter Tuning using GridSearchCV
param_grid = {
    'max_depth': [3, 6, 9],
    'eta': [0.01, 0.1, 0.3],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'n_estimators': [50, 100, 200]
}

xgb_reg = xgb.XGBRegressor(objective='reg:squarederror')
grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)

grid_search.fit(X_train, y_train)
print("Best Hyperparameters found by GridSearchCV:", grid_search.best_params_)

# Train the model with the best hyperparameters
best_model = grid_search.best_estimator_

# Predict on the test set with the best model
y_best_pred = best_model.predict(X_test)

# Re-evaluate the model with the best hyperparameters
best_rmse = mean_squared_error(y_test, y_best_pred, squared=False)
best_mae = mean_absolute_error(y_test, y_best_pred)
best_r2 = r2_score(y_test, y_best_pred)

print(f"Best Model - Root Mean Squared Error (RMSE): {best_rmse}")
print(f"Best Model - Mean Absolute Error (MAE): {best_mae}")
print(f"Best Model - R-squared (R2): {best_r2}")


# In[12]:


# Step 6: Prediction

# Output the first few predictions and corresponding true values
print("Step 6: Prediction")
for i in range(10):
    print(f"Predicted: {y_best_pred[i]}, Actual: {y_test.iloc[i]}")


# In[ ]:




