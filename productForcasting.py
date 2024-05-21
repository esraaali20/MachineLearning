import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the dataset
df = pd.read_csv('FINAL.csv')

# Drop unnecessary columns
columns_to_drop = ['ID', 'name', 'Date', 'Sales',  'Category_Section']
df = df.drop(columns_to_drop, axis=1)

# Print the first few rows of the dataframe
print(df.head())

# Check for missing values
print("*")
print(df.isnull().sum())
print("*")

# Drop rows with missing values
df = df.dropna()

# Split the dataset into features (X) and target variable (y)
X = df[['actual_price', 'Total_Price']]
y = df['Units_Sold']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the decision tree regressor
tree_regressor = DecisionTreeRegressor(random_state=42)
tree_regressor.fit(X_train, y_train)

# Make predictions on the test set
features = np.array([[1250, 7500]])
print("Decision Tree Prediction:", tree_regressor.predict(features))

y_pred_tree = tree_regressor.predict(X_test)

# Evaluate the decision tree model using Mean Squared Error
mse_tree = mean_squared_error(y_test, y_pred_tree)
print("Decision Tree Mean Squared Error:", mse_tree)

# Calculate R-squared (R²) for decision tree model
r2_tree = r2_score(y_test, y_pred_tree)
print("Decision Tree R-squared (R²):", r2_tree)

# Calculate Mean Absolute Error (MAE) for decision tree model
mae_tree = mean_absolute_error(y_test, y_pred_tree)
print("Decision Tree Mean Absolute Error (MAE):", mae_tree)

# Initialize and train the linear regression model
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)




# Make predictions on the test set using linear regression
y_pred_linear = linear_regressor.predict(X_test)

# Make predictions using Linear Regression model
print("Linear Regression Prediction:", np.floor(linear_regressor.predict(features)))


# Evaluate the Linear Regression model using Mean Squared Error
mse_linear = mean_squared_error(y_test, y_pred_linear)
print("Linear Regression Mean Squared Error:", mse_linear)

# Calculate R-squared (R²) for Linear Regression model
r2_linear = r2_score(y_test, y_pred_linear)
print("Linear Regression R-squared (R²):", r2_linear)

# Calculate Mean Absolute Error (MAE) for Linear Regression model
mae_linear = mean_absolute_error(y_test, y_pred_linear)
print("Linear Regression Mean Absolute Error (MAE):", mae_linear)
