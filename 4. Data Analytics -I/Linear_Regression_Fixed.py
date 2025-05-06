import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv('data4.csv')
X = data.iloc[:, 1].values.reshape(-1, 1)  # Feature (e.g., area)
y = data.iloc[:, -1].values  # Target (e.g., price)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print results
print(f'Intercept: {model.intercept_}')
print(f'Coefficient: {model.coef_[0]}')
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-Squared Score (R²): {r2}')

# Visualize the regression line
plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, color='blue', label='Training data')
plt.scatter(x_test, y_test, color='green', label='Testing data')
plt.plot(x_test, y_pred, color='red', label='Regression line')
plt.xlabel('Area (sq ft)')
plt.ylabel('Price ($)')
plt.title('Simple Linear Regression: Area vs Price')
plt.legend()
plt.savefig('regression_plot.png')
plt.close()

# User input for predicting price from area
try:
    user_area = float(input("Enter the house area (sq ft): "))
    predicted_price = model.predict([[user_area]])[0]
    print(f'Predicted price for area {user_area} sq ft: ${predicted_price:.2f}')
except ValueError:
    print("Please enter a valid number for the area.")