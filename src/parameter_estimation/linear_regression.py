from matplotlib import pyplot as plt

from src.data.datamodel import data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Linear Regression
X = data[['Humidity', 'Temperature']]
y = data['CO2_PPM']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

############### OUTPUT ####################
# Model Coefficients: [0.82774832 4.0717688]
# Model Intercept: 347.02394848073243
# Mean Squared Error: 1254.5024693969015
# R^2 Score: 0.1739134797817391

# Graphs
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title("Predicted vs Actual CO2_PPM")
plt.xlabel("Actual CO2_PPM")
plt.ylabel("Predicted CO2_PPM")
plt.show()

