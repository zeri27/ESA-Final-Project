from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from src.data.datamodel import data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Data
X = data[['Humidity', 'Temperature', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'DayOfWeek', 'DayOfYear', 'WeekOfYear']]
y = data['CO2_PPM']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Random Forest Regression
model2 = RandomForestRegressor(random_state=42)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
mse2 = mean_squared_error(y_test, y_pred2)
r2_2 = r2_score(y_test, y_pred2)

# Print results
print(f"Linear Regression MSE: {mse}")
print(f"Random Forest MSE: {mse2}")
print(f"Linear Regression R2: {r2}")
print(f"Random Forest R2: {r2_2}")

# Graph
plt.scatter(y_test, y_pred, color='green', alpha=0.5, label='Linear Regression')
plt.scatter(y_test, y_pred2, color='blue', alpha=0.5, label='Random Forest')

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title("Predicted vs Actual CO2_PPM")
plt.xlabel("Actual CO2_PPM")
plt.ylabel("Predicted CO2_PPM")
plt.legend()
plt.show()
