from src.data.datamodel import data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Linear Regression
X = data[['Humidity', 'Temperature', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'DayOfWeek', 'DayOfYear', 'WeekOfYear']]
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
# Model Coefficients: [ 8.51129184e-01  4.06888878e+00  2.08839490e+01 -9.93823348e+01
#  -3.43174087e+00 -6.35823987e-01  8.13939171e-03 -8.86333744e-02
#   3.32075694e+00  4.24025363e-02]
# Model Intercept: -41825.12831104357
# Mean Squared Error: 1171.3966157821772
# R^2 Score: 0.22863846207321314