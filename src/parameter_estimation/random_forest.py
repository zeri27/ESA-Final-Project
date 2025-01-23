from src.data.datamodel import data
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Random Forest Regression
X = data[['Humidity', 'Temperature']]
y = data['CO2_PPM']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

############### OUTPUT ####################
# Mean Squared Error: 499.54929856310093
# R^2 Score: 0.6710481232246155