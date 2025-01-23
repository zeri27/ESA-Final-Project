from src.data.datamodel import data
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X = data[['Humidity', 'Temperature', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'DayOfWeek', 'DayOfYear', 'WeekOfYear']]
y = data['CO2_PPM']

f_imp = ['Humidity', 'Temperature', 'Year', 'Day', 'Month', 'Hour', 'Minute', 'DayOfWeek', 'DayOfYear', 'WeekOfYear']

feature_removed = []
mse = []
r2 = []

# Backward Feature Selection: Random Forest Regression
while len(f_imp) > 1:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cur_mse = mean_squared_error(y_test, y_pred)
    cur_r2 = r2_score(y_test, y_pred)

    print(f_imp)
    print(cur_mse)
    print(cur_r2)

    mse.append(cur_mse)
    r2.append(cur_r2)

    elem = f_imp.pop()
    feature_removed.append(elem)

    X = data[f_imp]

# Print results
print("Order of features removed:")
print(feature_removed)
print("Mean Squared Error:")
print(mse)
print("R^2 Score:")
print(r2)

############### OUTPUT #################
# Order of features removed:
# ['WeekOfYear', 'DayOfYear', 'DayOfWeek', 'Minute', 'Hour', 'Month', 'Day', 'Year', 'Temperature']
# Mean Squared Error:
# [255.76887195897865, 255.80833298451236, 255.01342903306823, 255.30769929677692, 239.42553688381264, 265.52478868336567, 279.8775249687693, 483.1313876236117, 499.54929856310093]
# R^2 Score:
# [0.8315768820141755, 0.8315508970344224, 0.8320743391600262, 0.831880562978567, 0.8423389244416372, 0.8251526369489087, 0.8157013984995181, 0.6818592736592385, 0.6710481232246155]
