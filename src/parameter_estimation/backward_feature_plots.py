import matplotlib.pyplot as plt

# Data
features_removed = ['All features', 'WeekOfYear', 'DayOfYear', 'DayOfWeek', 'Minute', 'Hour', 'Month', 'Day', 'Year']
mse = [255.76887195897865, 255.80833298451236, 255.01342903306823, 255.30769929677692, 239.42553688381264, 265.52478868336567, 279.8775249687693, 483.1313876236117, 499.54929856310093]
r2 = [0.8315768820141755, 0.8315508970344224, 0.8320743391600262, 0.831880562978567, 0.8423389244416372, 0.8251526369489087, 0.8157013984995181, 0.6818592736592385, 0.6710481232246155]

# Plot MSE
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(features_removed, mse, marker='o', label='MSE')
plt.xlabel('Features Removed (Backward Selection)')
plt.ylabel('Mean Squared Error')
plt.title('MSE vs Features Removed')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()

# Plot R² Score
plt.subplot(1, 2, 2)
plt.plot(features_removed, r2, marker='o', color='orange', label='R² Score')
plt.xlabel('Features Removed (Backward Selection)')
plt.ylabel('R² Score')
plt.title('R² Score vs Features Removed')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
