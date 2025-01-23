import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from src.data.datamodel import data
from sklearn.preprocessing import StandardScaler

# Scaling
scaler = StandardScaler()
X = data[['Humidity', 'Temperature', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'DayOfWeek', 'DayOfYear', 'WeekOfYear']]
X_scaled = scaler.fit_transform(X)

# Applying PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_
pca_components = pca.components_
contributions = np.abs(pca_components) * explained_variance

total_contributions = contributions.sum(axis=0)
feature_contributions_df = pd.DataFrame(total_contributions, index=X.columns, columns=['Contribution'])
feature_contributions_df['Percentage Contribution'] = (feature_contributions_df['Contribution'] /
                                                       feature_contributions_df['Contribution'].sum()) * 100

plt.figure(figsize=(12, 8))
feature_contributions_df['Percentage Contribution'].sort_values().plot(kind='barh', color='skyblue')
plt.xlabel('Percentage Contribution')
plt.title('Total Feature Contribution to Explained Variance in PCA')
plt.show()