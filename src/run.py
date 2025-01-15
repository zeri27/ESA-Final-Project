import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score

def readFloor(prefix: str, ind: int):
  def getPath(name: str) -> str:
    return f"../data/{prefix}_co2meter_{name}"

  co2 = pd.read_csv(getPath('co2_ppm.csv'))
  humidity = pd.read_csv(getPath('humidity_perc.csv'))
  temp = pd.read_csv(getPath('temp_c.csv'))
  print('min', min(co2.size, humidity.size, temp.size))
  print('max', max(co2.size, humidity.size, temp.size))
  merged = pd.merge(co2, humidity, on = 'Time') #outter join? Leave unknown values? "introduce missing values in the features"
  #"allow yourself a reject option"? "what features do the classifiers find most informative?" learning curve? Explainable AI?
  #how many players in your training set can you get a good performance? (Hint; make a learning curve)
  #extremes in data
  merged = pd.merge(merged, temp, on = 'Time')
  merged['ind'] = ind
  columns = ['co2', 'humidity', 'temp']
  merged.columns = ['Time'] + columns + ['ind']
  merged['combined'] = merged[columns].apply(list, axis = 1)
  merged = merged.drop(columns = columns)
  columns = ['ind', 'combined']
  merged['combined'] = merged[columns].apply(list, axis = 1)
  merged = merged.drop(columns = ['ind'])
  print(merged.head())
  return merged

f1meetingRoom = readFloor("floor1_meetingroom", 0)
f1office = readFloor("floor1_office", 1)
f2meetingRoom = readFloor("floor2_meetingroom", 2)

merged = pd.merge(f1meetingRoom, f1office, on = 'Time')
merged = pd.merge(merged, f2meetingRoom, on = 'Time')
columns = ['f1meetingRoom', 'f1office', 'f2meetingRoom']
merged.columns = ['Time'] + columns
merged['combined'] = merged[columns].apply(list, axis = 1)
merged = merged.drop(columns = columns)
merged['Time'] = pd.to_datetime(merged['Time'])

def getTime(x):
  return [
      #x.timestamp() // 60,
   (x - pd.Timestamp(year = x.year, month = 1, day = 1)).total_seconds() // 60,
          #x.year, x.month, x.day, x.hour, x.minute, x.weekday()
          ]
# merged['Time'] = (merged['Time'].astype(int) // 10**9) // 60 #timestamp in minutes. Leave entries with timestamp existing for all floors.
merged['Time'] = merged['Time'].apply(getTime)

print('merged.size', merged.size)
print(merged.head())
print(merged['combined'][0])
flattened = merged.explode('combined', ignore_index = True)
print(flattened.head())

labels = []
a = []
for x in flattened.to_numpy():
  # data.append([x[1][0], np.array(x[0] + x[1][1], dtype=np.float32)])
  labels.append(x[1][0])
  a.append(np.array(x[0] + x[1][1], dtype=np.float32))

labels = np.array(labels)
data = np.array(a)
print("labels.shape", labels.shape)
print("data.shape", data.shape)

xTrain, xTest, yTrain, yTest = train_test_split(data, labels, random_state = 0, test_size = 0.2)

#Dummy classifier
clf = DummyClassifier(strategy = 'most_frequent', random_state = 0)
dummyCvScores = cross_val_score(clf, xTrain, yTrain, cv = 5)
print(dummyCvScores)
clf.fit(xTrain, yTrain)
clf.score(xTest, yTest)

print('Training Data Samples: ', len(xTrain))
print('Testing Data Samples: ', len(xTest))
pipe = Pipeline([
                ('scaler', StandardScaler()), #pretty bad without scaling
                 ('SVM',
                  SVC(random_state = 0)
                  )
                 ])
paramGrid = {
    # 'C': [0.1, 0.5, 1.0, 2.0],
    #           'gamma': [0.05, 0.1, 'scale', 'auto'],
    #           'kernel': ['linear', 'rbf']
    }
grid = GridSearchCV(
    pipe, paramGrid, n_jobs = -1, refit = True, verbose = 3)
grid.fit(xTrain, yTrain)
print(grid.best_params_)
svmModel = grid.best_estimator_
yPred = svmModel.predict(xTest)
accuracy = accuracy_score(yTest, yPred)
print('SVM Accuracy: ', accuracy)

pipe = Pipeline([
                ('scaler', StandardScaler()), #pretty bad without scaling
                 ('KNN',
                  KNeighborsClassifier()
                  )
                 ])
grid = GridSearchCV(pipe,
                    param_grid = {
                        # 'n_neighbors': range(1, 30)
                        },
                    n_jobs = -1, cv = 5, refit = True, verbose = 1)
grid.fit(xTrain, yTrain)
print(grid.best_params_)
knnModel = grid.best_estimator_
yPred = knnModel.predict(xTest)
knnAccuracy = accuracy_score(yTest, yPred)
print('KNN accuracy: ', knnAccuracy)

pipe = Pipeline([
                ('scaler', StandardScaler()),
                 ('logistic',
                  LogisticRegression(max_iter = 150, tol = 1e-3, random_state = 0)
                    # SGDClassifier(loss="log_loss", max_iter=150, tol=1e-3)
                  )
                 ])
#pipe.fit(Xtrain, yTrain).score(Xtest, yTest)
param_grid = {
    # 'logistic__fit_intercept': [True, False],
    # 'logistic__penalty': ['l1', 'l2'],
    # 'logistic__C': [0.1, 0.5, 1.0, 2.0],
    # 'logistic__class_weight': [None, 'balanced'],
    # 'logistic__max_iter': [150, 1000],
}
grid = GridSearchCV(pipe, n_jobs = -1, param_grid = param_grid, cv = 5,
                            scoring = 'accuracy', refit = True, verbose = 3)
grid.fit(xTrain, yTrain)
print(grid.best_params_)
print("grid.score(xTest, yTest)", grid.score(xTest, yTest))
be = grid.best_estimator_
yPred = be.predict(xTest)
accuracyEst = accuracy_score(yTest, yPred)
print("accuracyEst", accuracyEst)

pipe = Pipeline([
    ('scaler', StandardScaler()),
                 ('decision',
                    DecisionTreeClassifier(min_impurity_decrease = 0.0, random_state = 0)
                  )
                 ])
param_grid = {
    # 'decision__criterion': ['gini', 'entropy'],
    # 'decision__class_weight': [None, 'balanced'],
}
grid = GridSearchCV(pipe, n_jobs = -1, param_grid = param_grid, cv = 5,
                            scoring = 'accuracy', refit = True, verbose = 3)
grid.fit(xTrain, yTrain)
print(grid.best_params_)
print("grid.score(xTest, yTest)", grid.score(xTest, yTest))
be = grid.best_estimator_
yPred = be.predict(xTest)
accuracyEst = accuracy_score(yTest, yPred)
print("accuracyEst", accuracyEst)