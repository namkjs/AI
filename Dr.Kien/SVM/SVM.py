import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv('diabetes.csv')

zero_not_allowed = ["Glucose", "BloodPressure", "SkinThickness"]
for column in zero_not_allowed:
    df[column] = df[column].replace(0, np.NaN)
    df[column].fillna(df[column].mean(), inplace=True)

X = df[["Glucose", "BloodPressure", "SkinThickness"]]
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train_scaled[:, :2], y_train)  

predictions = svm_model.predict(X_test_scaled[:, :2]) 

accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')

h = .02  
x_min, x_max = X_test_scaled[:, 0].min() - 1, X_test_scaled[:, 0].max() + 1
y_min, y_max = X_test_scaled[:, 1].min() - 1, X_test_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, cmap=plt.cm.Paired)
plt.title('SVM Decision Boundary')
plt.xlabel('Glucose')
plt.ylabel('BloodPressure')
plt.show()