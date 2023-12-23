import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd

# Load data
df = pd.read_csv('diabetes.csv')

# Process missing values
zero_not_allowed = ["Glucose", "BloodPressure", "SkinThickness"]
for column in zero_not_allowed:
    df[column] = df[column].replace(0, np.NaN)
    median = df[column].median()  # Use median instead of mean
    df[column].fillna(median, inplace=True)

# Split data into features (X) and labels (y)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
y = np.where(y == 0, -1, 1)  # Convert 0 labels to -1

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0, test_size=0.2)  # Use stratified sampling

# Standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM parameters
C = 10.0  # Increase C
gamma = 0.01  # Decrease gamma
learning_rate = 0.01
epochs = 2000  # Increase the number of epochs

# SVM training
def rbf_kernel(x1, x2, gamma):
    return np.exp(-gamma * np.linalg.norm(x1 - x2)**2)

def hinge_loss(W, X, y, C, gamma):
    N = X.shape[0]
    distances = 1 - y * (np.dot(X, W))
    distances[distances < 0] = 0  # max(0, distance)
    hinge_loss = C * np.sum(distances) + 0.5 * np.dot(W, W)
    return hinge_loss

def gradient(W, X, y, C, gamma):
    if type(y) == np.ndarray:
        y = np.where(y == 0, -1, 1)
    distance = 1 - y * (np.dot(X, W))
    dw = np.zeros(len(W))

    for ind, d in enumerate(distance):
        if max(0, d) == 0:
            di = W
        else:
            di = W - C * y[ind] * X[ind]
        dw += di

    return dw / len(y)

def train_svm(X, y, learning_rate, C, gamma, epochs):
    n_samples, n_features = X.shape
    W = np.zeros(n_features)
    for epoch in range(1, epochs + 1):
        dw = gradient(W, X, y, C, gamma)
        W = W - learning_rate * dw
        if epoch % 100 == 0:
            loss = hinge_loss(W, X, y, C, gamma)
            print(f'Epoch {epoch}/{epochs}, Loss: {loss}')

    return W

W = train_svm(X_train_scaled, y_train, learning_rate, C, gamma, epochs)

# SVM testing
def predict(X, W):
    return np.sign(np.dot(X, W))

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Predict on the test set
y_pred_test = predict(X_test_scaled, W)

# Evaluate the model
acc = accuracy(y_test, y_pred_test)
print(f'Test Accuracy: {acc}')

# Display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_test)
print(f'Confusion Matrix:\n{conf_matrix}')

# Display classification report
classification_rep = classification_report(y_test, y_pred_test)
print(f'Classification Report:\n{classification_rep}')

# Display confusion matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Diabetic', 'Diabetic'], yticklabels=['Not Diabetic', 'Diabetic'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
