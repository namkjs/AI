import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Đọc dữ liệu từ tệp CSV
data = pd.read_csv('KNNDataset.csv')

X = data.drop(['id', 'diagnosis'], axis=1)  # Exclude 'id' and 'diagnosis' columns
y = data['diagnosis']

# Handle NaN values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Check for NaN values after imputation
if np.isnan(X_imputed).any():
    print("Warning: NaN values present after imputation. Handle them appropriately.")

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=1234)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hàm tính khoảng cách Euclidean giữa hai điểm
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

# Hàm dự đoán nhãn cho một điểm dữ liệu trong tập kiểm tra
def predict_label(X_train, y_train, x_test, k):
    distances = [(euclidean_distance(x_test, x), label) for x, label in zip(X_train, y_train)]
    sorted_distances = sorted(distances, key=lambda x: x[0])
    k_nearest_labels = [label for (_, label) in sorted_distances[:k]]
    # Lấy nhãn xuất hiện nhiều nhất trong k láng giềng
    predicted_label = max(set(k_nearest_labels), key=k_nearest_labels.count)
    return predicted_label

# Hàm dự đoán cho toàn bộ tập kiểm tra
def knn_predict(X_train, y_train, X_test, k):
    predictions = [predict_label(X_train, y_train, x_test, k) for x_test in X_test]
    return np.array(predictions)

# Chọn số láng giềng k
k_neighbors = 3

# Dự đoán nhãn cho tập kiểm tra
predictions = knn_predict(X_train_scaled, y_train.values, X_test_scaled, k_neighbors)

# Hiển thị kết quả phân loại
print("Kết quả phân loại:")
print(classification_report(y_test, predictions))

# Tính độ chính xác và hiển thị
accuracy = accuracy_score(y_test, predictions)
print("Độ chính xác:", accuracy)