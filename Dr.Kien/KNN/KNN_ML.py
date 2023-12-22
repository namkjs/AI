import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# Đọc dữ liệu từ tệp CSV
data = pd.read_csv('KNNDataset.csv')

X = data.drop(['id', 'diagnosis'], axis=1)  # Exclude 'id' and 'diagnosis' columns
y = data['diagnosis']

imputer = SimpleImputer(strategy='mean')  # You can choose other strategies as well
X_imputed = imputer.fit_transform(X)

# Chia dữ liệu đã được điền NaN thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=1234)

# Huấn luyện mô hình KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)


# Dự đoán nhãn cho tập kiểm tra
predictions = knn.predict(X_test)

# Hiển thị kết quả phân loại
print("Kết quả phân loại:")
print(classification_report(y_test, predictions))

# Tính độ chính xác và hiển thị
accuracy = accuracy_score(y_test, predictions)
print("Độ chính xác:", accuracy)