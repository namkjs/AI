import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import time

# Đọc dữ liệu
advertising = pd.read_csv("advertising.csv")

# Lấy dữ liệu
X = advertising[["TV", "Newspaper"]]
y = advertising["Sales"]

# Tạo mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X[:150], y[:150])  # Sử dụng 150 mẫu đầu tiên để huấn luyện

# Tạo hình 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Vẽ dữ liệu gốc
ax.scatter3D(
    advertising["TV"],
    advertising["Sales"],
    advertising["Newspaper"],
    c="green",
    marker="o",
    alpha=0.6,
    s=50,
)

# Tạo lưới cho x và z
x_range = np.linspace(X["TV"].min(), X["TV"].max(), num=100)
z_range = np.linspace(X["Newspaper"].min(), X["Newspaper"].max(), num=100)
x_grid, z_grid = np.meshgrid(x_range, z_range)

# Tính giá trị y tương ứng
y_grid = model.predict(np.column_stack((x_grid.ravel(), z_grid.ravel()))).reshape(
    x_grid.shape
)

# Vẽ mặt phẳng hồi quy
ax.plot_surface(x_grid, y_grid, z_grid, color="blue", alpha=0.2)

# Thêm tiêu đề và nhãn cho các trục
ax.set_title("3D Scatter plot with regression plane")
ax.set_xlabel("TV")
ax.set_ylabel("Sales")
ax.set_zlabel("Newspaper")

# Hiển thị biểu đồ
plt.show()

# Dùng 49 mẫu cuối cùng để dự đoán
X_test = X[151:200]
y_test = y[151:200]

# Dự đoán giá trị y từ mô hình đã huấn luyện
start_time = time.time()
y_predicted = model.predict(X_test)
end_time = time.time()
print(end_time - start_time)
# print(y_predicted)
# Tính Mean Squared Error (MSE)
MSE = mean_squared_error(y_test, y_predicted)
print("Mean Squared Error (MSE):", MSE)
r_squared = r2_score(y_test, y_predicted)
print("R-squared score:", r_squared)