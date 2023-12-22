import warnings

warnings.filterwarnings("ignore")
import time
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Đọc dữ liệu
advertising = pd.DataFrame(pd.read_csv("advertising.csv"))

# Tạo hình 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Lấy dữ liệu
X = np.array(advertising["TV"])
Y = np.array(advertising["Sales"])
Z = np.array(advertising["Newspaper"])
x = X[:150]
y = Y[:150]
z = Z[:150]

# Tính toán w
start_time = time.time()

X1 = np.column_stack((np.ones_like(x), x, z))  # Thêm cột 1 ở đầu để tính intercept
pseudo_inverse = np.linalg.pinv(X1.T.dot(X1))  # Ma trận pseudo-inverse
w = pseudo_inverse.dot(X1.T).dot(y)
end_time = time.time()
print(end_time - start_time)
# Tạo lưới cho x và z
x_range = np.linspace(x.min(), x.max(), num=100)
z_range = np.linspace(z.min(), z.max(), num=100)
x_grid, z_grid = np.meshgrid(x_range, z_range)

# Tính giá trị y tương ứng
y_grid = w[0] + w[1] * x_grid + w[2] * z_grid

# Vẽ dữ liệu gốc
ax.scatter3D(X, Y, Z, c="green", marker="o", alpha=0.6, s=50)

# Vẽ mặt phẳng hồi quy
ax.plot_surface(x_grid, y_grid, z_grid, color="blue", alpha=0.2)

# Thêm tiêu đề và nhãn cho các trục
ax.set_title("3D Scatter plot with regression plane")
ax.set_xlabel("TV")
ax.set_ylabel("Sales")
ax.set_zlabel("Newspaper")

# Hiển thị biểu đồ
plt.show()
x_test = X[151:200]
z_test = Z[151:200]
y_test = Z[151:200]
X_test = np.column_stack((np.ones_like(x_test), x_test, z_test))

# Tính y_test
y_test = Y[151:200]
print(y_test)
y_predicted = X_test.dot(w)
print(y_predicted)
# Tính số lượng mẫu
n_samples = len(y_test)

# Tính Mean Squared Error (MSE)
MSE = np.mean((y_test - y_predicted) ** 2)
print("Mean Squared Error (MSE):", MSE)
r_squared = r2_score(y_test, y_predicted)
print("R-squared score:", r_squared)