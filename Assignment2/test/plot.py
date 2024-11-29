import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 数据
demos = np.array([0, 5, 8, 15, 20, 25, 30])
accuracy = np.array([0.8271, 0.8514, 0.8484, 0.8400, 0.8393, 0.8302, 0.8385])

# 计算线性回归
slope, intercept, r_value, p_value, std_err = stats.linregress(demos, accuracy)

# 创建预测值
x_pred = np.linspace(0, 35, 100)
y_pred = slope * x_pred + intercept

# 计算最高点
if slope > 0:
    best_demo = 30  # 如果斜率为正，最大值在最右边
else:
    best_demo = 0   # 如果斜率为负，最大值在最左边

# 二次拟合
coeffs = np.polyfit(demos, accuracy, 2)
quad_fit = np.poly1d(coeffs)
best_demo_quad = -coeffs[1] / (2 * coeffs[0])
max_accuracy = quad_fit(best_demo_quad)

# 绘图
plt.figure(figsize=(10, 6))
plt.scatter(demos, accuracy, color='blue', label='Actual Data')
plt.plot(x_pred, y_pred, 'r--', label=f'Linear Fit (R² = {r_value**2:.4f})')
plt.plot(x_pred, quad_fit(x_pred), 'g-', label='Quadratic Fit')
plt.axvline(x=best_demo_quad, color='gray', linestyle=':', label=f'Best Demo Count ≈ {best_demo_quad:.1f}')
plt.axhline(y=max_accuracy, color='gray', linestyle=':')

plt.xlabel('Number of Demos')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Demos')
plt.legend()
plt.grid(True)
plt.show()

print(f"Linear correlation coefficient (R²): {r_value**2:.4f}")
print(f"Optimal number of demos (quadratic fit): {best_demo_quad:.1f}")
print(f"Predicted maximum accuracy: {max_accuracy:.4f}")