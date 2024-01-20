import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline  # Library for cubic spline interpolation


# Define the function to approximate
def f(x):
    return np.sin(x)


# Lagrange's interpolation function
def lagrange_interpolation(x, y, x_interp):
    n = len(x)
    result = 0.0

    for j in range(n):
        term = y[j]
        for i in range(n):
            if i != j:
                term = term * (x_interp - x[i]) / (x[j] - x[i])
        result += term

    return result


x_points = np.array(
    [-math.pi, -((2 * math.pi) / 3), -(math.pi / 2), -(math.pi / 4), -(math.pi / 6), 0, math.pi / 6, math.pi / 4,
     math.pi / 2, math.pi])
y_points = f(x_points)

# Evaluation for lagrange interpolation
x_all = np.linspace(-math.pi, math.pi, 200)
y_interp = lagrange_interpolation(x_points, y_points, x_all)
# Evaluation for cubic spline interpolation
y_cubic = CubicSpline(x_points, y_points, bc_type='natural')
# Evaluation for least-squares interpolation
degree = 9
coefficients = np.polyfit(x_points, y_points, degree)
y_least_square = np.polyval(coefficients, x_all)

# Calculating the difference
y_diff_interp = np.abs(f(x_all) - y_interp)
y_diff_cubic = np.abs(f(x_all) - y_cubic(x_all))
y_diff_least_square = np.abs(f(x_all) - y_least_square)

# Plotting the charts
plt.subplot(2, 1, 1)
plt.plot(x_all, y_diff_interp, label='Lagrange error', color='orange')
plt.subplot(2, 1, 2)
plt.plot(x_all, y_diff_cubic, "--", label='Spline error')
plt.grid()
plt.legend()
plt.subplot(2, 1, 1)
plt.plot(x_all, y_diff_least_square, "--", label='Least Square error', color="black")
plt.grid()
plt.legend()
plt.title('Interpolation of $sin(x)$')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.show()
