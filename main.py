import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Define the function to approximate
def f(x):
    return np.sin(x)

# Lagrange interpolation function
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

# Generate 10 different points
x_points = np.array([-(math.pi),-((2*math.pi)/3),-(math.pi/2),-(math.pi/4),-(math.pi/6),0,math.pi/6,math.pi/4,math.pi/2,math.pi])
y_points = f(x_points)

# Evaluate the interpolation on a finer grid for plotting
x_interp = np.linspace(-(math.pi), math.pi, 200)
y_interp = lagrange_interpolation(x_points, y_points, x_interp)
y_cubic = CubicSpline(x_points, y_points, bc_type='natural')

# Plot the original function and the Lagrange interpolation
plt.plot(x_interp, f(x_interp), label='Original Function: $sin(x)$')
plt.scatter(x_points, y_points, color='red', label='Interpolation Points')
plt.plot(x_interp, y_interp, '--', label='Lagrange Interpolation')
plt.plot(x_interp, y_cubic(x_interp), '--', label='Spline Interpolation')
plt.legend()
plt.title('Lagrange Interpolation of $sin(x)$')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.show()
