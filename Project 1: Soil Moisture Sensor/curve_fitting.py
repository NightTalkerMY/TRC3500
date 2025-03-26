import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# Define models
def linear_model(x, m, b):
    return m * x + b

def exponential_model(x, a, b):
    return a * np.exp(b * x)

def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c

def power4_model(x, a, b, c, d, e):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e

# Load CSV file
file_path = "voltage_data.csv"  # Change this to your actual CSV file path
df = pd.read_csv(file_path, header=None)

# Extract x and y values
x_data = df[0].values
y_data = df[1].values

### LINEAR FIT ###
params_linear, _ = curve_fit(linear_model, x_data, y_data)
m, b = params_linear
y_pred_linear = linear_model(x_data, m, b)
sse_linear = np.sum((y_data - y_pred_linear) ** 2)
r2_linear = r2_score(y_data, y_pred_linear)

### EXPONENTIAL FIT ###
initial_guess = (1, 0.1)  # Initial guess for curve fitting
params_exp, _ = curve_fit(exponential_model, x_data, y_data, p0=initial_guess, maxfev=10000)
a, b_exp = params_exp
y_pred_exp = exponential_model(x_data, a, b_exp)
sse_exp = np.sum((y_data - y_pred_exp) ** 2)
r2_exp = r2_score(y_data, y_pred_exp)

### QUADRATIC FIT ###
params_quad, _ = curve_fit(quadratic_model, x_data, y_data)
a_quad, b_quad, c_quad = params_quad
y_pred_quad = quadratic_model(x_data, a_quad, b_quad, c_quad)
sse_quad = np.sum((y_data - y_pred_quad) ** 2)
r2_quad = r2_score(y_data, y_pred_quad)

### POWER 4 FIT ###
params_power4, _ = curve_fit(power4_model, x_data, y_data)
a_p4, b_p4, c_p4, d_p4, e_p4 = params_power4
y_pred_power4 = power4_model(x_data, a_p4, b_p4, c_p4, d_p4, e_p4)
sse_power4 = np.sum((y_data - y_pred_power4) ** 2)
r2_power4 = r2_score(y_data, y_pred_power4)

# Print results
print(f"Linear Model: f(x) = {m:.4f}x + {b:.4f}")
print(f"SSE (Linear): {sse_linear:.2f}")
print(f"R² (Linear): {r2_linear:.2f}\n")

print(f"Exponential Model: f(x) = {a:.4f} * e^({b_exp:.4f}x)")
print(f"SSE (Exponential): {sse_exp:.2f}")
print(f"R² (Exponential): {r2_exp:.2f}\n")

print(f"Quadratic Model: f(x) = {a_quad:.4f}x² + {b_quad:.4f}x + {c_quad:.4f}")
print(f"SSE (Quadratic): {sse_quad:.2f}")
print(f"R² (Quadratic): {r2_quad:.2f}\n")

print(f"Power 4 Model: f(x) = {a_p4:.4f}x⁴ + {b_p4:.4f}x³ + {c_p4:.4f}x² + {d_p4:.4f}x + {e_p4:.4f}")
print(f"SSE (Power 4): {sse_power4:.2f}")
print(f"R² (Power 4): {r2_power4:.2f}")

# Plot results
plt.scatter(x_data, y_data, label="Data", color="black")
plt.plot(x_data, y_pred_linear, label="Linear Fit", linestyle="--", color="red")
plt.plot(x_data, y_pred_exp, label="Exponential Fit", linestyle="--", color="blue")
plt.plot(x_data, y_pred_quad, label="Quadratic Fit", linestyle="--", color="green")
plt.plot(x_data, y_pred_power4, label="Power 4 Fit", linestyle="--", color="purple")

plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.title("Curve Fitting Models")
plt.show()