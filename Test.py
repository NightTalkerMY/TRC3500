import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Define models
def linear_model(x, m, b):
    return m * x + b

def exponential_model(x, a, b):
    return a * np.exp(b * x)

def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c

def cubic_model(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def quartic_model(x, a, b, c, d, e):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e

# Load data from CSV file
file_path = "DataFileDataOnly.csv"  # Update with the correct path if needed
data = pd.read_csv(file_path, skiprows=1)

# Extract ADC values and water volume
# water_volume = data.iloc[4:30, 0].values
# adc_values_theoritical = data.iloc[4:30, 1].values
# adc_values_raw1 = data.iloc[4:30, 2].values
# adc_values_raw2 = data.iloc[4:30, 3].values
# adc_values_raw3 = data.iloc[4:30, 4].values
# adc_values_mean = data.iloc[4:30, 5].values

water_volume = data.iloc[:, 0].values
adc_values_theoritical = data.iloc[:, 1].values
adc_values_raw1 = data.iloc[:, 2].values
adc_values_raw2 = data.iloc[:, 3].values
adc_values_raw3 = data.iloc[:, 4].values
adc_values_mean = data.iloc[:, 5].values

plt.plot(water_volume, adc_values_theoritical, label="Theoritical ADC", linestyle="--", color="black")
plt.scatter(water_volume, adc_values_raw1, label="Raw ADC 1", color="red")
plt.scatter(water_volume, adc_values_raw2, label="Raw ADC 2", color="green")
plt.scatter(water_volume, adc_values_raw3, label="Raw ADC 3", color="blue")
plt.scatter(water_volume, adc_values_mean, label="Mean ADC", color="purple")
plt.xlabel("Water Volume (ml)")
plt.ylabel("ADC Value")
plt.legend()
plt.grid(True)
plt.show()

# Fit Linear Model
params_linear, _ = curve_fit(linear_model, adc_values_mean, adc_values_theoritical)
m, b = params_linear
y_pred_linear = linear_model(adc_values_mean, m, b)

# Fit Quadratic Model
params_quad, _ = curve_fit(quadratic_model, adc_values_mean, adc_values_theoritical)
a_quad, b_quad, c_quad = params_quad
y_pred_quad = quadratic_model(adc_values_mean, a_quad, b_quad, c_quad)

# Fit Cubic Model
params_cubic, _ = curve_fit(cubic_model, adc_values_mean, adc_values_theoritical)
a_cubic, b_cubic, c_cubic, d_cubic = params_cubic
y_pred_cubic = cubic_model(adc_values_mean, a_cubic, b_cubic, c_cubic, d_cubic)

# Fit Quartic Model
params_quartic, _ = curve_fit(quartic_model, adc_values_mean, adc_values_theoritical)
a_quart, b_quart, c_quart, d_quart, e_quart = params_quartic
y_pred_quartic = quartic_model(adc_values_mean, a_quart, b_quart, c_quart, d_quart, e_quart)

# Print fitted equations
print(f"Linear Model: f(x) = {m:.4f}x + {b:.4f}")
print(f"Quadratic Model: f(x) = {a_quad:.4f}x² + {b_quad:.4f}x + {c_quad:.4f}")
print(f"Cubic Model: f(x) = {a_cubic:.8f}x³ + {b_cubic:.8f}x² + {c_cubic:.8f}x + {d_cubic:.8f}")
print(f"Quartic Model: f(x) = {a_quart:.8f}x⁴ + {b_quart:.8f}x³ + {c_quart:.8f}x² + {d_quart:.8f}x + {e_quart:.8f}")
print('\n')

#--------------------------Mean ADC------------------------------------
# Compute R² score for each model
r2_raw = r2_score(adc_values_theoritical, adc_values_mean)
r2_linear = r2_score(adc_values_theoritical, y_pred_linear)
r2_quad = r2_score(adc_values_theoritical, y_pred_quad)
r2_cubic = r2_score(adc_values_theoritical, y_pred_cubic)
r2_quartic = r2_score(adc_values_theoritical, y_pred_quartic)

# Print R² scores
print(f"R² Score (Raw Data): {r2_raw:.4f}")
print(f"R² Score (Linear Fit): {r2_linear:.4f}")
print(f"R² Score (Quadratic Fit): {r2_quad:.4f}")
print(f"R² Score (Cubic Fit): {r2_cubic:.4f}")
print(f"R² Score (Quartic Fit): {r2_quartic:.4f}")
print("\n")

# Plot data and fitted models
plt.scatter(adc_values_mean, adc_values_theoritical, label="Data", color="black")
plt.plot(adc_values_mean, y_pred_linear, label="Linear Fit", linestyle="--", color="red")
plt.plot(adc_values_mean, y_pred_quad, label="Quadratic Fit", linestyle="--", color="green")
plt.plot(adc_values_mean, y_pred_cubic, label="Cubic Fit", linestyle="--", color="blue")
plt.plot(adc_values_mean, y_pred_quartic, label="Quartic Fit", linestyle="--", color="purple")
plt.xlabel("Mean ADC Value")
plt.ylabel("Theoretical ADC Value")
plt.legend()
plt.grid(True)
plt.show()

#--------------------------Raw 1------------------------------------
# Compute predicted values using raw1 ADC values
y_pred_linear_raw1 = linear_model(adc_values_raw1, m, b)
y_pred_quad_raw1 = quadratic_model(adc_values_raw1, a_quad, b_quad, c_quad)
y_pred_cubic_raw1 = cubic_model(adc_values_raw1, a_cubic, b_cubic, c_cubic, d_cubic)
y_pred_quartic_raw1 = quartic_model(adc_values_raw1, a_quart, b_quart, c_quart, d_quart, e_quart)

# Compute R² scores
r2_linear_raw1 = r2_score(adc_values_theoritical, y_pred_linear_raw1)
r2_quad_raw1 = r2_score(adc_values_theoritical, y_pred_quad_raw1)
r2_cubic_raw1 = r2_score(adc_values_theoritical, y_pred_cubic_raw1)
r2_quartic_raw1 = r2_score(adc_values_theoritical, y_pred_quartic_raw1)

# Print R² scores
print(f"R² Score (Linear Fit - Raw1): {r2_linear_raw1:.4f}")
print(f"R² Score (Quadratic Fit - Raw1): {r2_quad_raw1:.4f}")
print(f"R² Score (Cubic Fit - Raw1): {r2_cubic_raw1:.4f}")
print(f"R² Score (Quartic Fit - Raw1): {r2_quartic_raw1:.4f}")
print("\n")

# Plot data and fitted models
plt.scatter(adc_values_raw1, adc_values_theoritical, label="Data", color="black")
plt.plot(adc_values_raw1, y_pred_linear_raw1, label="Linear Fit", linestyle="--", color="red")
plt.plot(adc_values_raw1, y_pred_quad_raw1, label="Quadratic Fit", linestyle="--", color="green")
plt.plot(adc_values_raw1, y_pred_cubic_raw1, label="Cubic Fit", linestyle="--", color="blue")
plt.plot(adc_values_raw1, y_pred_quartic_raw1, label="Quartic Fit", linestyle="--", color="purple")
plt.xlabel("Raw ADC Value 1")
plt.ylabel("Theoretical ADC Value")
plt.legend()
plt.grid(True)
plt.show()

#--------------------------Raw 2------------------------------------
# Compute predicted values using raw2 ADC values
y_pred_linear_raw2 = linear_model(adc_values_raw2, m, b)
y_pred_quad_raw2 = quadratic_model(adc_values_raw2, a_quad, b_quad, c_quad)
y_pred_cubic_raw2 = cubic_model(adc_values_raw2, a_cubic, b_cubic, c_cubic, d_cubic)
y_pred_quartic_raw2 = quartic_model(adc_values_raw2, a_quart, b_quart, c_quart, d_quart, e_quart)

# Compute R² scores
r2_linear_raw2 = r2_score(adc_values_theoritical, y_pred_linear_raw2)
r2_quad_raw2 = r2_score(adc_values_theoritical, y_pred_quad_raw2)
r2_cubic_raw2 = r2_score(adc_values_theoritical, y_pred_cubic_raw2)
r2_quartic_raw2 = r2_score(adc_values_theoritical, y_pred_quartic_raw2)

# Print R² scores
print(f"R² Score (Linear Fit - Raw2): {r2_linear_raw2:.4f}")
print(f"R² Score (Quadratic Fit - Raw2): {r2_quad_raw2:.4f}")
print(f"R² Score (Cubic Fit - Raw2): {r2_cubic_raw2:.4f}")
print(f"R² Score (Quartic Fit - Raw2): {r2_quartic_raw2:.4f}")
print("\n")

# Plot data and fitted models
plt.scatter(adc_values_raw2, adc_values_theoritical, label="Data", color="black")
plt.plot(adc_values_raw2, y_pred_linear_raw2, label="Linear Fit", linestyle="--", color="red")
plt.plot(adc_values_raw2, y_pred_quad_raw2, label="Quadratic Fit", linestyle="--", color="green")
plt.plot(adc_values_raw2, y_pred_cubic_raw2, label="Cubic Fit", linestyle="--", color="blue")
plt.plot(adc_values_raw2, y_pred_quartic_raw2, label="Quartic Fit", linestyle="--", color="purple")
plt.xlabel("Raw ADC Value 2")
plt.ylabel("Theoretical ADC Value")
plt.legend()
plt.grid(True)
plt.show()

#--------------------------Raw 3------------------------------------
# Compute predicted values using raw3 ADC values
y_pred_linear_raw3 = linear_model(adc_values_raw3, m, b)
y_pred_quad_raw3 = quadratic_model(adc_values_raw3, a_quad, b_quad, c_quad)
y_pred_cubic_raw3 = cubic_model(adc_values_raw3, a_cubic, b_cubic, c_cubic, d_cubic)
y_pred_quartic_raw3 = quartic_model(adc_values_raw3, a_quart, b_quart, c_quart, d_quart, e_quart)

# Compute R² scores
r2_linear_raw3 = r2_score(adc_values_theoritical, y_pred_linear_raw3)
r2_quad_raw3 = r2_score(adc_values_theoritical, y_pred_quad_raw3)
r2_cubic_raw3 = r2_score(adc_values_theoritical, y_pred_cubic_raw3)
r2_quartic_raw3 = r2_score(adc_values_theoritical, y_pred_quartic_raw3)

# Print R² scores
print(f"R² Score (Linear Fit - Raw3): {r2_linear_raw3:.4f}")
print(f"R² Score (Quadratic Fit - Raw3): {r2_quad_raw3:.4f}")
print(f"R² Score (Cubic Fit - Raw3): {r2_cubic_raw3:.4f}")
print(f"R² Score (Quartic Fit - Raw3): {r2_quartic_raw3:.4f}")
print("\n")

# Plot data and fitted models
plt.scatter(adc_values_raw3, adc_values_theoritical, label="Data", color="black")
plt.plot(adc_values_raw3, y_pred_linear_raw3, label="Linear Fit", linestyle="--", color="red")
plt.plot(adc_values_raw3, y_pred_quad_raw3, label="Quadratic Fit", linestyle="--", color="green")
plt.plot(adc_values_raw3, y_pred_cubic_raw3, label="Cubic Fit", linestyle="--", color="blue")
plt.plot(adc_values_raw3, y_pred_quartic_raw3, label="Quartic Fit", linestyle="--", color="purple")
plt.xlabel("Raw ADC Value 3")
plt.ylabel("Theoretical ADC Value")
plt.legend()
plt.grid(True)
plt.show()

# Create a DataFrame with ADC values and predicted outputs
output_df = pd.DataFrame({
    "ADC Mean": adc_values_mean,
    "Theoretical ADC": adc_values_theoritical,
    "Linear Fit": y_pred_linear,
    "Quadratic Fit": y_pred_quad,
    "Cubic Fit": y_pred_cubic,
    "Quartic Fit": y_pred_quartic
})

# Save to CSV file
output_file = "calibrated_data.csv"
output_df.to_csv(output_file, index=False)

print(f"Predicted data saved to {output_file}")

