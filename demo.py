import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==============================================
# USER PARAMETERS (HARD-CODED TRANSFER FUNCTION COEFFICIENTS)
# ==============================================
# ADC transfer function: y = m_adc * x + c_adc
m_adc = -40.3254    # Replace with actual slope value
c_adc = 3309.0607    # Replace with actual intercept value

# Percentage transfer function: y = m_percent * x + c_percent
m_percent = 1.3288  # Replace with actual slope value
c_percent = 12.5993  # Replace with actual intercept value

# ==============================================
# DATA PROCESSING
# ==============================================

# Read CSV file (assuming tab-separated format)
df = pd.read_csv("demo.csv", 
                delimiter=",", 
                engine="python")  # ADD THIS LINE TO IGNORE INDEX COLUMN
print("DataFrame columns:", df.columns.tolist())

# change in adc
change_in_adc = c_adc - (70*(m_adc)+ c_adc)
change_in_percent = c_percent - (70*(m_percent)+ c_percent)


# Rest of your code remains the same...
# Calculate values from transfer functions
df['adc_calculated'] = m_adc * df['actual_input'] + c_adc
df['percent_calculated'] = m_percent * df['actual_input'] + c_percent

# Calculate percentage differences
df['adc_diff_pct'] = (df['ADC'] - df['adc_calculated']) / change_in_adc * 100
df['humidity_diff_pct'] = (df['Humidity'] - df['percent_calculated']) / change_in_percent * 100

# ==============================================
# PLOTTING
# ==============================================

# Create figure for ADC comparison
plt.figure(1, figsize=(10, 6))
x_range = np.linspace(df['actual_input'].min() - 1, df['actual_input'].max() + 1, 100)
plt.plot(x_range, m_adc * x_range + c_adc, 
         label=f'Transfer Function: y = {m_adc:.2f}x + {c_adc:.2f}', 
         color='blue')

# Plot calculated and measured ADC values
plt.scatter(df['actual_input'], df['adc_calculated'], 
            color='blue', marker='o', label='Calculated ADC', zorder=3)
plt.scatter(df['actual_input'], df['ADC'], 
            color='blue', marker='x', s=100, label='Measured ADC', zorder=3)

# Annotate differences
for idx, row in df.iterrows():
    plt.annotate(f"{row['adc_diff_pct']:+.1f}%", 
                 (row['actual_input'], row['ADC']),
                 textcoords="offset points",
                 xytext=(0, 10),
                 ha='center',
                 color='blue',
                 fontsize=8)

plt.title('ADC Transfer Function Validation')
plt.xlabel('actual_input (mL)')
plt.ylabel('ADC Value')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Create figure for Percentage comparison
plt.figure(2, figsize=(10, 6))
plt.plot(x_range, m_percent * x_range + c_percent, 
         label=f'Transfer Function: y = {m_percent:.2f}x + {c_percent:.2f}', 
         color='green')

# Plot calculated and measured humidity values
plt.scatter(df['actual_input'], df['percent_calculated'], 
            color='green', marker='o', label='Calculated Percentage', zorder=3)
plt.scatter(df['actual_input'], df['Humidity'], 
            color='green', marker='x', s=100, label='Measured Humidity', zorder=3)

# Annotate differences
for idx, row in df.iterrows():
    plt.annotate(f"{row['humidity_diff_pct']:+.1f}%", 
                 (row['actual_input'], row['Humidity']),
                 textcoords="offset points",
                 xytext=(0, 10),
                 ha='center',
                 color='green',
                 fontsize=8)

plt.title('Percentage Transfer Function Validation')
plt.xlabel('actual_input (mL)')
plt.ylabel('Percentage (%)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Show both plots
plt.tight_layout()
plt.show()