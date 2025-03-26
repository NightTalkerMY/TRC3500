# import pandas as pd

# # Define your ADC min and max values
# min_adc = 600   # Replace with actual minimum ADC value
# max_adc = 3600  # Replace with actual maximum ADC value

# # Load the CSV file with the correct delimiter
# df = pd.read_csv('adc_log_avg_20250320_192527.csv', delimiter=',')  # Try comma instead of tab

# print(df.columns)  # Debugging step

# # Strip whitespace from column names
# df.columns = df.columns.str.strip()

# # Convert humidity percentage back to ADC value (inverted relationship)
# df['adc_avg_measured'] = max_adc - ((df['humidity_percent'] / 100) * (max_adc - min_adc))

# # Round for readability
# df['adc_avg_measured'] = df['adc_avg_measured'].round(2)

# # Save back to CSV
# df.to_csv('dat2.csv', index=False, sep=',')  # Use the same delimiter as input file

# print("Conversion complete! Check dat1.csv.")


import pandas as pd

# Define max and min ADC values
ADC_MAX = 3600
ADC_MIN = 600

# Load CSV file with the correct delimiter
df = pd.read_csv("idk.csv", delimiter=",", engine="python")

# Strip spaces from column names (if any)
df.columns = df.columns.str.strip()

# Debug: Check if the columns are correctly parsed
print(df.head())

# Check if 'adc_avg_measured' column exists
if "adc_avg_measured" not in df.columns:
    print("Error: 'adc_avg_measured' column not found. Check your CSV format.")
else:
    # Convert 'adc_avg_measured' column to float
    df["adc_avg_measured"] = df["adc_avg_measured"].astype(float)

    # Recalculate the humidity percentage
    df["corrected_humidity_percent"] = round(100 - ((df["adc_avg_measured"] - ADC_MIN) / (ADC_MAX - ADC_MIN) * 100),2)

    # Save to a new CSV file
    df.to_csv("corrected_data.csv", index=False, sep=",")  # Save with the correct delimiter
    print("File saved as corrected_data.csv")
