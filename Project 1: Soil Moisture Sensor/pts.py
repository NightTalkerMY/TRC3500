import csv

# Given values
m = -40.3254  # Change as needed
c = 3309.0607  # Change as needed

delta_x = 1  # Separation of points (in mL)
range_x = 70  # Maximum x value (in mL)
margin = 2  # +- margin for ADC values

# Generate data
rows = [(x, m * x + c - margin, m * x + c + margin) for x in range(0, range_x + 1, delta_x)]

# Print data
print("mL, adc_lower, adc_upper")
for row in rows:
    print(f"{row[0]}, {row[1]}, {row[2]}")

# Save to CSV
filename = "adc_values.csv"
with open(filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["mL", "adc_lower", "adc_upper"])
    writer.writerows(rows)

print(f"Data saved to {filename}")
