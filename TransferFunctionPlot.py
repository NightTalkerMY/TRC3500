import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor

class TransferFunctionPlot:
    def __init__(self, filename):
        self.x_values = None
        self.y_values = None
        self.y_mean_values = None
        self.x_terminal_points = None
        self.y_terminal_points = None   
        self.m = None
        self.c = None
        self.choice = ""
        self.load_data(filename)
        pass

    def load_data(self, filename):
        # Read CSV file
        df = pd.read_csv(filename, header=1)

        # Extract X values (first column) and Y values (remaining columns)
        self.x_values = df.iloc[:, 0]  # all rows, first column
        self.y_values = df.iloc[:, 2:3] # all rows, all columns except the first

    def compute_mean(self):
        self.y_mean_values = self.y_values.mean(axis=1) # row-wise mean

    def terminal_points(self):
        self.x_terminal_points = pd.Series([self.x_values.iloc[0], self.x_values.iloc[-1]])
        self.y_terminal_points = pd.Series([self.y_mean_values.iloc[0], self.y_mean_values.iloc[-1]])
    
    def fit_terminal_points(self):
                
        # Compute slope (m) and intercept (c) of best-fit line
        x1, x2 = self.x_terminal_points
        y1, y2 = self.y_terminal_points
        self.m = (y2 - y1) / (x2 - x1) if x2 != x1 else 0  # Avoid division by zero
        self.c = y1 - self.m * x1

        # Print the slope and intercept of the best-fit line
        print(f"Slope     (m)                        : {sensor.m}")
        print(f"Intercept (c)                        : {sensor.c}")

    def fit_linear_regression(self):
        model = LinearRegression()
        x_reshaped = self.x_values.values.reshape(-1, 1)  # Convert 1D Series to 2D array
        model.fit(x_reshaped, self.y_mean_values)
        self.m = model.coef_[0]
        self.c = model.intercept_
        print("                                                    ")
        print(f"Slope     (m)                        : {self.m:.4f}")
        print(f"Intercept (c)                        : {self.c:.4f}")

    def fit_ransac_regression(self):
        model = RANSACRegressor(estimator=LinearRegression(), random_state=42, residual_threshold=1.0)
        x_reshaped = self.x_values.values.reshape(-1, 1)
        model.fit(x_reshaped, self.y_mean_values)

        # Extract best-fit model after removing outliers
        self.m = model.estimator_.coef_[0]
        self.c = model.estimator_.intercept_

        # Identify inliers and outliers
        inlier_mask = model.inlier_mask_
        outlier_mask = ~inlier_mask
        num_outliers = outlier_mask.sum()
        
        print(f"RANSAC Slope     (m)                 : {self.m:.4f}")
        print(f"RANSAC Intercept (c)                 : {self.c:.4f}")
        print(f"Detected {num_outliers} outliers removed.")


    def choice_best_fit(self, choice):
        self.choice = choice
        if self.choice == "R":
            self.fit_linear_regression()
        elif self.choice == "T":
            self.terminal_points()  # Ensure terminal points are computed
            self.fit_terminal_points()
        else:
            print("RANSAC Regression")
            self.fit_ransac_regression()


    def compute_max_deviation(self):
        # Compute the expected y values on the best-fit line for all x_values
        expected_y_values = self.m * self.x_values.values[:, np.newaxis] + self.c
        
        # Compute absolute deviation for each actual y value from the best-fit line
        deviations = np.abs(self.y_values.values - expected_y_values)

        # print(self.y_values.values)
        # print(expected_y_values)

        # Find the maximum deviation
        # print(f"All Deviation from Best Fit Line: {(deviations)}")
        self.max_deviation = np.max(deviations)
        print(f"Maximum Deviation from Best Fit Line : {self.max_deviation:.4f}")


    def nonLinearity(self):
        # Compute expected ideal response using terminal points
        expected_y_values = self.m * self.x_values + self.c

        # Compute absolute deviations from the ideal line
        deviations = np.abs(self.y_mean_values - expected_y_values)

        # Find maximum deviation
        self.max_deviation = np.max(deviations)

        # Compute Full-Scale Output (FSO)
        full_scale_output = self.y_mean_values.max() - self.y_mean_values.min()

        # Compute nonlinearity
        nonLinearity = (self.max_deviation / full_scale_output) * 100 if full_scale_output != 0 else 0
        print(f"Nonlinearity                         : {nonLinearity:.4f}%")


    def fullScaleInput(self):   
        fullScaleInput = self.x_values.max() - self.x_values.min()
        print(f"Full Scale Input                     : {fullScaleInput:.4f}")

    def compute_sensitivity(self):
        """Computes local sensitivity (dy/dx) instead of a single terminal-point slope."""
        dx = np.diff(self.x_values)  # Differences in x
        dy = np.diff(self.y_mean_values)  # Differences in y (mean values)
        local_sensitivity = dy / dx  # Compute local slopes
        max_sensitivity = np.max(np.abs(local_sensitivity))  # Find max sensitivity

        print(f"Maximum Sensitivity                  : {max_sensitivity:.4f} ")

    def repeatibility(self):
        # Compute max and min for each column
        each_result_max_y = self.y_values.max(axis=1)  # Series of max values per row
        each_result_min_y = self.y_values.min(axis=1)  # Series of min values per row

        range = each_result_max_y - each_result_min_y
        fullrange = self.y_values.max().max() - self.y_values.min().min()
        n_repeatability = (range/fullrange) * 100
        av_repeatability = n_repeatability.mean()
        print(f"Each Repeatability: \n{n_repeatability}")
        print(f"Average Repeatability                : {av_repeatability:.4f}%")




    def plot_data(self):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.x_values.repeat(self.y_values.shape[1]), self.y_values.values.flatten(), label="Data Points", color="blue", alpha=0.6)

        # Generate line values based on the chosen best-fit method
        best_fit_y = self.m * self.x_values + self.c
        plt.plot(self.x_values, best_fit_y, color='red', linestyle='-', label=f"Best Fit ({self.choice})")

        plt.xlabel("X Values")
        plt.ylabel("Y Values")
        plt.legend()
        plt.title("Scatter Plot with Best Fit Line")
        equation_text = f"y = {self.m:.2f}x + {self.c:.2f}"
        plt.text(self.x_values.min(), self.y_values.max().max(), equation_text, fontsize=12, color='red')
        plt.show()

    def runner(self):
        # Printing Statement 
        print("###### Transfer Function Analysis ######")

        # Compute mean of Y values
        self.compute_mean()

        # Compute terminal points and best-fit line
        self.choice_best_fit("R")

        # Plot data points and best-fit line
        self.plot_data()

        # Compute maximum deviation from best-fit line
        self.compute_max_deviation()
        
        # Compute nonlinearity
        self.nonLinearity()

        # Compute full scale input
        self.fullScaleInput()

        # Compute sensitivity 
        self.compute_sensitivity()

        # Compute repeatability
        self.repeatibility()

        # Printing Statement 
        print("\n##########################################\n")


if __name__ == "__main__":
    # Create an instance of the class
    sensor = TransferFunctionPlot("output_ADC_value/3/transfer_function.csv")

    # Run the analysis
    sensor.runner()