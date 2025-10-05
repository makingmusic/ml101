# this script will load the dataset and train a linear regression model using scikit-learn
# import matplotlib.pyplot as plt  # Uncomment if you want to see the plot
import numpy as np
import time
import os

# this is the dataset
# x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# y = np.array([2, 4, 6, 8, 200, 12, 14, -1600, 18, 20])

# load the values of x and y from the csv file named dataset/sales_tv.csv
x, y = np.loadtxt("dataset/sales_tv.csv", delimiter=",", unpack=True)

# plot the data for x and y
# plt.scatter(x, y)
# plt.show()

# visualization related functions


def clear_screen():
    """Clear the terminal screen"""
    os.system("cls" if os.name == "nt" else "clear")

def print_training_header():
    """Print the header for the training visualization"""
    print("=" * 80)
    print("LINEAR REGRESSION TRAINING - TEXT-BASED VISUALIZATION")
    print("=" * 80)


def print_training_progress(
    iteration,
    total_iterations,
    current_slope,
    current_intercept,
    best_slope,
    best_intercept,
    best_error,
    current_error,
):
    """Print the current training progress"""
    clear_screen()
    print_training_header()

    # Progress bar
    progress = iteration / total_iterations
    bar_length = 50
    filled_length = int(bar_length * progress)
    bar = "█" * filled_length + "░" * (bar_length - filled_length)

    print(f"\nPROGRESS: [{bar}] {progress:.1%} ({iteration}/{total_iterations})")
    print("-" * 80)

    # Current iteration info
    print(f"ITERATION #{iteration}")
    print(f"Current Slope:     {current_slope:8.4f}")
    print(f"Current Intercept: {current_intercept:8.4f}")
    print(f"Current Error:     {current_error:12.6f}")
    print("-" * 80)

    # Best values found so far
    print("BEST VALUES FOUND SO FAR:")
    print(f"Best Slope:        {best_slope:8.4f}")
    print(f"Best Intercept:    {best_intercept:8.4f}")
    print(f"Best Error:        {best_error:12.6f}")
    print("-" * 80)

    # Improvement indicator
    if iteration > 1:
        improvement = (
            "🟢 IMPROVEMENT!" if current_error < best_error else "🔴 No improvement"
        )
        print(f"Status: {improvement}")

    print("=" * 80)
    time.sleep(VISUALIZATION_DELAY)  # Configurable delay for visualization effect


best_slope, best_intercept, best_error = None, None, float("inf")

# Visualization settings
VISUALIZATION_DELAY = 0.00  # Seconds to pause between iterations
# Adjust this value to control visualization speed:
# - 0.1 = fast (good for quick overview)
# - 0.5 = medium (good for following progress)
# - 1.0 = slow (good for detailed observation)
# - 0.0 = no delay (fastest, but hard to follow)

# Define the range for slope and intercept
slope_min, slope_max = 2.75, 2.77
intercept_min, intercept_max = 8.23, 8.25

# Configurable number of points to check for slope and intercept
num_slope_points = 100  # Number of distinct slope values to check
num_intercept_points = num_slope_points  # Number of distinct intercept values to check

# Generate values for slope from slope_min to slope_max (inclusive)
slope_values = [
    slope_min + (slope_max - slope_min) / (num_slope_points - 1) * i
    for i in range(num_slope_points)
]
# Generate values for intercept from intercept_min to intercept_max (inclusive)
intercept_values = [
    intercept_min + (intercept_max - intercept_min) / (num_intercept_points - 1) * i
    for i in range(num_intercept_points)
]

# Calculate total iterations for progress tracking
total_iterations = len(slope_values) * len(intercept_values)
iteration = 0

print_training_header()
print(f"Starting training with {total_iterations} total iterations...")
print(f"Visualization delay: {VISUALIZATION_DELAY} seconds per iteration")
print("Press Ctrl+C to stop training early")

for slope in slope_values:
    for intercept in intercept_values:
        iteration += 1

        # Compute predicted y for each x
        error = 0
        for xi, yi in zip(x, y):
            y_pred = slope * xi + intercept
            error += (yi - y_pred) ** 2

        # Update best values if current error is better
        if error < best_error:
            best_slope, best_intercept, best_error = slope, intercept, error

        # Print training progress
        print_training_progress(
            iteration,
            total_iterations,
            slope,
            intercept,
            best_slope,
            best_intercept,
            best_error,
            error,
        )


# Final results
clear_screen()
print_training_header()
print("\n🎉 TRAINING COMPLETED! 🎉")
print("-" * 80)
print("FINAL RESULTS:")
print(f"Best Error:         {best_error:.6f}")
print(f"Best Slope:         {best_slope:.4f}")
print(f"Best Intercept:     {best_intercept:.4f}")
print(f"Final Equation:     y = {best_slope:.2f}x + {best_intercept:.2f}")
print("-" * 80)
print("Training completed successfully!")
print("=" * 80)

# plot the original data and the best fit line
# plt.scatter(x, y)
# plt.plot(x, best_slope * x + best_intercept, color="red")
# plt.show()
