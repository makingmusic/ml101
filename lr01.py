# this script will load the dataset and train a linear regression model using scikit-learn

# this is the dataset
x = [1, 2, 3]
y = [2, 4, 6]

best_slope, best_intercept, best_error = None, None, float("inf")

# Define the range for slope and intercept
slope_min, slope_max = 0, 3
intercept_min, intercept_max = -2, 2

# Configurable number of points to check for slope and intercept
num_slope_points = 10  # Number of distinct slope values to check
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

for slope in slope_values:
    for intercept in intercept_values:
        # Compute predicted y for each x
        error = 0
        for xi, yi in zip(x, y):
            y_pred = slope * xi + intercept
            error += (yi - y_pred) ** 2
        if error < best_error:
            best_slope, best_intercept, best_error = slope, intercept, error


print(f"Best error: {best_error:.6f}")

print(f"equation looks like this: y = {best_slope:.2f}x + {best_intercept:.2f}")
