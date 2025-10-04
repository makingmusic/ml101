import numpy as np


# this script helps me to create a dataset
def create_single_variable_linear_regression_dataset(filename, n_samples=100):
    print(f"Creating dataset with {n_samples} samples")
    # we will use a simple slope and intercept to generate the dataset
    slope = 3
    intercept = 4
    # we will introduce some random noise to the dataset by randomizing the y values

    # use numpy to generate the dataset
    x = np.linspace(0, 100, n_samples)
    # Add small randomness to the slope for each sample
    random_slopes = slope + np.random.normal(0, 0.5, n_samples)
    y = random_slopes * x + intercept + np.random.normal(0, 1, n_samples)
    # we will save the dataset to a file
    np.savetxt(filename, np.column_stack((x, y)), delimiter=",", fmt="%.1f")
    print(f"Dataset saved to {filename}")


if __name__ == "__main__":
    create_single_variable_linear_regression_dataset("dataset/sales_tv.csv")
