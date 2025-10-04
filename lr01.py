# this script will load the dataset and train a linear regression model using scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# load the dataset
df = pd.read_csv("dataset/sales_tv.csv")

# print the first 5 rows of the dataset
print(df.head())

# print the columns of the dataset
print(df.columns)

# split the dataset into training and testing sets
