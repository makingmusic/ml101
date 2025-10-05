# this will be a brute force implementation of multi-variable linear regression
# it will look and feel like the lr01.py but it will be for multi-variable linear regression
import numpy as np
import time
import os


f1, f2, f3, y = np.loadtxt("dataset/sales_multi.csv", delimiter=",", unpack=True)

f1best_slope, f1best_intercept, f1best_error = None, None, float("inf")

slope_minf1, slope_maxf1 = 2, 3
slope_minf2, slope_maxf2 = -2, 0
slope_minf3, slope_maxf3 = 1, 2

intercept_min, intercept_max = 6.38, 6.4

slope_steps = 10


'''Slope Values'''
slope_valuesf1 = [
    slope_minf1 + (slope_maxf1 - slope_minf1) / (slope_steps - 1) * i
    for i in range(slope_steps)
]

slope_valuesf2 = [
    slope_minf2 + (slope_maxf2 - slope_minf2) / (slope_steps - 1) * i
    for i in range(slope_steps)
]

slope_valuesf3 = [
    slope_minf3 + (slope_maxf3 - slope_minf3) / (slope_steps - 1) * i
    for i in range(slope_steps)
]

'''Intercept Values'''
intercept_values = [
    intercept_min + (intercept_max - intercept_min) / (slope_steps - 1) * i
    for i in range(slope_steps)
]

iteration = 0

#Start of brute force for f1
for slope in slope_valuesf1:
    for f2slope in slope_valuesf2:
        for f3slope in slope_valuesf3:
            for intercept in intercept_values:

                # Compute predicted y for each x
                error = 0
                for xi1, xi2, xi3, yi in zip(f1, f2, f3, y):
                    y_pred = slope * xi1 + f2slope * xi2 + f3slope * xi3 + intercept
                    error += (yi - y_pred) ** 2
                    iteration += 1

                # Update best values if current error is better
                if error < f1best_error:
                    f1best_slope, f2best_slope, f3best_slope, best_intercept, f1best_error = slope, f2slope, f3slope, intercept, error
                if iteration % 100000 == 0:
                    print(f"Iteration: {iteration}, Current error: {error}, Slopes: f1={slope}, f2={f2slope}, f3={f3slope}, Intercept: {intercept}")
                    
                    

# end of brute force for f1

print(f"Best slope for f1: {f1best_slope}, Best slope for f2: {f2best_slope}, Best slope for f3: {f3best_slope}, Best intercept: {best_intercept}, Best error: {f1best_error}")
