# ML101 - Linear Regression Tutorial

A beginner-friendly machine learning project designed for **empirical learning** of linear regression concepts. This repository provides multiple approaches to understand linear regression, from brute force methods to modern scikit-learn implementations, allowing you to experiment and record your own learning journey.

## Project Overview

This project contains multiple implementations to help you understand linear regression empirically:

- **Dataset Creation**: A Python script that generates synthetic linear regression data with randomization
- **Brute Force Implementation** (`lr01.py`): The most basic linear regression that works by systematically testing different slope and intercept values to find the best fit
- **Python Implementation** (`sales_tv.py`): A simplified Python script variant of the Jupyter notebook that uses scikit-learn for linear regression
- **Interactive Notebook** (`sales_tv.ipynb`): A Jupyter notebook with step-by-step analysis and visualizations

## Prerequisites

- Python 3.8 or higher
- `uv` package manager

## Setup Instructions

### 1. Environment Setup

First, create and activate a virtual environment:

```bash
# Using uv (recommended)
uv venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate

# Install required packages
uv pip install -U notebook ipykernel numpy pandas matplotlib scikit-learn

# Register the kernel for Jupyter
python -m ipykernel install --user --name ml101 --display-name "Python (ml101)"
```

### 2. Create the Dataset
(ignore if you already have the dataset created)

Generate a synthetic dataset with randomized linear relationships:

```bash
python create-dataset.py
```

This will create `dataset/sales_tv.csv` with 100 data points following a linear relationship (y = 3x + 4) with added noise for realism.

### 3. Run the Different Implementations

#### Option A: Brute Force Approach (Understanding the Fundamentals)
```bash
python lr01.py
```
This script demonstrates linear regression by systematically testing different slope and intercept values. Watch as it finds the best fit through brute force search. You can modify the search ranges in the script to experiment with different parameters.

#### Option B: Modern Scikit-learn Approach
```bash
python sales_tv.py
```
This script uses scikit-learn's LinearRegression for a more efficient solution. It includes interactive prediction capabilities where you can test custom values and compare with nearby data points.

#### Option C: Interactive Notebook for Analysis
```bash
jupyter notebook
```
Then open `sales_tv.ipynb` and run all cells. The notebook provides:

1. **Load and visualize the data** - Display scatter plots of the dataset
2. **Train a linear regression model** - Fit the model using scikit-learn
3. **Evaluate model performance** - Show R² score, MSE, and regression line
4. **Make predictions** - Demonstrate how to predict y values for new x inputs
5. **Interactive analysis** - Allow you to test custom x values and compare with nearby data points

## Learning Approach: Empirical Experimentation

This repository is designed as a **hands-on learning tool** for understanding linear regression empirically. Here's how you can use it to record and conduct your own experiments:

### Understanding the Different Approaches

1. **Start with Brute Force** (`lr01.py`):
   - Run this script to see how linear regression works "under the hood"
   - Watch as it systematically tests different slope and intercept combinations
   - Observe how the algorithm finds the best fit by minimizing error
   - Modify the search ranges and step sizes to see how it affects results

2. **Compare with Modern Methods** (`sales_tv.py`):
   - Run this script to see how scikit-learn solves the same problem
   - Compare the results with your brute force findings
   - Use the interactive prediction tool to test your understanding

3. **Deep Dive with Notebooks** (`sales_tv.ipynb`):
   - Step through the analysis cell by cell
   - Modify parameters and see immediate results
   - Add your own visualizations and experiments

### Recording Your Experiments

I have so far recorded the trained values of slope and intercept on paper, but would be nice to put them on wandb or some such tool so it is easier to see/compare. 

## Project Structure

```
ml101/
├── README.md                                    # This file
├── create-dataset.py                           # Dataset generation script
├── lr01.py                                     # Brute force linear regression implementation
├── sales_tv.py                                 # Simplified Python script (scikit-learn version)
├── sales_tv.ipynb                             # Main analysis notebook
├── nb01..ipynb                                # Simple plotting notebook
└── dataset/
    └── sales_tv.csv                           # Generated dataset
```



## Basic Troubleshooting

**Jupyter kernel issues**: Make sure you've registered the kernel and selected "Python (ml101)" in the notebook.

**Import errors**: Ensure all packages are installed in your virtual environment.

**Dataset not found**: Run `python create-dataset.py` first to generate the dataset.
