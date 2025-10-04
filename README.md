# ML101 - Linear Regression Tutorial

A beginner-friendly machine learning project that demonstrates linear regression using scikit-learn. This project walks you through creating a synthetic dataset, training a linear regression model, and making predictions.

## Project Overview

This project contains:
- **Dataset Creation**: A Python script that generates synthetic linear regression data with randomization
- **Model Training**: A Jupyter notebook that trains a linear regression model and visualizes results
- **Interactive Predictions**: Tools to test the model with custom inputs

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

### 3. Train the Model and Explore Results

Launch Jupyter Notebook and run the analysis:

```bash
jupyter notebook
```

Then open `sales_tv.ipynb` and run all cells. The notebook will:

1. **Load and visualize the data** - Display scatter plots of the dataset
2. **Train a linear regression model** - Fit the model using scikit-learn
3. **Evaluate model performance** - Show R² score, MSE, and regression line
4. **Make predictions** - Demonstrate how to predict y values for new x inputs
5. **Interactive analysis** - Allow you to test custom x values and compare with nearby data points

## Project Structure

```
ml101/
├── README.md                                    # This file
├── create-dataset.py                           # Dataset generation script
├── lr01.py                                     # Basic linear regression script
├── sales_tv.ipynb                             # Main analysis notebook
├── nb01..ipynb                                # Simple plotting notebook
└── dataset/
    └── sales_tv.csv                           # Generated dataset
```



## Basic Troubleshooting

**Jupyter kernel issues**: Make sure you've registered the kernel and selected "Python (ml101)" in the notebook.

**Import errors**: Ensure all packages are installed in your virtual environment.

**Dataset not found**: Run `python create-dataset.py` first to generate the dataset.
