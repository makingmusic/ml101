# ML101 Project - Agent Instructions

## Virtual Environment Setup

**CRITICAL**: Before running any Python commands in this project, you MUST activate the virtual environment first:

```bash
source myenv/bin/activate
```

The virtual environment is located at `myenv/` and contains all the required packages:
- notebook
- ipykernel  
- numpy
- pandas
- matplotlib
- scikit-learn

## Project Workflow

1. **Always activate venv first**: `source myenv/bin/activate`
2. **Create dataset**: `python utils/create-dataset.py` (generates `dataset/sales_tv.csv`)
3. **Run analysis**: Choose from multiple approaches:
   - **Brute force**: `python lr01.py`
   - **Scikit-learn script**: `python sales_tv.py`
   - **Interactive notebook**: `jupyter notebook` then open `sales_tv.ipynb`

## Key Files

- `utils/create-dataset.py` - Generates synthetic linear regression dataset (moved from root)
- `sales_tv.ipynb` - Main analysis notebook with model training
- `sales_tv.py` - **NEW**: Standalone Python script with scikit-learn implementation and interactive prediction tool
- `lr01.py` - Brute force linear regression implementation
- `utils/test_if_nb_is_working.ipynb` - **NEW**: Test notebook for Jupyter functionality
- `dataset/sales_tv.csv` - Generated dataset (created by running utils/create-dataset.py)

## Common Commands

```bash
# Activate environment (REQUIRED FIRST)
source myenv/bin/activate

# Generate dataset (note: moved to utils/ directory)
python utils/create-dataset.py

# Run brute force linear regression
python lr01.py

# Run scikit-learn implementation with interactive predictions
python sales_tv.py

# Start Jupyter
jupyter notebook
```

## Implementation Options

### 1. Brute Force Approach (`lr01.py`)
- Demonstrates linear regression fundamentals
- Systematically tests different slope and intercept values
- Good for understanding the underlying mathematics

### 2. Modern Scikit-learn Approach (`sales_tv.py`)
- Uses scikit-learn's LinearRegression
- Includes interactive prediction capabilities
- Allows testing custom values and comparing with nearby data points
- Provides comprehensive analysis and visualizations

### 3. Interactive Notebook (`sales_tv.ipynb`)
- Step-by-step analysis with visualizations
- Cell-by-cell execution for learning
- Easy to modify and experiment with

## Notes

- The project uses `myenv` as the virtual environment name (not `.venv`)
- All Python commands will fail without activating the virtual environment first
- The dataset is generated with randomization to simulate real-world data
- The `create-dataset.py` script has been moved to the `utils/` directory
- The project now offers multiple learning approaches for different skill levels
