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
2. **Create dataset**: `python create-dataset.py` (generates `dataset/sales_tv.csv`)
3. **Run analysis**: `jupyter notebook` then open `sales_tv.ipynb`

## Key Files

- `create-dataset.py` - Generates synthetic linear regression dataset
- `sales_tv.ipynb` - Main analysis notebook with model training
- `lr01.py` - Basic linear regression script
- `dataset/sales_tv.csv` - Generated dataset (created by running create-dataset.py)

## Common Commands

```bash
# Activate environment (REQUIRED FIRST)
source myenv/bin/activate

# Generate dataset
python create-dataset.py

# Start Jupyter
jupyter notebook

# Test basic script
python lr01.py
```

## Notes

- The project uses `myenv` as the virtual environment name (not `.venv`)
- All Python commands will fail without activating the virtual environment first
- The dataset is generated with randomization to simulate real-world data
