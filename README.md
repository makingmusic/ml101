# ml101

Machine Learning 101


First, we setup the repo:

uv venv .venv
source .venv/bin/activate
uv pip install -U notebook ipykernel numpy pandas matplotlib scikit-learn
python -m ipykernel install --user --name ml101 --display-name "Python (ml101)"
# now either:
jupyter notebook             # browser workflow
# or, in Cursor: open .ipynb and select kernel "Python (ml101)"