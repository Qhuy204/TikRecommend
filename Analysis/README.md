# Analysis

Data analysis notebooks for the Tiki Recommendation System.

## Notebooks

1. **data_overview.py** - Dataset statistics and distributions
2. **user_analysis.py** - User behavior analysis
3. **item_analysis.py** - Product/item analysis
4. **model_comparison.py** - Model performance comparison

## Usage

```bash
cd Analysis

# Run as Python scripts
python data_overview.py

# Or convert to Jupyter notebooks
pip install jupytext
jupytext --to notebook *.py
jupyter notebook
```

## Requirements

```bash
pip install matplotlib seaborn pandas numpy plotly
```
