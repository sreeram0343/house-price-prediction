# House Price Prediction

A simple Python project for predicting house prices using a linear regression model.

## Project structure

- `main.py` - Loads the housing dataset, trains a `LinearRegression` model, evaluates it on test data, and makes a sample prediction.
- `data/House Price Prediction Dataset.csv` - The dataset used for training and evaluation.
- `venv/` - Python virtual environment for project dependencies.

## Requirements

- Python 3.8+
- `pandas`
- `scikit-learn`

## Setup

1. Activate the virtual environment:

```powershell
& .\venv\Scripts\Activate.ps1
```

2. Install dependencies if needed:

```powershell
pip install pandas scikit-learn
```

## Run the project

```powershell
python main.py
```

The script will:

- load the dataset
- train a linear regression model
- evaluate performance with MAE and R²
- print a sample house price prediction

## Notes

- The model currently uses the features `Area`, `Bedrooms`, and `Bathrooms`.
- Update `main.py` to add more features, preprocess data, or use a different model.
