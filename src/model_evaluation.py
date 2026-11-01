"""MAE/RMSE su u USD, MSE u USD², R² je bez jedinice."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data_cleaning import load_clean_data, TARGET


def predict_prices(model, X):
    # Negativna cena nije smislen rezultat; pravilo je isto tokom evaluacije i upotrebe.
    return np.maximum(model.predict(X), 0)


def metrics(y, pred):
    mse = mean_squared_error(y, pred)
    return {'MAE_USD': mean_absolute_error(y, pred), 'MSE_USD2': mse,
            'RMSE_USD': float(np.sqrt(mse)), 'R2': r2_score(y, pred)}


def evaluate_saved():
    root = Path(__file__).resolve().parents[1]
    _, df, _ = load_clean_data(root / 'data/cars.csv')
    meta = json.loads((root / 'reports/metadata.json').read_text())
    test = df.iloc[meta['test_indices']]
    model = joblib.load(root / 'models/car_price_model.joblib')
    pred = predict_prices(model, test.drop(columns=TARGET))
    result = metrics(test[TARGET], pred)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    evaluate_saved()
