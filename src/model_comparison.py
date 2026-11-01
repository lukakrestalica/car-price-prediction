"""Fer poređenje i čuvanje modela; python -m src.model_comparison."""
from pathlib import Path
import json
import hashlib
import time
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupShuffleSplit
from src.data_cleaning import load_clean_data, TARGET
from src.model_training import make_model
from src.model_evaluation import metrics, predict_prices

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = {'make':'volkswagen', 'model':'golf', 'year':2014, 'condition':'with mileage',
           'mileage(kilometers)':180000, 'fuel_type':'diesel', 'volume(cm3)':1600,
           'color':'black', 'transmission':'mechanics', 'drive_unit':'front-wheel drive', 'segment':'C'}


def run_comparison():
    reports = ROOT / 'reports'; reports.mkdir(exist_ok=True)
    models = ROOT / 'models'; models.mkdir(exist_ok=True)
    raw, df, cleaning = load_clean_data(ROOT / 'data/cars.csv')
    X, y = df.drop(columns=TARGET), df[TARGET]
    # Isti atributi sa različitim oglašenim cenama ostaju u istoj grupi.
    groups = pd.util.hash_pandas_object(X, index=False)
    dev_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(X, y, groups))
    train_local, val_local = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=42).split(
        X.iloc[dev_idx], y.iloc[dev_idx], groups.iloc[dev_idx]))
    train_idx, val_idx = dev_idx[train_local], dev_idx[val_local]
    assert set(groups.iloc[dev_idx]).isdisjoint(groups.iloc[test_idx])
    assert set(groups.iloc[train_idx]).isdisjoint(groups.iloc[val_idx])
    names = ['Median baseline', 'Ridge regression', 'Decision tree', 'Random forest']
    validation = []
    for name in names:
        t = time.perf_counter()
        model = make_model(name).fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = predict_prices(model, X.iloc[val_idx])
        row = {'model':name, **metrics(y.iloc[val_idx], pred), 'seconds':time.perf_counter()-t}
        validation.append(row)
        print(f"{name}: validation MAE={row['MAE_USD']:.2f} USD", flush=True)
    val = pd.DataFrame(validation)
    chosen = val.loc[val.model.ne('Median baseline')].sort_values(['MAE_USD','RMSE_USD'], kind='stable').iloc[0]['model']
    # Izbor je zaključan pre testiranja. Svi modeli potom koriste isti dev i test skup.
    test_results = []
    for name in names:
        model = make_model(name).fit(X.iloc[dev_idx], y.iloc[dev_idx])
        pred = predict_prices(model, X.iloc[test_idx])
        test_results.append({'model':name, **metrics(y.iloc[test_idx], pred)})
        if name == chosen:
            joblib.dump(model, models / 'car_price_model.joblib', compress=3)
            examples = X.iloc[test_idx].copy()
            examples['actual_price_USD'] = y.iloc[test_idx]
            examples['predicted_price_USD'] = pred
            examples['absolute_error_USD'] = abs(y.iloc[test_idx]-pred)
            examples.to_csv(reports / 'test_predictions.csv', index=False)
            example_price = float(predict_prices(model, pd.DataFrame([EXAMPLE]))[0])
    test = pd.DataFrame(test_results)
    val.to_csv(reports / 'validation_metrics.csv', index=False)
    test.to_csv(reports / 'test_metrics.csv', index=False)
    meta = {'author':'Luka Krestalica', 'source_sha256':hashlib.sha256((ROOT/'data/cars.csv').read_bytes()).hexdigest(),
            'cleaning':cleaning, 'selected_model':chosen, 'selection_metric':'validation MAE',
            'split_sizes':{'train':len(train_idx),'validation':len(val_idx),'test':len(test_idx),'saved_model_training':len(dev_idx)},
            'test_indices':test_idx.tolist(), 'example':EXAMPLE, 'example_prediction_USD':example_price,
            'reference_year':2026, 'versions':{'python':platform.python_version(),'numpy':np.__version__,
                                             'pandas':pd.__version__,'scikit-learn':sklearn.__version__}}
    (reports / 'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2),encoding='utf-8')
    print(test.to_string(index=False)); print('Selected:',chosen,'Example USD:',round(example_price,2))
    return meta, val, test


if __name__ == '__main__':
    run_comparison()
