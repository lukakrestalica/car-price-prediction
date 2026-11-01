"""Primer procene: python predict_price.py; opciono --input automobil.json."""
from pathlib import Path
import argparse
import json
import joblib
import pandas as pd
from src.model_comparison import EXAMPLE
from src.model_evaluation import predict_prices

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    args = parser.parse_args()
    car = json.loads(args.input.read_text(encoding='utf-8')) if args.input else EXAMPLE
    model = joblib.load(Path(__file__).resolve().parent / 'models/car_price_model.joblib')
    price = predict_prices(model, pd.DataFrame([car]))[0]
    print(f'Procenjena cena prema dostavljenom skupu: {price:,.2f} USD')
