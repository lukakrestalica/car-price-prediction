"""Karakteristike imaju istu referentnu godinu pri treningu i učitavanju."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from src.data_cleaning import clean_features, REFERENCE_YEAR


class CarFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, reference_year=REFERENCE_YEAR):
        self.reference_year = reference_year

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = clean_features(X)
        df['car_age'] = self.reference_year - df['year']
        # Za auto star nula godina koristimo delilac 1; nepoznata starost ostaje NaN.
        df['mileage_per_year'] = df.mileage_km / df.car_age.clip(lower=1)
        df['engine_volume_liters'] = df.engine_cm3 / 1000
        df['is_high_mileage'] = np.where(df.mileage_km.isna(), np.nan, (df.mileage_km >= 200000).astype(float))
        df['brand_model'] = (df.make.fillna('unknown') + ' | ' + df.model.fillna('unknown'))
        return df
