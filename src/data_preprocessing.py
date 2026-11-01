"""Imputacija i kodiranje uče isključivo iz trening podataka."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from src.feature_engineering import CarFeatures
from src.data_cleaning import TARGET

NUMERIC_FEATURES = ['car_age', 'mileage_km', 'mileage_per_year', 'engine_volume_liters', 'is_high_mileage']
CAT_FEATURES = ['make', 'model', 'brand_model', 'condition', 'fuel_type', 'color', 'transmission', 'drive_unit', 'segment']


def make_preprocessor():
    num = Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True)),
                    ('scale', StandardScaler())])
    cat = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore', min_frequency=10,
                                              max_categories=150, sparse_output=True))])
    return Pipeline([('features', CarFeatures()),
                     ('columns', ColumnTransformer([('numeric', num, NUMERIC_FEATURES),
                                                    ('categorical', cat, CAT_FEATURES)]))])
