"""Determinističko čišćenje bez učenja statistika iz test skupa."""
from pathlib import Path
import numpy as np
import pandas as pd

REFERENCE_YEAR = 2026
TARGET = 'priceUSD'
NUMERIC = ['year', 'mileage_km', 'engine_cm3']
CATEGORICAL = ['make', 'model', 'condition', 'fuel_type', 'color', 'transmission', 'drive_unit', 'segment']
RENAME = {'mileage(kilometers)': 'mileage_km', 'volume(cm3)': 'engine_cm3'}


def clean_features(frame):
    """Ista pravila pri treningu i predikciji; sumnjivi brojevi postaju NaN."""
    df = frame.copy()
    df.columns = df.columns.str.strip()
    df = df.rename(columns=RENAME)
    missing = set(NUMERIC + CATEGORICAL) - set(df.columns)
    if missing:
        raise ValueError(f'Nedostaju ulazne kolone: {sorted(missing)}')
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    df.loc[~df.year.between(1886, REFERENCE_YEAR) | df.year.mod(1).ne(0), 'year'] = np.nan
    df.loc[~df.mileage_km.between(0, 2_000_000), 'mileage_km'] = np.nan
    df.loc[~df.engine_cm3.between(100, 10_000), 'engine_cm3'] = np.nan
    for col in CATEGORICAL:
        df[col] = df[col].astype('string').str.strip().str.lower().replace('', pd.NA)
        # sklearn imputer očekuje numpy NaN umesto pandas.NA.
        df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
    return df[NUMERIC + CATEGORICAL]


def load_clean_data(path):
    raw = pd.read_csv(path)
    raw.columns = raw.columns.str.strip()
    if TARGET not in raw:
        raise ValueError('CSV mora sadržati priceUSD.')
    X = clean_features(raw)
    y = pd.to_numeric(raw[TARGET], errors='coerce')
    valid = y.notna() & np.isfinite(y) & y.gt(0)
    df = X.loc[valid].copy(); df[TARGET] = y.loc[valid]
    duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    info = {'raw_rows': len(raw), 'raw_columns': len(raw.columns),
            'missing_before': raw.isna().sum().to_dict(), 'invalid_target_rows': int((~valid).sum()),
            'duplicates_removed': duplicates, 'clean_rows': len(df),
            'missing_features_after_rules': X.isna().sum().to_dict(),
            'reference_year': REFERENCE_YEAR}
    return raw, df, info


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    _, df, info = load_clean_data(root / 'data/cars.csv')
    print(info)
    print(df.head().to_string(index=False))
