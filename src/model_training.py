"""Kandidati i njihovi parametri; python -m src.model_training pokreće ceo tok."""
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from src.data_preprocessing import make_preprocessor


def make_model(name):
    candidates = {
        'Median baseline': DummyRegressor(strategy='median'),
        'Ridge regression': Ridge(alpha=10.0, solver='lsqr'),
        'Decision tree': DecisionTreeRegressor(max_depth=16, min_samples_leaf=8, random_state=42),
        'Random forest': RandomForestRegressor(n_estimators=50, max_depth=20, min_samples_leaf=4,
                                               max_features=0.8, n_jobs=2, random_state=42),
    }
    return Pipeline([('preprocessor', make_preprocessor()), ('regressor', candidates[name])])


if __name__ == '__main__':
    from src.model_comparison import run_comparison
    run_comparison()
