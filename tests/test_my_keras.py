import pytest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from ..data_experts.my_keras import KerasRegressor, MyKerasRegressor, CustomSearchCV

# Fixture para generar datos de ejemplo
@pytest.fixture
def example_data():
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1)
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Prueba para KerasRegressor
def test_keras_regressor_fit_predict(example_data):
    X_train, X_test, y_train, y_test = example_data
    model = KerasRegressor(build_fn=lambda: MyKerasRegressor(input_dim=X_train.shape[1]).build_model())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    assert len(y_pred) == len(y_test)

# Prueba para MyKerasRegressor
def test_my_keras_regressor_fit_predict(example_data):
    X_train, X_test, y_train, y_test = example_data
    model = MyKerasRegressor(input_dim=X_train.shape[1], layers=2, units=64, activation='relu', optimizer='adam')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    assert len(y_pred) == len(y_test)
    assert model.model is not None  # Verificar que el modelo esté entrenado

# Prueba para la función score en MyKerasRegressor
def test_my_keras_regressor_score(example_data):
    X_train, X_test, y_train, y_test = example_data
    model = MyKerasRegressor(input_dim=X_train.shape[1], layers=2, units=64, activation='relu', optimizer='adam')
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    assert isinstance(score, float)
    assert -1 <= score <= 1  # El R2 debe estar en este rango

# Prueba para CustomSearchCV con GridSearch
def test_custom_search_cv_grid_search(example_data):
    X_train, X_test, y_train, y_test = example_data
    model = MyKerasRegressor(input_dim=X_train.shape[1])
    param_grid = {
        'layers': [1, 2],
        'units': [32, 64],
        'batch_size': [16, 32]
    }
    search = CustomSearchCV(model, param_grid, search_type='grid', cv=3, scoring='r2', verbose=0)
    search.fit(X_train, y_train, X_test, y_test)
    
    assert search.best_params_() is not None
    assert len(search.train_scores_) > 0
    assert len(search.test_scores_) > 0
    assert isinstance(search.results_(), dict)

# Prueba para detenerse temprano en CustomSearchCV
def test_custom_search_cv_early_stopping(example_data):
    X_train, X_test, y_train, y_test = example_data
    model = MyKerasRegressor(input_dim=X_train.shape[1])
    param_grid = {
        'layers': [1, 2],
        'units': [64],
        'batch_size': [32]
    }
    search = CustomSearchCV(model, param_grid, search_type='grid', cv=3, scoring='r2', verbose=0, stop_threshold=0.5)
    search.fit(X_train, y_train, X_test, y_test)
    
    assert max(search.train_scores_) >= 0.5

# Prueba para RandomizedSearch en CustomSearchCV
def test_custom_search_cv_random_search(example_data):
    X_train, X_test, y_train, y_test = example_data
    model = MyKerasRegressor(input_dim=X_train.shape[1])
    param_grid = {
        'layers': [1, 2, 3],
        'units': [32, 64, 128],
        'batch_size': [16, 32, 64]
    }
    search = CustomSearchCV(model, param_grid, search_type='random', cv=3, scoring='r2', verbose=0, n_iter=5)
    search.fit(X_train, y_train, X_test, y_test)
    
    assert search.best_params_() is not None
    assert len(search.train_scores_) > 0
    assert len(search.test_scores_) > 0

# Prueba para HalvingGridSearch en CustomSearchCV
def test_custom_search_cv_halving_search(example_data):
    X_train, X_test, y_train, y_test = example_data
    model = MyKerasRegressor(input_dim=X_train.shape[1])
    param_grid = {
        'layers': [1, 2, 3],
        'units': [32, 64, 128],
        'batch_size': [16, 32, 64]
    }
    search = CustomSearchCV(model, param_grid, search_type='halving', cv=3, scoring='r2', verbose=0)
    search.fit(X_train, y_train, X_test, y_test)
    
    assert search.best_params_() is not None
    assert len(search.train_scores_) > 0
    assert len(search.test_scores_) > 0
