import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from ..data_experts.learning_functions import ( 
    normalized_rmse, 
    normalized_mse, 
    r2,
    custom_scorer,
    custom_combined_scorer,
    add_categorical_random,
    add_numerical_random,
    partition_numerical_features,
    remove_outliers,
    prepare_data,
    build_pipeline,
    run_grid_search,
    calculate_metrics,
    compile_results,
    calculate_and_compile_metrics,
    plot_and_get_feature_importances
)

# Ejemplo de fixture para un DataFrame de prueba
@pytest.fixture
def example_dataframe():
    data = {
        'feature1': np.random.randn(100),
        'feature2': np.random.rand(100),
        'feature3': np.random.choice(['A', 'B', 'C'], size=100)
    }
    return pd.DataFrame(data)

# Prueba para normalized_rmse
def test_normalized_rmse(example_dataframe):
    y_true = example_dataframe['feature1']
    y_pred = y_true + np.random.randn(100) * 0.1
    result = normalized_rmse(y_true, y_pred)
    assert result >= 0

# Prueba para normalized_mse
def test_normalized_mse(example_dataframe):
    y_true = example_dataframe['feature1']
    y_pred = y_true + np.random.randn(100) * 0.1
    result = normalized_mse(y_true, y_pred)
    assert result >= 0

# Prueba para r2
def test_r2(example_dataframe):
    y_true = example_dataframe['feature1']
    y_pred = y_true + np.random.randn(100) * 0.1
    result = r2(y_true, y_pred)
    assert -1 <= result <= 1

# Prueba para add_categorical_random
def test_add_categorical_random(example_dataframe):
    result = add_categorical_random(example_dataframe)
    assert f'cat_random_0' in result.columns

# Prueba para add_numerical_random
def test_add_numerical_random(example_dataframe):
    result = add_numerical_random(example_dataframe)
    assert f'num_random_0' in result.columns

# Prueba para remove_outliers
def test_remove_outliers(example_dataframe):
    df_clean = remove_outliers(example_dataframe, ['feature1', 'feature2'])
    assert len(df_clean) <= len(example_dataframe)

# Prueba para prepare_data
def test_prepare_data(example_dataframe):
    categorical_features = ['feature3']
    quasinormal_features = ['feature1']
    non_normal_features = ['feature2']
    preprocessor = prepare_data(categorical_features, quasinormal_features, non_normal_features)
    assert preprocessor is not None

# Prueba para build_pipeline
def test_build_pipeline(example_dataframe):
    categorical_features = ['feature3']
    quasinormal_features = ['feature1']
    non_normal_features = ['feature2']
    model = RandomForestRegressor()
    pipeline = build_pipeline(model, categorical_features, quasinormal_features, non_normal_features)
    assert pipeline is not None

# Prueba para run_grid_search
def test_run_grid_search(example_dataframe):
    X_train, X_test, y_train, y_test = train_test_split(example_dataframe[['feature1', 'feature2']], example_dataframe['feature1'], test_size=0.2)
    pipeline = build_pipeline(RandomForestRegressor(), ['feature3'], ['feature1'], ['feature2'])
    param_grid = {'model__n_estimators': [10, 20]}
    best_model, grid_search = run_grid_search(pipeline, param_grid, X_train, y_train)
    assert best_model is not None

# Prueba para calculate_metrics
def test_calculate_metrics(example_dataframe):
    X_train, X_test, y_train, y_test = train_test_split(example_dataframe[['feature1', 'feature2']], example_dataframe['feature1'], test_size=0.2)
    model = RandomForestRegressor().fit(X_train, y_train)
    metrics = calculate_metrics(model, X_test, y_test)
    assert 'MSE' in metrics and 'R2' in metrics

# Prueba para compile_results
def test_compile_results():
    metrics = {"MSE": 0.1, "RMSE": 0.3, "MAE": 0.2, "R2": 0.9}
    params = {"n_estimators": 100}
    result_df = compile_results("RandomForest", metrics, params)
    assert not result_df.empty

# Prueba para calculate_and_compile_metrics
def test_calculate_and_compile_metrics(example_dataframe):
    X_train, X_test, y_train, y_test = train_test_split(example_dataframe[['feature1', 'feature2']], example_dataframe['feature1'], test_size=0.2)
    model = RandomForestRegressor().fit(X_train, y_train)
    params = {"n_estimators": 100}
    result_df = calculate_and_compile_metrics("RandomForest", model, X_train, X_test, y_train, y_test, params)
    assert not result_df.empty

# Prueba para plot_and_get_feature_importances
def test_plot_and_get_feature_importances(example_dataframe):
    X_train, X_test, y_train, y_test = train_test_split(example_dataframe[['feature1', 'feature2']], example_dataframe['feature1'], test_size=0.2)
    pipeline = build_pipeline(RandomForestRegressor(), ['feature3'], ['feature1'], ['feature2'])
    pipeline.fit(X_train, y_train)
    importances_df = plot_and_get_feature_importances(pipeline, X_train, N=5)
    assert not importances_df.empty
