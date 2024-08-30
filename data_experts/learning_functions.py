import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from bayes_opt import BayesianOptimization

# 1. Función para preparar los datos
def prepare_data(categorical_features: List[str], numerical_features: List[str]) -> ColumnTransformer:
    """
    Prepara los datos para el entrenamiento del modelo aplicando transformaciones a las características categóricas y numéricas.

    Args:
        categorical_features (List[str]): Lista de nombres de columnas categóricas.
        numerical_features (List[str]): Lista de nombres de columnas numéricas.

    Returns:
        ColumnTransformer: Transformador de columnas preparado para el preprocesamiento de datos.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_features)
        ])
    return preprocessor

# 2. Función para construir el pipeline
def build_pipeline(model: Any, categorical_features: List[str], numerical_features: List[str]) -> Pipeline:
    """
    Construye un pipeline que incluye la preparación de los datos y el modelo de machine learning.

    Args:
        model (Any): Modelo de machine learning a utilizar.
        categorical_features (List[str]): Lista de nombres de columnas categóricas.
        numerical_features (List[str]): Lista de nombres de columnas numéricas.

    Returns:
        Pipeline: Pipeline que incluye el preprocesamiento y el modelo de machine learning.
    """
    preprocessor = prepare_data(categorical_features, numerical_features)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler(with_mean=False)),
        ('model', model)
    ])
    return pipeline

# 3. Función para ejecutar GridSearchCV
def run_grid_search(pipeline: Pipeline, param_grid: Dict[str, List[Any]], X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Any, GridSearchCV]:
    """
    Ejecuta un GridSearchCV para encontrar los mejores hiperparámetros para un modelo de machine learning.

    Args:
        pipeline (Pipeline): Pipeline que incluye el modelo y la preparación de datos.
        param_grid (Dict[str, List[Any]]): Diccionario de hiperparámetros a probar en el GridSearch.
        X_train (pd.DataFrame): Conjunto de datos de entrenamiento (características).
        y_train (pd.Series): Conjunto de datos de entrenamiento (target).

    Returns:
        Tuple: (Mejor estimador del grid search, objeto GridSearchCV con resultados completos)
    """
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search

# 4. Función para calcular las métricas
def calculate_metrics(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Calcula las métricas de rendimiento para un modelo dado en un conjunto de prueba.

    Args:
        model (Any): Modelo entrenado.
        X_test (pd.DataFrame): Conjunto de datos de prueba (características).
        y_test (pd.Series): Conjunto de datos de prueba (target).

    Returns:
        Dict[str, float]: Diccionario con las métricas MSE, RMSE, MAE y R2.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

# 5. Función para compilar los resultados
def compile_results(model_name: str, metrics_dict: Dict[str, float], params: Dict[str, Any]) -> pd.DataFrame:
    """
    Compila las métricas de rendimiento y los parámetros del modelo en un DataFrame.

    Args:
        model_name (str): Nombre del modelo.
        metrics_dict (Dict[str, float]): Diccionario con las métricas de rendimiento.
        params (Dict[str, Any]): Diccionario con los mejores parámetros encontrados por GridSearch.

    Returns:
        pd.DataFrame: DataFrame con las métricas y los parámetros del modelo.
    """
    result_row = {**metrics_dict, "Model": model_name, "Best_Params": params}
    return pd.DataFrame([result_row])

# 6. Función para calcular y compilar métricas
def calculate_and_compile_metrics(model_name: str, best_model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, best_params: Dict[str, Any]) -> pd.DataFrame:
    """
    Calcula y compila las métricas de entrenamiento y prueba en un DataFrame.

    Args:
        model_name (str): Nombre del modelo.
        best_model (Any): Modelo entrenado.
        X_train (pd.DataFrame): Conjunto de datos de entrenamiento (características).
        X_test (pd.DataFrame): Conjunto de datos de prueba (características).
        y_train (pd.Series): Conjunto de datos de entrenamiento (target).
        y_test (pd.Series): Conjunto de datos de prueba (target).
        best_params (Dict[str, Any]): Diccionario con los mejores parámetros encontrados por GridSearch.

    Returns:
        pd.DataFrame: DataFrame con las métricas de entrenamiento, prueba y los parámetros del modelo.
    """
    train_metrics = calculate_metrics(best_model, X_train, y_train)
    train_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
    test_metrics = calculate_metrics(best_model, X_test, y_test)
    test_metrics = {f"test_{k}": v for k, v in test_metrics.items()}
    combined_metrics = {**train_metrics, **test_metrics, "Model": model_name, "Best_Params": best_params}
    return pd.DataFrame([combined_metrics])

# 7. Función para la optimización bayesiana
def bayesian_optimization(model_class: Any, param_bounds: Dict[str, Tuple[int, int]], X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, categorical_features: List[str], numerical_features: List[str], init_points: int = 5, n_iter: int = 25) -> Tuple[Dict[str, Any], List[float], List[float]]:
    """
    Realiza la optimización bayesiana para ajustar hiperparámetros de un modelo.

    Args:
        model_class (Any): Clase del modelo de sklearn.
        param_bounds (Dict[str, Tuple[int, int]]): Diccionario con los límites de los hiperparámetros a optimizar.
        X_train (pd.DataFrame): Conjunto de entrenamiento de características.
        y_train (pd.Series): Conjunto de entrenamiento de la variable objetivo.
        X_test (pd.DataFrame): Conjunto de prueba de características.
        y_test (pd.Series): Conjunto de prueba de la variable objetivo.
        categorical_features (List[str]): Lista de características categóricas.
        numerical_features (List[str]): Lista de características numéricas.
        init_points (int): Número de puntos iniciales para la optimización bayesiana.
        n_iter (int): Número de iteraciones para la optimización bayesiana.

    Returns:
        Tuple: El mejor conjunto de parámetros, los scores de entrenamiento y prueba.
    """
    training_scores = []
    testing_scores = []

    def model_evaluation(**params):
        for param in params:
            if isinstance(params[param], float) and param.endswith('_estimators'):
                params[param] = int(params[param])
            elif isinstance(params[param], float) and param.endswith('depth'):
                params[param] = int(params[param])
            elif isinstance(params[param], float) and param.endswith('samples_split'):
                params[param] = int(params[param])
            elif isinstance(params[param], float) and param.endswith('child_weight'):
                params[param] = int(params[param])
            elif isinstance(params[param], float) and param.endswith('leaves'):
                params[param] = int(params[param])

        model = model_class(**params)
        pipeline = build_pipeline(model, categorical_features, numerical_features)
        pipeline.fit(X_train, y_train)

        train_score = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error').mean()
        training_scores.append(train_score)

        test_score = -mean_squared_error(y_test, pipeline.predict(X_test))
        testing_scores.append(test_score)

        return train_score

    optimizer = BayesianOptimization(f=model_evaluation, pbounds=param_bounds, random_state=42)
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    
    best_params = optimizer.max['params']
    
    print("Training Scores: ", training_scores)
    print("Testing Scores: ", testing_scores)
    
    return best_params, training_scores, testing_scores

def show_training(train_scores, test_scores):
    return pd.DataFrame([list(t) for t in zip(train_scores, test_scores)], columns = ["train", "test"])
