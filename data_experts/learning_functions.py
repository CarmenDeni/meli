import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from bayes_opt import BayesianOptimization
import numpy as np


def prepare_data(df: pd.DataFrame, categorical_features, numerical_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_features)
        ])
    
    return preprocessor


def build_pipeline(model, df, categorical_features, numerical_features):
    preprocessor = prepare_data(df, categorical_features, numerical_features)
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler(with_mean=False)),  
        ('model', model)
    ])
    
    return pipeline


def run_grid_search(pipeline, param_grid, X_train, y_train):
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    return grid_search


def bayesian_optimization(model_class, param_bounds, X_train, y_train, categorical_features, numerical_features, init_points=5, n_iter=25):
    """
    Realiza la optimización bayesiana para ajustar hiperparámetros de un modelo.

    Args:
        model_class: Clase del modelo de sklearn.
        param_bounds: Diccionario con los límites de los hiperparámetros a optimizar.
        X_train: Conjunto de entrenamiento de características.
        y_train: Conjunto de entrenamiento de la variable objetivo.
        categorical_features: Lista de características categóricas.
        numerical_features: Lista de características numéricas.
        init_points: Número de puntos iniciales para la optimización bayesiana.
        n_iter: Número de iteraciones para la optimización bayesiana.

    Returns:
        Optimizer con los resultados de la optimización bayesiana.
    """

    def model_evaluation(**params):
        # Convertir los parámetros que deben ser enteros
        params['n_estimators'] = int(params['n_estimators'])
        params['max_depth'] = int(params['max_depth'])
        params['min_samples_split'] = int(params['min_samples_split'])
        
        model = model_class(**params)
        pipeline = build_pipeline(model, df, categorical_features, numerical_features)
        score = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error').mean()
        return score

    optimizer = BayesianOptimization(f=model_evaluation, pbounds=param_bounds, random_state=42)
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    return optimizer


def grid_search_results_to_dataframe(grid_search_result):
    results = pd.DataFrame(grid_search_result.cv_results_)
    results = results[['mean_test_score', 'std_test_score', 'params']]
    results['mean_test_score'] = -results['mean_test_score']  
    results.sort_values(by='mean_test_score', ascending=True, inplace=True)
    return results
