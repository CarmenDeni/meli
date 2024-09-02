import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))
from data_experts.learning_functions import *

import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


def normalized_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred)) / (y_true.max() - y_true.min())

def normalized_mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) / (y_true.max() - y_true.min())

def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)


def custom_scorer(estimator, X, y):
    # Predicciones
    y_pred_train = estimator.predict(X)
    
    # Calcula R2
    r2_train = r2_score(y, y_pred_train)
    
    # Devuelve el score negativo para que GridSearchCV pueda minimizarlo
    return r2_train

def custom_combined_scorer(estimator, X_train, y_train, X_test, y_test):
    y_pred_train = estimator.predict(X_train)
    y_pred_test = estimator.predict(X_test)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    combined_score = (r2_train + r2_test) / 2
    return combined_score

# 0. Para el uso de SHAP y las técnicas de feature importance
def add_categorical_random(X: pd.DataFrame, num_random_columns: int = 1) -> pd.DataFrame:
    for i in range(num_random_columns):
        X[f'cat_random_{i}'] = np.random.choice(['A', 'B', 'C'], size=len(X))
    return X

def add_numerical_random(X: pd.DataFrame, num_random_columns: int = 1) -> pd.DataFrame:
    for i in range(num_random_columns):
        X[f'num_random_{i}'] = np.random.randn(len(X))
    return X

def partition_numerical_features(df: pd.DataFrame, numerical_features: List[str]) -> Tuple[List[str], List[str]]:
    """
    Particiona las características numéricas en dos listas: una con características con distribución cuasi-normal
    y otra con características que no siguen una distribución normal.

    Args:
        df (pd.DataFrame): DataFrame con los datos.
        numerical_features (List[str]): Lista de nombres de columnas numéricas.

    Returns:
        Tuple[List[str], List[str]]: Dos listas de características numéricas (cuasi-normales y no normales).
    """
    features_with_quasinormal_distr = []
    features_with_no_normal_distr = []

    for feature in numerical_features:
        stat, p_value = shapiro(df[feature])
        # Si p-value > 0.05, no podemos rechazar la hipótesis nula de que la distribución es normal
        if p_value > 0.05:
            features_with_quasinormal_distr.append(feature)
        else:
            features_with_no_normal_distr.append(feature)
    
    return features_with_quasinormal_distr, features_with_no_normal_distr



def remove_outliers(df: pd.DataFrame, features: List[str], exclude: List[str] = [], z_thresh: float = 3.0) -> pd.DataFrame:
    """
    Elimina outliers de las características numéricas basándose en el z-score,
    con la opción de excluir ciertas columnas del proceso de eliminación.

    Args:
        df (pd.DataFrame): DataFrame con los datos.
        features (List[str]): Lista de nombres de columnas numéricas.
        exclude (List[str]): Lista de columnas a excluir del proceso de eliminación de outliers.
        z_thresh (float): Umbral del z-score para definir outliers.

    Returns:
        pd.DataFrame: DataFrame sin outliers en las columnas especificadas.
    """
    from scipy.stats import zscore

    # Filtrar las características para excluir las especificadas
    features_to_check = [feature for feature in features if feature not in exclude]

    # Eliminar outliers solo en las columnas que no están en 'exclude'
    df_clean = df[(np.abs(zscore(df[features_to_check])) < z_thresh).all(axis=1)]
    
    return df_clean


# 1. Función para preparar los datos
def prepare_data(categorical_features: List[str], quasinormal_features: List[str], non_normal_features: List[str], one_hot: bool = True) -> ColumnTransformer:
    """
    Prepara los datos para el entrenamiento del modelo aplicando transformaciones a las características categóricas
    y numéricas según su distribución. Incluye la opción de usar OneHotEncoder para las características categóricas.

    Args:
        categorical_features (List[str]): Lista de nombres de columnas categóricas.
        quasinormal_features (List[str]): Lista de nombres de columnas cuasi-normales.
        non_normal_features (List[str]): Lista de nombres de columnas no normales.
        one_hot (bool): Si es True, aplica OneHotEncoder a las características categóricas. De lo contrario, usa OrdinalEncoder.

    Returns:
        ColumnTransformer: Transformador de columnas preparado para el preprocesamiento de datos.
    """
    transformers = [
        ('num_quasinormal', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler(with_mean=False))
        ]), quasinormal_features),
        
        ('num_non_normal', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler())
        ]), non_normal_features)
    ]
    
    if one_hot:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_features))
    else:
        transformers.append(('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ordinal_encoder', OrdinalEncoder())
        ]), categorical_features))
    
    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor


# 2. Función para construir el pipeline
def build_pipeline(model: Any, categorical_features: List[str], quasinormal_features: List[str], non_normal_features: List[str], one_hot: bool = True) -> Pipeline:
    """
    Construye un pipeline que incluye la preparación de los datos y el modelo de machine learning.

    Args:
        model (Any): Modelo de machine learning a utilizar.
        categorical_features (List[str]): Lista de nombres de columnas categóricas.
        quasinormal_features (List[str]): Lista de nombres de columnas cuasi-normales.
        non_normal_features (List[str]): Lista de nombres de columnas no normales.
        one_hot (bool): Si es True, aplica OneHotEncoder a las características categóricas. De lo contrario, usa OrdinalEncoder.

    Returns:
        Pipeline: Pipeline que incluye el preprocesamiento y el modelo de machine learning.
    """
    preprocessor = prepare_data(categorical_features, quasinormal_features, non_normal_features, one_hot)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    return pipeline


# 3. Función para ejecutar GridSearchCV
def run_grid_search(pipeline: Pipeline, param_grid: Dict[str, List[Any]], X_train: pd.DataFrame, y_train: pd.Series, scoring ='neg_mean_squared_error') -> Tuple[Any, GridSearchCV]:
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
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring=scoring, n_jobs=-1, verbose=1)
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

# Función para construir el modelo con un input_dim dinámico usando Input layer
def build_model(input_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))  # Define explícitamente la forma de entrada
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # Para regresión, la capa de salida tiene una neurona
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['r2_score'])
    return model

# Wrapp Keras model into a scikit-learn compatible model
def create_keras_model(input_dim):
    return KerasRegressor(build_fn=lambda: build_model(input_dim), epochs=100, batch_size=32, verbose=1)


# 7. Función para la optimización bayesiana
def bayesian_optimization2(
    model_class: Any,
    param_bounds: Dict[str, Tuple[float, float]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    categorical_features: List[str],
    quasinormal_features: List[str],
    non_normal_features: List[str],
    init_points: int = 5,
    n_iter: int = 25
) -> Tuple[Dict[str, Any], List[float], List[float]]:
    """
    Realiza la optimización bayesiana para ajustar los hiperparámetros de un modelo dado,
    utilizando el R2 score como métrica de evaluación.

    Args:
        model_class (Any): Clase del modelo a entrenar (e.g., RandomForestRegressor).
        param_bounds (Dict[str, Tuple[float, float]]): Diccionario con los límites inferiores y superiores
            de los hiperparámetros a optimizar.
        X_train (pd.DataFrame): Conjunto de datos de entrenamiento.
        y_train (pd.Series): Valores objetivo del conjunto de entrenamiento.
        X_test (pd.DataFrame): Conjunto de datos de prueba.
        y_test (pd.Series): Valores objetivo del conjunto de prueba.
        categorical_features (List[str]): Lista de nombres de características categóricas.
        quasinormal_features (List[str]): Lista de nombres de características cuasi-normales.
        non_normal_features (List[str]): Lista de nombres de características no normales.
        init_points (int, optional): Número de puntos iniciales para la optimización bayesiana. Por defecto es 5.
        n_iter (int, optional): Número de iteraciones para la optimización bayesiana. Por defecto es 25.

    Returns:
        Tuple[Dict[str, Any], List[float], List[float]]:
            - best_params (Dict[str, Any]): Diccionario con los mejores hiperparámetros encontrados.
            - training_scores (List[float]): Lista de R2 scores en el conjunto de entrenamiento por iteración.
            - testing_scores (List[float]): Lista de R2 scores en el conjunto de prueba por iteración.
    """
    training_scores = []
    testing_scores = []
    
    def model_evaluation(**params):
        # Convertir parámetros que deben ser enteros
        for param in params:
            if isinstance(params[param], float) and param in ['n_estimators', 'max_depth', 
                                                              'min_samples_split', 'min_child_weight',
                                                              'num_leaves']:
                params[param] = int(round(params[param]))
        
        model = model_class(**params)
        pipeline = build_pipeline(model, categorical_features, quasinormal_features, non_normal_features)
        pipeline.fit(X_train, y_train)
        
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        training_scores.append(train_r2)
        testing_scores.append(test_r2)
        
        # Retornar el R2 de validación (en este caso usamos el de prueba) para maximizar
        return test_r2
    
    optimizer = BayesianOptimization(
        f=model_evaluation,
        pbounds=param_bounds,
        random_state=42,
        verbose=2
    )
    
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    
    best_params = optimizer.max['params']
    # Convertir posibles parámetros enteros
    for param in best_params:
        if param in ['n_estimators', 'max_depth', 'min_samples_split', 'min_child_weight', 'num_leaves']:
            best_params[param] = int(round(best_params[param]))
    
    print("\nMejores Hiperparámetros Encontrados:")
    for param, value in best_params.items():
        print(f" - {param}: {value}")
    
    print("\nR2 Scores por Iteración:")
    for i, (train_score, test_score) in enumerate(zip(training_scores, testing_scores), 1):
        print(f"Iteración {i}: Train R2 = {train_score:.4f}, Test R2 = {test_score:.4f}")
    
    return best_params, training_scores, testing_scores

def show_training(train_scores, test_scores):
    return pd.DataFrame([list(t) for t in zip(train_scores, test_scores)], columns = ["train", "test"])
pd.options.display.float_format = '{:,.2f}'.format


#feature importances
def plot_and_get_feature_importances(pipeline, X_train, N=10):
    """
    Visualiza las N características más importantes de un modelo RandomForestRegressor 
    dentro de un pipeline de scikit-learn, y devuelve un DataFrame con todas las importancias.

    Args:
    - pipeline: Pipeline entrenado que contiene un RandomForestRegressor.
    - X_train: DataFrame de entrenamiento con las características originales.
    - N: Número de características más importantes a mostrar (por defecto 10).

    Returns:
    - DataFrame con todas las características y sus importancias.
    """
    # Acceder al ColumnTransformer desde el pipeline
    preprocessor = pipeline.named_steps['preprocessor']
    
    # Obtener nombres de columnas de las características numéricas y categóricas
    num_quasinormal_features = preprocessor.transformers_[0][2]
    num_non_normal_features = preprocessor.transformers_[1][2]
    
    # Obtener los nombres de las columnas que salen del OneHotEncoder
    ohe = preprocessor.transformers_[2][1]
    cat_features = preprocessor.transformers_[2][2]
    ohe_features = ohe.get_feature_names_out(cat_features)
    
    # Concatenar todos los nombres de las características
    feature_names = np.concatenate([num_quasinormal_features, num_non_normal_features, ohe_features])
    
    # Obtener las importancias de las características
    importances = pipeline.named_steps['model'].feature_importances_
    
    # Crear un DataFrame con las características y sus importancias
    feature_importances_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Ordenar el DataFrame por importancia descendente
    feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    # Graficar las N características más importantes
    plt.figure(figsize=(10, 8))
    plt.title(f"Top {N} Feature Importances")
    plt.barh(feature_importances_df['Feature'][:N], feature_importances_df['Importance'][:N], align="center")
    plt.xlabel("Importance")
    plt.gca().invert_yaxis()  # Invertir para que la característica más importante esté en la parte superior
    plt.show()
    
    return feature_importances_df
