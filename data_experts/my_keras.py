import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.metrics import  r2_score
import numpy as np

class KerasRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, build_fn, epochs=100, batch_size=32, verbose=1):
        self.build_fn = build_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None
    
    def fit(self, X, y):
        self.model = self.build_fn()
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def score(self, X, y):
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
    
class MyKerasRegressor(KerasRegressor):
    def __init__(self, input_dim, layers=2, units=64, activation='relu', optimizer='adam', epochs=100, batch_size=32, **kwargs):
        self.input_dim = input_dim
        self.layers = layers
        self.units = units
        self.activation = activation
        self.optimizer = optimizer
        super().__init__(build_fn=self.build_model, epochs=epochs, batch_size=batch_size, **kwargs)
    
    def build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.input_dim,)))
        for _ in range(self.layers):
            model.add(Dense(units=self.units, activation=self.activation))
        model.add(Dense(1))  # Para regresión, la capa de salida tiene una neurona
        model.compile(optimizer=self.optimizer, loss='mean_squared_error', metrics=['r2_score'])
        return model

class CustomSearchCV:
    def __init__(self, estimator, param_grid, search_type='grid', cv=3, scoring='r2', n_iter=10, factor=3, verbose=1, random_state=42, stop_threshold=0.98):
        self.estimator = estimator
        self.param_grid = param_grid
        self.search_type = search_type
        self.cv = cv
        self.scoring = scoring
        self.n_iter = n_iter
        self.factor = factor
        self.verbose = verbose
        self.random_state = random_state
        self.stop_threshold = stop_threshold
        
        if self.search_type == 'grid':
            self.searcher = GridSearchCV(estimator, param_grid, cv=cv, scoring=scoring, verbose=verbose)
        elif self.search_type == 'random':
            self.searcher = RandomizedSearchCV(estimator, param_grid, cv=cv, scoring=scoring, n_iter=n_iter, random_state=random_state, verbose=verbose)
        elif self.search_type == 'halving':
            self.searcher = HalvingGridSearchCV(estimator, param_grid, cv=cv, scoring=scoring, factor=factor, random_state=random_state, verbose=verbose)
        else:
            raise ValueError("search_type must be 'grid', 'random', or 'halving'")

        self.train_scores_ = []
        self.test_scores_ = []
        self.params_ = []

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        self.searcher.fit(X_train, y_train)
        
        for params in self.searcher.cv_results_['params']:
            self.estimator.set_params(**params)
            self.estimator.fit(X_train, y_train)

            train_score = r2_score(y_train, self.estimator.predict(X_train))
            if X_test is not None and y_test is not None:
                test_score = r2_score(y_test, self.estimator.predict(X_test))
            else:
                test_score = None

            self.train_scores_.append(train_score)
            self.test_scores_.append(test_score)
            self.params_.append(params)

            print(f"Train R2 Score: {train_score:.4f} | Test R2 Score: {test_score:.4f} | Params: {params}")

            if train_score >= self.stop_threshold:
                print(f"Stopping early as train R2 score reached {train_score:.4f} (threshold: {self.stop_threshold})")
                break

    def best_params_(self):
        return self.searcher.best_params_

    def best_score_(self):
        return self.searcher.best_score_

    def results_(self):
        return {
            "train_scores": self.train_scores_,
            "test_scores": self.test_scores_,
            "params": self.params_
        }

