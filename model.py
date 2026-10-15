from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import pandas as pd
from sklearn.metrics import accuracy_score

from config import config

class My_model:
    def __init__(self):
        if config.model.name == 'LogisticRegression':
            params = config.model.LogisticRegression
            self.model = LogisticRegression(
                l1_ratio=params.l1_ratio,
                max_iter=params.max_iter,
                C=params.C,
                solver = params.solver,
                random_state=config.general.seed
            )

        elif config.model.name == 'KNN':
            params = config.model.KNN
            self.model = KNeighborsClassifier(
                n_neighbors=params.n_neighbors,
                weights=params.weights,
                metric=params.metric
            )
        elif config.model.name == 'DecisionTreeClassifier':
            params = config.model.DecisionTreeClassifier
            self.model = DecisionTreeClassifier(
                max_depth = params.max_depth,
                random_state = params.random_state
            )
        elif config.model.name == 'RandomForestClassifier':
                    params = config.model.RandomForestClassifier
                    self.model = RandomForestClassifier(
                        n_estimators = params.n_estimators,
                        max_depth = params.max_depth,
                        max_features = params.max_features,
                        bootstrap = params.bootstrap,
                        random_state = params.random_state
                    )
        elif config.model.name == 'CatBoostClassifier':
            params = config.model.CatBoostClassifier

            self.model = CatBoostClassifier(
                iterations=params.iterations,
                learning_rate=params.learning_rate,
                depth=params.depth,
                random_seed=params.random_seed,
                verbose=params.verbose,
                cat_features = (
                    ['Sex', 'Embarked', 'Title'] if config.preprocessing.feature_engineering == True 
                    else ['Sex', 'Embarked']
                )
            )
        elif config.model.name == 'LGBMClassifier':
            params = config.model.LGBMClassifier

            self.model = LGBMClassifier(
                n_estimators=params.n_estimators,
                learning_rate=params.learning_rate,
                max_depth=params.max_depth,
                num_leaves=params.num_leaves,
                min_child_samples = params.min_child_samples,
                reg_lambda = params.reg_lambda,
                reg_alpha = params.reg_alpha,
                colsample_bytree = params.colsample_bytree,
                subsample = params.subsample,
                subsample_freq = params.subsample_freq,
                random_state=params.random_state,
                verbosity=params.verbosity
            )
        elif config.model.name == 'XGBClassifier':
            params = config.model.XGBClassifier

            self.model = XGBClassifier(
                n_estimators=params.n_estimators,
                learning_rate=params.learning_rate,
                max_depth=params.max_depth,
                min_child_weight=params.min_child_weight,
                subsample=params.subsample,
                colsample_bytree=params.colsample_bytree,
                reg_alpha=params.reg_alpha,
                reg_lambda=params.reg_lambda,
                random_state=params.random_state,
                eval_metric='logloss',
                tree_method='hist',
            )
        else:
            raise ValueError(f'Unknown model: {config.model.name}')

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self
        
    def get_score(self, X_val, y_val):
        predictions = self.model.predict(X_val)
        return accuracy_score(y_val, predictions)