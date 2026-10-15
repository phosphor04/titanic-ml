import pandas as pd
from sklearn.model_selection import StratifiedKFold
from config import config

def get_train_val_split():
    train_path = config.paths.path_to_train_data
    test_path = config.paths.path_to_test_data

    train_raw = pd.read_csv(train_path)
    train = train_raw.copy()
    test = pd.read_csv(test_path)

    X = train.drop(columns = 'Survived')
    y = train['Survived']

    skf = StratifiedKFold(
        n_splits=config.split.n_splits,
        shuffle=config.split.shuffle,
        random_state=config.general.seed
    )

    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X.loc[train_index], X.loc[val_index]
        y_train, y_val = y.loc[train_index], y.loc[val_index]
        yield X_train, X_val, y_train, y_val