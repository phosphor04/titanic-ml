import re
import pandas as pd
import numpy as np

def get_title(name):
    main_titles = ['Mr', 'Miss', 'Mrs', 'Master']
    match = re.search(r',\s*([^\.]+)\.', name)

    return match.group(1) if match and match.group(1) in main_titles else 'Other'


def create_features(X_train, X_val):
    X_train = X_train.copy()
    X_val = X_val.copy()

    # Title
    X_train['Title'] = X_train['Name'].apply(get_title)
    X_val['Title'] = X_val['Name'].apply(get_title)

    # FamilySize
    X_train['FamilySize'] = X_train['SibSp'] + X_train['Parch'] + 1
    X_val['FamilySize'] = X_val['SibSp'] + X_val['Parch'] + 1

    return X_train, X_val