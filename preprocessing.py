import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
import re
from config import config
from feature_engineering import create_features 

### Filling NaNs in Embarked and Age ###
def handle_missing_values(X_train, X_val):
    X_train_preprocessed, X_val_preprocessed = X_train.copy(), X_val.copy()

    # Embarked
    mode_embarked = X_train_preprocessed['Embarked'].mode()[0]
    X_train_preprocessed['Embarked'] = X_train_preprocessed['Embarked'].fillna(mode_embarked)
    X_val_preprocessed['Embarked'] = X_val_preprocessed['Embarked'].fillna(mode_embarked)

    # if feature_engineering True, then calculate median in title, otherwise median overall for Nans in Age
    if  config.preprocessing.feature_engineering:
        age_medians = X_train_preprocessed.groupby('Title')['Age'].median()
        X_train_preprocessed['Age'] = X_train_preprocessed['Age'].fillna(
            X_train_preprocessed['Title'].map(age_medians)
        )
        X_val_preprocessed['Age'] = X_val_preprocessed['Age'].fillna(
            X_val_preprocessed['Title'].map(age_medians)
        )
    else:
        age_median = X_train_preprocessed['Age'].median()
        X_train_preprocessed['Age'] = (
            X_train_preprocessed['Age'].fillna(age_median)
        )

        X_val_preprocessed['Age'] = (
            X_val_preprocessed['Age'].fillna(age_median)
        )

    return X_train_preprocessed, X_val_preprocessed

### Dealing with categorical features Sex, Embarked and Title ###
def cat_featuring(X_train, X_val):
    X_train_preprocessed, X_val_preprocessed = X_train.copy(), X_val.copy()

    X_train_preprocessed['Sex'] = X_train_preprocessed['Sex'].map({'male': 0, 'female': 1})
    X_val_preprocessed['Sex'] = X_val_preprocessed['Sex'].map({'male': 0, 'female': 1})
    cat_features = ['Embarked', 'Title', 'AgeGroup'] if config.preprocessing.age_binning else ['Embarked', 'Title']

    # Training encoder
    encoder = OneHotEncoder(handle_unknown='ignore')
    X_train_encoded = encoder.fit_transform(X_train_preprocessed[cat_features])
    X_val_encoded = encoder.transform(X_val_preprocessed[cat_features])
    feature_names = encoder.get_feature_names_out(cat_features)

    # Encoder result into DataFrame
    X_train_encoded_df = pd.DataFrame(
        X_train_encoded.toarray(),
        columns=feature_names,
        index=X_train_preprocessed.index
    )
    # Deleting previous cat_features
    X_train_preprocessed.drop(columns=cat_features, inplace=True)
    # Adding new OHE features
    X_train_preprocessed = pd.concat([X_train_preprocessed, X_train_encoded_df], axis=1)

    # Same for X_val
    X_val_encoded_df = pd.DataFrame(
        X_val_encoded.toarray(),
        columns=feature_names,
        index=X_val_preprocessed.index
    )
    X_val_preprocessed.drop(columns=cat_features, inplace=True)
    X_val_preprocessed = pd.concat([X_val_preprocessed, X_val_encoded_df], axis=1)

    return X_train_preprocessed, X_val_preprocessed

# Bining Age into AgeGroup
def create_AgeGroup(df):
    df = df.copy()
    bins = [0, 18, 36, 60, float('inf')]
    labels = ['Child', 'YoungAdult', 'Adult', 'Senior']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    return df

### MAIN PREPROP
def preprocess_data(X_train, X_val):

    X_train_preprocessed, X_val_preprocessed = X_train.copy(), X_val.copy()

    # Feature_engineering if needed
    if config.preprocessing.feature_engineering:
        X_train_preprocessed, X_val_preprocessed = create_features(X_train_preprocessed, X_val_preprocessed)

    # Filling NaNs if needed
    if config.preprocessing.data_gaps:
        X_train_preprocessed, X_val_preprocessed = handle_missing_values(X_train_preprocessed, X_val_preprocessed)

    # Age binning if needed
    if config.preprocessing.age_binning:
        X_train_preprocessed = create_AgeGroup(X_train_preprocessed)
        X_val_preprocessed = create_AgeGroup(X_val_preprocessed)
        X_train_preprocessed.drop(columns=['Age'], inplace=True)
        X_val_preprocessed.drop(columns=['Age'], inplace=True)


    # Deleting unneeded features
    X_train_preprocessed.drop(columns=['PassengerId','Name', 'Cabin', 'Ticket', 'SibSp', 'Parch'], inplace=True)
    X_val_preprocessed.drop(columns=['PassengerId','Name', 'Cabin', 'Ticket', 'SibSp', 'Parch'], inplace=True)

    # One Hot Encoding for cat_features
    if config.preprocessing.cat_features_processing:
        X_train_preprocessed, X_val_preprocessed = cat_featuring(X_train_preprocessed, X_val_preprocessed)

    # Scaling numeric features if needed
    if config.preprocessing.scaler == 'StandardScaler':
            numeric_features_to_scale = ['Fare'] if config.preprocessing.age_binning else ['Fare', 'Age']
            scaler = StandardScaler()
            X_train_preprocessed[numeric_features_to_scale] = scaler.fit_transform(X_train_preprocessed[numeric_features_to_scale])
            X_val_preprocessed[numeric_features_to_scale] = scaler.transform(X_val_preprocessed[numeric_features_to_scale])

    return X_train_preprocessed, X_val_preprocessed
    





















































    # if config.preprocessing.data_gaps == True and config.model.name != 'CatBoostClassifier':
        
    #     #################### Обработка NaN в Age #################### 
        
        
    #     X_train_preprocessed['Age_missing'] = X_train_preprocessed['Age'].isna()
    #     X_val_preprocessed['Age_missing'] = X_val_preprocessed['Age'].isna()

    #     def get_title(name):
    #         main_titles = ['Mr', 'Miss', 'Mrs', 'Master']
    #         match = re.search(r',\s*([^\.]+)\.', name)

    #         return match.group(1) if match and match.group(1) in main_titles else 'Other'
        
    #     X_train_preprocessed['Title'] = X_train_preprocessed['Name'].apply(get_title)
    #     X_val_preprocessed['Title'] = X_val_preprocessed['Name'].apply(get_title)

    #     # Вычисляем медиану возраста для каждой группы Title в X_train
    #     age_medians = X_train_preprocessed.groupby('Title')['Age'].median()

    #     X_train_preprocessed['Age'] = X_train_preprocessed['Age'].fillna(
    #         X_train_preprocessed['Title'].map(age_medians)
    #     )
    #     X_val_preprocessed['Age'] = X_val_preprocessed['Age'].fillna(
    #         X_val_preprocessed['Title'].map(age_medians)
    #     )

    #     X_train_preprocessed.drop(columns='Name', inplace=True)
    #     X_val_preprocessed.drop(columns='Name', inplace=True)


    #     #################### Обработка NaN в Cabin #################### 
        
        
    #     sectors = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']
    #     def get_sector(cabin):
    #         if pd.isna(cabin):
    #             return 'Unknown'
    #         match = re.search(r'[A-Z]', cabin)
    #         return match.group(0) if match and match.group(0) in sectors else 'Unknown'
    #     X_train_preprocessed['Deck'] = X_train_preprocessed['Cabin'].apply(get_sector)
    #     X_val_preprocessed['Deck'] = X_val_preprocessed['Cabin'].apply(get_sector)

    #     X_train_preprocessed.drop(columns='Cabin', inplace=True)
    #     X_val_preprocessed.drop(columns='Cabin', inplace=True)


    #     #################### Обработка NaN в Embarked #################### 
    #     mode_embarked = X_train_preprocessed['Embarked'].mode()[0]
    #     X_train_preprocessed.loc[X_train_preprocessed['Embarked'].isna() == True, 'Embarked'] = mode_embarked
    #     X_val_preprocessed.loc[X_val_preprocessed['Embarked'].isna() == True, 'Embarked'] = mode_embarked













    # if config.model.name == 'CatBoostClassifier':
    #     #################### Удаление ненужных столбцов #################### 
    #     X_train_preprocessed.drop(columns=['PassengerId', 'Ticket'], inplace=True)
    #     X_val_preprocessed.drop(columns=['PassengerId', 'Ticket'], inplace=True)

    #     #################### Обработка NaN в Cabin #################### 
        
        
    #     sectors = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']
    #     def get_sector(cabin):
    #         if pd.isna(cabin):
    #             return 'Unknown'
    #         match = re.search(r'[A-Z]', cabin)
    #         return match.group(0) if match and match.group(0) in sectors else 'Unknown'
    #     X_train_preprocessed['Deck'] = X_train_preprocessed['Cabin'].apply(get_sector)
    #     X_val_preprocessed['Deck'] = X_val_preprocessed['Cabin'].apply(get_sector)

    #     X_train_preprocessed.drop(columns='Cabin', inplace=True)
    #     X_val_preprocessed.drop(columns='Cabin', inplace=True)


    #     #################### Обработка NaN в Embarked #################### 

    #     #X_train_preprocessed['Age_missing'] = X_train_preprocessed['Age'].isna()
    #     #X_val_preprocessed['Age_missing'] = X_val_preprocessed['Age'].isna()

    #     def get_title(name):
    #         main_titles = ['Mr', 'Miss', 'Mrs', 'Master']
    #         match = re.search(r',\s*([^\.]+)\.', name)

    #         return match.group(1) if match and match.group(1) in main_titles else 'Other'
        
    #     X_train_preprocessed['Title'] = X_train_preprocessed['Name'].apply(get_title)
    #     X_val_preprocessed['Title'] = X_val_preprocessed['Name'].apply(get_title)

    #     X_train_preprocessed.drop(columns='Name', inplace=True)
    #     X_val_preprocessed.drop(columns='Name', inplace=True)
        
    #     #################### Обработка NaN в Embarked #################### 
    #     mode_embarked = X_train_preprocessed['Embarked'].mode()[0]
    #     X_train_preprocessed.loc[X_train_preprocessed['Embarked'].isna() == True, 'Embarked'] = mode_embarked
    #     X_val_preprocessed.loc[X_val_preprocessed['Embarked'].isna() == True, 'Embarked'] = mode_embarked   



    # if config.preprocessing.cat_features_processing == True:
    #     #################### Удаление ненужных столбцов #################### 
    #     X_train_preprocessed.drop(columns=['PassengerId', 'Ticket'], inplace=True)
    #     X_val_preprocessed.drop(columns=['PassengerId', 'Ticket'], inplace=True)

    #     #################### Преобразование категориальных переменных Sex, Age_missing и Embarked#################### 
    #     X_train_preprocessed['Sex'] = X_train_preprocessed['Sex'].map({'male': 0, 'female': 1})
    #     X_train_preprocessed['Age_missing'] = X_train_preprocessed['Age_missing'].map({False: 0, True: 1})
    #     X_val_preprocessed['Sex'] = X_val_preprocessed['Sex'].map({'male': 0, 'female': 1})
    #     X_val_preprocessed['Age_missing'] = X_val_preprocessed['Age_missing'].map({False: 0, True: 1})

    #     cat_features = ['Embarked', 'Title', 'Deck']        
    #     # Создаём и обучаем encoder
    #     encoder = OneHotEncoder(handle_unknown='ignore')
    #     X_train_encoded = encoder.fit_transform(X_train_preprocessed[cat_features])
    #     X_val_encoded = encoder.transform(X_val_preprocessed[cat_features])

    #     # Получаем названия новых признаков
    #     feature_names = encoder.get_feature_names_out(cat_features)

    #     # Превращаем результат encoder в DataFrame
    #     X_train_encoded_df = pd.DataFrame(
    #         X_train_encoded.toarray(),
    #         columns=feature_names,
    #         index=X_train_preprocessed.index
    #     )
    #     # Удаляем исходные категориальные признаки
    #     X_train_preprocessed.drop(columns=cat_features, inplace=True)
    #     # Добавляем One-Hot признаки
    #     X_train_preprocessed = pd.concat([X_train_preprocessed, X_train_encoded_df], axis=1)

    #     # Повторяем для валидационного набора
    #     X_val_encoded_df = pd.DataFrame(
    #         X_val_encoded.toarray(),
    #         columns=feature_names,
    #         index=X_val_preprocessed.index
    #     )
    #     X_val_preprocessed.drop(columns=cat_features, inplace=True)
    #     X_val_preprocessed = pd.concat([X_val_preprocessed, X_val_encoded_df], axis=1)