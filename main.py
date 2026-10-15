import numpy as np
import pandas as pd

from data import get_train_val_split
from preprocessing import preprocess_data
from model import My_model
from feature_analysis import get_permutation_importance

from config import config
from omegaconf import OmegaConf

def main():
    scores= []
    importance_results = []

    for X_train, X_val, y_train, y_val in get_train_val_split():
        # Preprocess the data
        X_train_processed, X_val_processed= preprocess_data(X_train, X_val)

        # Training
        model = My_model()
        model.fit(X_train_processed, y_train)

        # Permutation importance on validation fold
        importance = get_permutation_importance(
            model=model.model,
            X_val=X_val_processed,
            y_val=y_val,
            scoring='accuracy',
            n_repeats=10,
            seed=config.general.seed,
        )

        importance_results.append(importance)

        # Accuracy
        scores.append(model.get_score(X_val_processed, y_val))

    
    with open('results.txt', 'a', encoding='utf-8') as file:
        file.write('=' *  150 + '\n')
        model_params = OmegaConf.to_container(
            config.model[config.model.name],
            resolve=True,
        )

        file.write(f'Model: {config.model.name}\n')
        #file.write(f'Preprocessing parameters:\n\t{OmegaConf.to_container(config.preprocessing, resolve=True)}\n')
        file.write(f'Parameters:\n\t{OmegaConf.to_container(config.model[config.model.name],resolve=True,)}\n')
        #file.write(f'CV scores: {scores}\n')
        file.write(f'Mean accuracy: {np.mean(scores):.4f}\n')
        file.write(f'Std accuracy: {np.std(scores):.4f}\n\n')

        # Aggregate importance across folds
        all_importance = pd.concat(
            importance_results,
            ignore_index=True,
        )

        # mean_importance = (
        #     all_importance
        #     .groupby('feature')
        #     .agg(
        #         importance_mean=('importance_mean', 'mean'),
        #         importance_std=('importance_mean', 'std'),
        #     )
        #     .sort_values('importance_mean', ascending=False)
        #     .reset_index()
        # )

        # file.write('\nPermutation importance:')
        # file.write(
        #     mean_importance.to_string(
        #         index=False,
        #         formatters={
        #             'importance_mean': '{:.4f}'.format,
        #             'importance_std': '{:.4f}'.format,
        #         },
        #     )
        # )
        # file.write('\n\n')

if __name__ == "__main__":
    main()