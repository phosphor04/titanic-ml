
import pandas as pd
from sklearn.inspection import permutation_importance


def get_builtin_importance(model, feature_names):
    """Возвращает встроенную важность, если модель её поддерживает."""

    estimator = model.model

    if hasattr(estimator, 'feature_importances_'):
        values = estimator.feature_importances_
        method = 'builtin'

    elif hasattr(estimator, 'coef_'):
        # Для бинарной LogisticRegression
        values = estimator.coef_[0]
        method = 'coefficient'

    else:
        return None

    result = pd.DataFrame({
        'feature': feature_names,
        'importance': values,
    })

    if method == 'coefficient':
        result['abs_importance'] = result['importance'].abs()
        result = result.sort_values(
            'abs_importance', ascending=False
        ).drop(columns='abs_importance')
    else:
        result = result.sort_values(
            'importance', ascending=False
        )

    return result.reset_index(drop=True)


def get_permutation_importance(
    model,
    X_val,
    y_val,
    scoring='accuracy',
    n_repeats=10,
    seed=42,
):
    result = permutation_importance(
        model,
        X_val,
        y_val,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=seed,
    )

    return pd.DataFrame({
        'feature': X_val.columns,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std,
    }).sort_values(
        'importance_mean', ascending=False
    ).reset_index(drop=True)
