from typing import Union
import numpy as np
from numpy import ndarray
from pandas import Series, DataFrame
from preProcessing.PreProcessing import cross_val_score

def featureSelectionWrapper(X: DataFrame, y: DataFrame, estimator: str):

    n_iterations: int = 5
    current_mask: ndarray = np.zeros(shape=len(X.columns), dtype=bool)
    cross_val_percentage: float = 0.5

    for _ in range(n_iterations):
        new_feature_idx = _get_best_new_feature(estimator, X[: int(cross_val_percentage * X.shape[0])],
                                                y[: int(cross_val_percentage * X.shape[0])], current_mask)
        current_mask[new_feature_idx] = True

    # filter like [True, False, False]
    print('\t' + str(X.loc[:, current_mask].columns.values))

    return X.loc[:,current_mask.tolist()]

def featureSelectionCorr(X: DataFrame, y: DataFrame):

    corr = {}
    for column in X:
        corr[column] = np.abs(X[column].corr(y, method='spearman'))
    sortedCorr = sorted(corr.items(), key=lambda x: x[1], reverse=True)
    filter = list(dict(sortedCorr).keys())[:5]
    print('\t' + str(filter))
    return X[filter]

def _get_best_new_feature(estimator, X, y, current_mask):
    # Return the best new feature to add to the current_mask, i.e. return
    # the best new feature to add (resp. remove) when doing forward
    # selection (resp. backward selection)
    candidate_feature_indices = np.flatnonzero(~current_mask)
    scores = {}
    for feature_idx in candidate_feature_indices:
        candidate_mask = current_mask.copy()
        candidate_mask[feature_idx] = True
        X_new = X.iloc[:, candidate_mask]
        scores[feature_idx] = np.mean(cross_val_score(
            estimator, X_new, y))
    return min(scores, key=lambda feature_idx: scores[feature_idx])