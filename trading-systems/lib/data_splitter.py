import pandas as pd
from typing import Dict, Any


def split_features_and_target_into_train_and_test_set(
    features: pd.DataFrame, targets: Dict[Any, pd.Series], percentage: int
):
    """
    Splitting the data into a test set and a training set.
    Set percentage to 0 for putting all data into the training set.
    """
    test_set_size = int(len(features) * (percentage / 100))
    training_set_size = int(len(features) - test_set_size)

    training_set_features = features.iloc[:training_set_size]
    training_set_targets = {}
    for key, series in targets.items():
        training_set_targets[key] = series[:training_set_size]

    test_set_features = features.iloc[training_set_size: len(features)]
    test_set_targets = {}
    for key, series in targets.items():
        test_set_targets[key] = series[training_set_size: len(features)]

    return training_set_features, training_set_targets, test_set_features, test_set_targets
