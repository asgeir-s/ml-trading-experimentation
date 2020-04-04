import pandas as pd
from typing import List


def generteTrainAndTestSet(
    features: pd.DataFrame, target: pd.Series, percentage: int, featuresColumnsToDrop: List[str]
):
    """
    Splitting the data into a test set and a training set.
    Set percentage to 0 for putting all data into the training set.
    """
    testSetSize = int(len(features) * (percentage / 100))
    trainingSetSize = int(len(features) - testSetSize)

    features = features.drop(columns=featuresColumnsToDrop)

    trainingSetFeatures = features.iloc[:trainingSetSize]
    trainingSetTarget = target[:trainingSetSize]

    testSetFeatures = features.iloc[trainingSetSize : len(features)]
    testSetTarget = target[trainingSetSize : len(features)]

    return trainingSetFeatures, trainingSetTarget, testSetFeatures, testSetTarget
