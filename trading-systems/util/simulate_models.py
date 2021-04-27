from typing import Any
from models.lightgbm.regression_baby.max_model import RegressionBabyMaxModel
from models.lightgbm.regression_baby.min_model import RegressionBabyMinModel
from pandas.core.frame import DataFrame, Series
from models.tensorflow.price_prediction_lstm.model import PricePreditionLSTMModel
from models.xgboost import ClassifierSklienSimpleModel, ClassifierUpDownModel
from lib.model import Model


def simulate_running(candlesticks: DataFrame, start_running_index = 10000) -> DataFrame:
    models = (
        PricePreditionLSTMModel(
            target_name="close", forward_look_for_target=1, window_size=25,
        ),
        PricePreditionLSTMModel(
            target_name="close", forward_look_for_target=3, window_size=25,
        ),
        PricePreditionLSTMModel(
            target_name="ema", forward_look_for_target=3, window_size=25,
        ),
        PricePreditionLSTMModel(
            target_name="low", forward_look_for_target=6, window_size=25,
        ),
        PricePreditionLSTMModel(
            target_name="high", forward_look_for_target=6, window_size=25,
        ),
        RegressionBabyMinModel(),
        RegressionBabyMaxModel(),
        ClassifierSklienSimpleModel(),
        ClassifierUpDownModel(),
    )

    predictions: Any = None
    for m in models:
        print(f"Running model: {m}")
        features = DataFrame(index=candlesticks.index)
        features = m.generate_features(candlesticks, features)
        target = m.generate_target(candlesticks, features)

        pred = simulate_periodically_retrain(
            m,
            features,
            target,
            start_running_index=start_running_index,
            training_interval=720,
            window_range=m.window_size,
        )
        if predictions is None:
            predictions = DataFrame(index=pred.index)
        predictions[m] = pred

    return predictions


def simulate_periodically_retrain(
    model: Model,
    features: DataFrame,
    target: Series,
    start_running_index=10000,
    training_interval=720,
    window_range=1,
):
    last_index = start_running_index
    predictions = None

    # %% initial training
    model.train(features[0:start_running_index], target[0:start_running_index])

    # %% simulate live running with periodically retraining
    while last_index < len(features):
        start_index = last_index-window_range+1
        stop_index = (
            (last_index + training_interval)
            if len(features) > (last_index + training_interval)
            else len(features)
        )
        print(f"{start_index}")
        print(f"{stop_index}")
        new_pred = model.predict_dataframe(features[start_index:stop_index])
        if predictions is None:
            predictions = new_pred
        else:
            predictions = predictions.append([new_pred])
        model.train(features[:stop_index], target[:stop_index])
        last_index = stop_index
        print(f"{last_index=}")
    
    return predictions