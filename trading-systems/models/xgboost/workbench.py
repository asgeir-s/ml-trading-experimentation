from model import XgboostBaseModel
import pandas as pd
from lib.data_splitter import split_features_and_target_into_train_and_test_set
from lib.data_loader import load_candlesticks


def main():
    candlesticks = load_candlesticks("1h")

    features = XgboostBaseModel.generate_features(candlesticks)
    target = XgboostBaseModel.generate_target(features)
    (
        training_set_features,
        training_set_target,
        test_set_features,
        test_set_target,
    ) = split_features_and_target_into_train_and_test_set(features, target, 20)

    xg_boost = XgboostBaseModel()
    xg_boost.train(training_set_features, training_set_target)
    xg_boost.evaluate(test_set_features, test_set_target)
    xg_boost.print_info()


if __name__ == "__main__":
    main()
