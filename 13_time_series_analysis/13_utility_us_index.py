import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as sk_model_selection
import sklearn.metrics as sk_metrics
import statsmodels.tsa.seasonal as statsmodels_seasonal
import statsmodels.tsa.statespace.sarimax as statsmodels_sarimax


def make_cv_splits(data_df, n_splits):
    time_series_cv_splits = sk_model_selection.TimeSeriesSplit(n_splits=n_splits)
    utility_index_cs_splits_indices = time_series_cv_splits.split(data_df)

    utility_index_cv_splits = []
    for train_indices, test_indices in utility_index_cs_splits_indices:
        train, test = data_df.iloc[train_indices], data_df.iloc[test_indices]
        utility_index_cv_splits.append((train, test))

        # plt.figure()
        # plt.plot(train.index, train["value"], color="g")
        # plt.plot(test.index, test["value"], color="b")

    utility_index_cv_splits.pop(0)
    return utility_index_cv_splits


def create_data_frame(values, last_date):
    dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=len(values), freq="MS")
    predicted_df = pd.DataFrame({"value": values}, index=dates)
    return predicted_df


def naive_prediction(train_df, observation_to_predict, **kwargs):
    values = [train_df.iat[-1, 0] for i in range(observation_to_predict)]
    return create_data_frame(values, train_df.index[-1])


def average_prediction(train_df, observation_to_predict, **kwargs):
    values = [train_df["value"].mean() for i in range(observation_to_predict)]
    return create_data_frame(values, train_df.index[-1])


def sarima_prediction(train_df, observation_to_predict, **kwargs):
    sarima_model = statsmodels_sarimax.SARIMAX(train_df, order=kwargs["order"], seasonal_order=kwargs["seasonal_order"], freq="MS")
    sarima_model_fit = sarima_model.fit(disp=False)
    values = sarima_model_fit.forecast(observation_to_predict)
    return create_data_frame(values, train_df.index[-1])


def make_cv_predictions(cv_splits, model, **kwargs):
    predictions = []
    for train_df, test_df in cv_splits:
        predicted_df = model(train_df, len(test_df), **kwargs)
        predictions.append(predicted_df)
    return predictions


def get_cv_errors(cv_splits, predictions):
    errors = {"MAE": [], "MSE": [], "RMSLE": []}
    for z in range(len(predictions)):
        test_df = cv_splits[z][1]
        predicted_df = predictions[z]
        errors["MAE"].append(sk_metrics.mean_absolute_error(test_df["value"], predicted_df["value"]))
        errors["MSE"].append(sk_metrics.mean_squared_error(test_df["value"], predicted_df["value"]))
        errors["RMSLE"].append(math.sqrt(sk_metrics.mean_squared_log_error(test_df["value"], predicted_df["value"])))
    for error_type, error_list in errors.items():
        errors[error_type] = np.mean(error_list)
    return errors


def plot_cv_predictions(predictions):
    for prediction in predictions:
        plt.plot(prediction.index, prediction["value"], color="y")


plt.figure(figsize=(20, 10))

utility_index_df = pd.read_csv("data/IPG2211A2N.csv", parse_dates=["DATE"])
utility_index_df.rename(columns={"DATE": "date", "IPG2211A2N": "value"}, inplace=True)
utility_index_df.set_index("date", inplace=True)

utility_index_df = utility_index_df[
    (utility_index_df.index >= pd.Timestamp("1980-01-01")) & (utility_index_df.index < pd.Timestamp("2020-01-01"))]

print(len(utility_index_df))
print(utility_index_df.index.min())
print(utility_index_df.index.max())

plt.plot(utility_index_df.index, utility_index_df["value"])

number_of_splits = 6
utility_index_cv_splits = make_cv_splits(utility_index_df, number_of_splits)

naive_predictions = make_cv_predictions(utility_index_cv_splits, naive_prediction)
naive_errors = get_cv_errors(utility_index_cv_splits, naive_predictions)
print("Naive errors:", naive_errors)
plot_cv_predictions(naive_predictions)

average_predictions = make_cv_predictions(utility_index_cv_splits, average_prediction)
average_errors = get_cv_errors(utility_index_cv_splits, average_predictions)
print("Average errors:", average_errors)
plot_cv_predictions(average_predictions)

sarima_order_kwargs = {"order": (1, 1, 1), "seasonal_order": (1, 1, 1, 12)}
sarima_predictions = make_cv_predictions(utility_index_cv_splits, sarima_prediction, **sarima_order_kwargs)
sarima_errors = get_cv_errors(utility_index_cv_splits, sarima_predictions)
print("SARIMA errors:", sarima_errors)
plot_cv_predictions(sarima_predictions)

sarima_extrapolation = sarima_prediction(utility_index_df, 80, **sarima_order_kwargs)
plt.plot(sarima_extrapolation.index, sarima_extrapolation["value"], color="g")

utility_index_additive_decomposition = statsmodels_seasonal.seasonal_decompose(utility_index_df, model="additive",
                                                                               period=12)
utility_index_additive_decomposition.plot()
utility_index_multiplicative_decomposition = statsmodels_seasonal.seasonal_decompose(utility_index_df,
                                                                                     model="multiplicative", period=12)
utility_index_multiplicative_decomposition.plot()

plt.show()
