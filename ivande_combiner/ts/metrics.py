from copy import deepcopy

import numpy as np
import pandas as pd
from prophet import Prophet

from .ts_utils import generate_metric_df, get_exp_values


class TSMetric:
    """
    weighted monthly metrics
    """
    def __init__(self, df_test: pd.DataFrame, df_pred: pd.DataFrame, decay_rate: float = 1.0):
        self.df = generate_metric_df(df_test, df_pred)
        self.df["weights"] = get_exp_values(n=len(self.df), decay_rate=decay_rate)

    def rmse(self) -> float:
        """
        calculate MSE for each
        """
        return float(np.sqrt(sum((self.df["y_test"] - self.df["y_pred"]) ** 2 * self.df["weights"])))

    def mae(self) -> float:
        """
        calculate MAE for each
        """
        return float(sum((self.df["y_test"] - self.df["y_pred"]).abs() * self.df["weights"]))

    def mape(self) -> float:
        """
        calculate MAPE for each
        """
        return float(sum(((self.df["y_test"] - self.df["y_pred"]) / self.df["y_test"]).abs() * self.df["weights"]))

    def smape(self) -> float:
        """
        calculate sMAPE for
        """
        return abs(float(sum(((self.df["y_test"] - self.df["y_pred"]) / (self.df["y_test"] + self.df["y_pred"]) / 2))))


def evaluate_prophet_on_months(prophet_model: Prophet, df: pd.DataFrame, horizon: int = 12, n: int = 6) -> dict:
    first_days = np.sort(df["ds"].dt.to_period("M").dt.to_timestamp().unique()).astype("datetime64[D]")

    metrics_dict = {
        "RMSE": 0,
        "MAE": 0,
        "MAPE": 0,
        "SMAPE": 0,
    }

    for i in range(n):
        fd_l = first_days[-horizon - i - 1]
        fd_r = first_days[- i - 1]
        df_train = df[df["ds"] < fd_l]
        df_test = df[(df["ds"] >= fd_l) & (df["ds"] < fd_r)]

        m = deepcopy(prophet_model)
        m.fit(df_train)
        df_forecast = m.make_future_dataframe(periods=len(df_test))

        df_pred = m.predict(df_forecast)
        df_pred["ds"] = pd.to_datetime(df_forecast["ds"]).astype("datetime64[ns]")
        df_pred = df_pred.loc[df_pred["ds"] >= fd_l, ["ds", "yhat"]]
        df_pred.rename(columns={"yhat": "y"}, inplace=True)

        tsm = TSMetric(df_test, df_pred)

        metrics_dict["RMSE"] += tsm.rmse()
        metrics_dict["MAE"] += tsm.mae()
        metrics_dict["MAPE"] += tsm.mape()
        metrics_dict["SMAPE"] += tsm.smape()

        for key in metrics_dict:
            metrics_dict[key] /= n

    return metrics_dict
