import numpy as np
import pandas as pd

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
        return float(
            np.sqrt(
                (
                    (self.df["y_test"] - self.df["y_pred"]) ** 2 * self.df["weights"]
                ).mean()
            )
        )

    def mae(self) -> float:
        """
        calculate MAE for each
        """
        return float(
            (
                (self.df["y_test"] - self.df["y_pred"]).abs() * self.df["weights"]
            ).mean()
        )

    def mape(self) -> float:
        """
        calculate MAPE for each
        """
        return float(
            (
                ((self.df["y_test"] - self.df["y_pred"]) / self.df["y_test"]).abs() * self.df["weights"]
            ).mean()
        )

    def smape(self) -> float:
        """
        calculate sMAPE for
        """
        return float(
            (
                (np.abs(self.df["y_test"] - self.df["y_pred"]) / (self.df["y_test"] + self.df["y_pred"]) / 2).mean()
            )
        )
