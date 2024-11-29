from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import os

import numpy as np
import pandas as pd


def extract_years(start_ds: str | date | datetime, end_ds: str | date | datetime) -> list[int]:
    """
    extract all years between start_ds and end_ds (inclusive)

    args:
        start_ds (str): Start date in the format 'YYYY-MM-DD'
        end_ds (str): End date in the format 'YYYY-MM-DD'

    returns:
        list[int]: a list of years between the start and end dates
    """
    if isinstance(start_ds, str):
        start_ds = datetime.strptime(start_ds, "%Y-%m-%d")

    if isinstance(end_ds, str):
        end_ds = datetime.strptime(end_ds, "%Y-%m-%d")

    start_year = start_ds.year
    end_year = end_ds.year

    return list(range(start_year, end_year + 1))


def add_row_to_df(df: pd.DataFrame, a: list) -> pd.DataFrame:
    df.reset_index(inplace=True, drop=True)

    if len(a) != len(df.columns):
        raise ValueError(f"length of row {len(a)} does not match number of columns {len(df.columns)}")

    df.loc[len(df)] = a

    return df


def get_closest_same_day_of_week(date):
    shifted_date = date + relativedelta(years=1)
    days_difference = (date.weekday() - shifted_date.weekday()) % 7
    return shifted_date + timedelta(days=days_difference)


def extend_holidays_to_the_next_year(df: pd.DataFrame) -> pd.DataFrame:
    df["ds"] = pd.to_datetime(df["ds"])

    new_rows = []
    for _, row in df.iterrows():
        new_date = get_closest_same_day_of_week(row["ds"])
        new_row = row.copy()
        new_row["ds"] = new_date
        new_rows.append(new_row)

    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    return df.sort_values(by="ds").reset_index(drop=True)


def get_local_sales(query: str) -> pd.DataFrame:
    con = os.getenv("DWH_PG_CON")
    if con is None:
        raise ValueError("DWH_PG_CON environment variable is not set")

    sales_df = pd.read_sql(query, con=con)
    sales_df["lower_window"] = 0

    sales_df = sales_df[["holiday", "ds", "lower_window", "upper_window"]]

    return extend_holidays_to_the_next_year(sales_df)


def leave_only_full_month(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("ds")
    df.loc[:, "month"] = df["ds"].dt.to_period("M")

    first_ds = df["ds"].iloc[0]
    first_day = first_ds.day

    if first_day != 1:
        first_month = df.loc[df["ds"] == first_ds, "month"].squeeze()
        df = df[df["month"] != first_month]

    last_ds = df["ds"].iloc[-1]
    last_day = last_ds.day
    last_month_day = last_ds.days_in_month

    if last_day != last_month_day:
        last_month = df.loc[df["ds"] == last_ds, "month"].squeeze()
        df = df[df["month"] != last_month]

    return df


def prepare_ts_df(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[:, "ds"] = pd.to_datetime(df["ds"])
    df = leave_only_full_month(df)

    df["month"] = df["ds"].dt.to_period("M")

    return df.groupby("month")["y"].sum().reset_index()


def generate_metric_df(df_test: pd.DataFrame, df_pred: pd.DataFrame):
    df_test = prepare_ts_df(df_test)
    df_pred = prepare_ts_df(df_pred)

    return pd.merge(df_test, df_pred, on="month", suffixes=("_test", "_pred"))


def get_exp_values(n: int, decay_rate: float = 1.0) -> list[float]:
    """
    Generate exponentially decreasing values that sum to 1.

    Parameters:
    - n: Number of values to generate.
    - decay_rate: Controls the rate of exponential decay (smaller is smoother).

    Returns:
    - A list of normalized exponential values.
    """
    exp_vals = np.exp(-np.arange(n) * decay_rate)

    return (exp_vals / exp_vals.sum()).tolist()
