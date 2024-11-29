from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import os

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


def get_local_sales():
    con = os.getenv("DWH_PG_CON")
    if con is None:
        raise ValueError("DWH_PG_CON environment variable is not set")

    sales_query = """
        select 
            event as holiday,
            ds_start as ds,
            duration as upper_window
        from 
            dict.bank_sales        
    """

    sales_df = pd.read_sql(sales_query, con=con)
    sales_df["lower_window"] = 0

    sales_df = sales_df[["holiday", "ds", "lower_window", "upper_window"]]

    return extend_holidays_to_the_next_year(sales_df)
