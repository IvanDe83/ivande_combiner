import os

import pandas as pd
import requests

from .ts_utils import add_row_to_df, extract_years


class HolidayGenerator:
    """
    args:
        country (str): country code (e.g., "KZ").
        start_ds (str): start date in "YYYY-MM-DD" format.
        end_ds (str): end date in "YYYY-MM-DD" format.
    """
    def __init__(self, country: str, start_ds: str, end_ds: str):
        self.country = country
        self.start_ds = pd.to_datetime(start_ds).date()
        self.end_ds = pd.to_datetime(end_ds).date()
        self.df = pd.DataFrame()

    def _normalize(self) -> None:
        self.df.drop_duplicates(inplace=True)
        self.df["ds"] = pd.to_datetime(self.df["ds"]).dt.date
        self.df = self.df[(self.df["ds"] >= self.start_ds) & (self.df["ds"] <= self.end_ds)]
        self.df.sort_values("ds", inplace=True)

    def get_holidays(self) -> pd.DataFrame:
        """
        fetch holidays from Calendarific API for a given country and range of years
        """
        if len(self.df) == 0:
            self.generate()

        self._normalize()

        return self.df

    def generate(self) -> None:
        """
        fetch holidays from Calendarific API for a given country and range of years
        """
        base_url = os.getenv("CALENDARIFIC_BASE_URL")
        if base_url is None:
            raise ValueError("CALENDARIFIC_BASE_URL environment variable is not set")

        api_key = os.getenv("CALENDARIFIC_API_KEY")
        if api_key is None:
            raise ValueError("CALENDARIFIC_API_KEY environment variable is not set")

        years = extract_years(self.start_ds, self.end_ds)

        all_holidays = []

        for year in years:
            params = {
                "api_key": api_key,
                "country": self.country,
                "year": year,
            }
            response = requests.get(base_url, params=params)

            if response.status_code == 200:
                data = response.json()
                for holiday in data["response"]["holidays"]:
                    all_holidays.append(
                        {
                            "holiday": holiday["name"],
                            "ds": holiday["date"]["iso"],
                            "type": tuple(holiday["type"]),
                        }
                    )
            else:
                print(
                    f"failed to fetch data for {year}, country {self.country}: {response.status_code}, {response.text}"
                )

        df = pd.DataFrame(all_holidays)

        if self.country == "KZ":
            ramadan_2023 = ["Ramadan starts", "2023-03-22", ("Muslim",)]
            df = add_row_to_df(df, a=ramadan_2023)

        self.df = df.drop_duplicates()
        self.df["ds"] = self.df["ds"].str.split("T").str[0]
        self.df["ds"] = pd.to_datetime(self.df["ds"]).dt.date
        self._normalize()

    def filter_holidays(self, exclude_types: list[str]) -> None:
        """
        filter holidays by a list of holiday types
        """
        self.df = self.df.loc[~self.df["type"].apply(lambda x: all(t in exclude_types for t in x)), ["holiday", "ds"]]

    def add_influence(self, windows: dict[str, tuple] = None) -> None:
        """
        add influence windows for holidays

        args:
            windows (dict[str, tuple], optional): a dictionary with holiday names as keys and tuples of lower and upper
            window sizes as values
        """
        self.df["lower_window"] = 0
        self.df["upper_window"] = 0

        if windows is not None:
            self.df["lower_window"] = (
                self.df["holiday"]
                .map(
                    lambda x: windows[x][0]
                    if x in windows
                    else self.df.loc[self.df["holiday"] == x, "lower_window"].values[0]
                )
            )
            self.df["upper_window"] = (
                self.df["holiday"]
                .map(
                    lambda x: windows[x][1]
                    if x in windows
                    else self.df.loc[self.df["holiday"] == x, "upper_window"].values[0]
                )
            )

    def add_external_holidays(self, df: pd.DataFrame) -> None:
        """
        add external holidays to the existing DataFrame
        """
        self.df = pd.concat([self.df, df], ignore_index=True)
        self.df["ds"] = pd.to_datetime(self.df["ds"])
        self.df["ds"] = self.df["ds"].dt.date
        self._normalize()
