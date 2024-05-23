import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CalendarExtractor(BaseEstimator, TransformerMixin):
    """
    extract number data from date column and them to the pandas dataframe
    """
    def __init__(self, date_col, calendar_level: int = None):
        """
        :param date_col: column with dates
        :param calendar_level: from 0 to 5,
        0 - only year,
        1 - year and month,
        2 - year, month, day,
        3 - year, month, day, dayofweek,
        4 - year, month, day, dayofweek, dayofyear,
        5 - year, month, day, dayofweek, dayofyear, weekofyear
        """
        self.date_col = date_col
        self.calendar_level = calendar_level
        self.what_to_generate = ["year", "month", "day", "dayofweek", "dayofyear", "weekofyear"]
        if self.calendar_level is not None:
            self.what_to_generate = self.what_to_generate[: self.calendar_level]

    def fit(self, X, y=None) -> "CalendarExtractor":
        """
        legacy from parent class
        :param X:
        :param y:
        :return: self
        """
        return self

    def transform(self, X) -> pd.DataFrame:
        """
        transform input pandas dataframe into new pandas dataframe with new columns
        :param X: input pandas dataframe
        :return: transformed pandas dataframe with new columns
        """
        X_ = X.copy()
        X_[self.date_col] = pd.to_datetime(X_[self.date_col])
        cols_to_add = []

        for what in self.what_to_generate:
            if what == "year":
                year_col = X_[self.date_col].dt.year
                year_col.name = "year"
                cols_to_add.append(year_col)
            elif what == "month":
                month_col = X_[self.date_col].dt.month
                month_col.name = "month"
                cols_to_add.append(month_col)
            elif what == "day":
                day_col = X_[self.date_col].dt.day
                day_col.name = "day"
                cols_to_add.append(day_col)
            elif what == "dayofweek":
                dayofweek_col = X_[self.date_col].dt.dayofweek
                dayofweek_col.name = "dayofweek"
                cols_to_add.append(dayofweek_col)
            elif what == "dayofyear":
                dayofyear_cl = X_[self.date_col].dt.dayofyear
                dayofyear_cl.name = "dayofyear"
                cols_to_add.append(dayofyear_cl)
            elif what == "weekofyear":
                weekofyear_col = X_[self.date_col].dt.isocalendar().week
                weekofyear_col.name = "weekofyear"
            else:
                raise ValueError(f"Unknown parameter {what} in what_to_generate")

        X_.drop(self.date_col, axis=1, inplace=True)
        X_ = pd.concat([X_, *cols_to_add], axis=1)

        return X_


class NoInfoFeatureRemover(BaseEstimator, TransformerMixin):
    """
    remove columns with the same values along all rows
    """
    def __init__(self, cols_to_except: list[str] = None, verbose=False):
        self.cols_to_remove = []
        self.cols_to_except = cols_to_except if cols_to_except is not None else []
        self.verbose = verbose

    def fit(self, X, y=None) -> "NoInfoFeatureRemover":
        """
        define what columns to remove
        :param X:
        :param y:
        :return: self
        """
        self.cols_to_remove = []

        for col in X.columns:
            if X[col].nunique() <= 1 and col not in self.cols_to_except:
                self.cols_to_remove.append(col)

        if self.verbose and self.cols_to_remove:
            print(f"Columns {self.cols_to_remove} have no info and will be removed")

        return self

    def transform(self, X) -> pd.DataFrame:
        """
        remove columns with no info
        :param X:
        :return: transformed pandas dataframe
        """
        X_ = X.drop(self.cols_to_remove, axis=1)
        return X_


class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_transform: list[str] = None, method: str = "iqr"):
        self.cols_to_transform = [] if cols_to_transform is None else cols_to_transform
        self.method = method
        self.col_thresholds = {}

    def fit(self, X, y=None):
        self.cols_to_transform = [col for col in self.cols_to_transform if col in X.columns]
        self.col_thresholds = {}

        for col in self.cols_to_transform:
            if self.method == "iqr":
                q1 = X[col].quantile(.25)
                q3 = X[col].quantile(.75)
                iqr = q3 - q1
                left_bound = q1 - 1.5 * iqr
                right_bound = q3 + 1.5 * iqr
            elif self.method == "std":
                mean = X[col].mean()
                std = X[col].std()
                left_bound = mean - 3 * std
                right_bound = mean + 3 * std
            elif self.method == "quantile":
                left_bound = X[col].quantile(.01)
                right_bound = X[col].quantile(.99)
            elif self.method == "skip":
                left_bound = X[col].min()
                right_bound = X[col].max()
            else:
                raise ValueError(f"unknown method {self.method} for outlier removing")

            s = X[col][(X[col] >= left_bound) & (X[col] <= right_bound)]
            self.col_thresholds[col] = (s.min(), s.max())

        return self

    def transform(self, X):
        X_ = X.copy()

        for col in self.cols_to_transform:
            X_[col] = X_[col].clip(*self.col_thresholds[col])

        return X_
