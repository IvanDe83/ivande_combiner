import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, RobustScaler, StandardScaler

from .utils import check_fill, check_key_tuple_empty_intersection, check_transform


class CalendarExtractor(BaseEstimator, TransformerMixin):
    """
    extract number data from date column and them to the pandas dataframe

    :param date_col: column with dates
    :param calendar_level: from 0 to 5,
        0 - only year,
        1 - year and month,
        2 - year, month, day,
        3 - year, month, day, dayofweek,
        4 - year, month, day, dayofweek, dayofyear,
        5 - year, month, day, dayofweek, dayofyear, weekofyear
    """
    def __init__(self, date_col, calendar_level: int = None):
        self.date_col = date_col
        self.calendar_level = calendar_level

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        check_transform(X, is_check_fill=False)

        X_ = X.copy()
        X_[self.date_col] = pd.to_datetime(X_[self.date_col])
        cols_to_add = []

        what_to_generate = ["year", "month", "day", "dayofweek", "dayofyear", "weekofyear"]
        if self.calendar_level is not None:
            what_to_generate = what_to_generate[: self.calendar_level]

        for what in what_to_generate:
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

    :param cols_to_except: list of columns that should not be removed
    :param verbose: True if you want to see the list of removed columns
    """
    def __init__(self, cols_to_except: list[str] = None, verbose=False):
        self.cols_to_remove = None
        self.cols_to_except = cols_to_except if cols_to_except is not None else []
        self.verbose = verbose

    def fit(self, X, y=None):
        check_fill(X)
        self.cols_to_remove = []

        for col in X.columns:
            if X[col].nunique() <= 1 and col not in self.cols_to_except:
                self.cols_to_remove.append(col)

        if self.verbose and self.cols_to_remove:
            print(f"Columns {self.cols_to_remove} have no info and will be removed")

        return self

    def transform(self, X):
        check_transform(X, fitted_item=self.cols_to_remove, transformer_name="NoInfoFeatureRemover")
        X_ = X.drop(self.cols_to_remove, axis=1)
        return X_


class OutlierRemover(BaseEstimator, TransformerMixin):
    """
    remove outliers from the columns

    :param cols_to_transform: list of column names from which to remove outliers
    :param method:
    "iqr" - remove outliers by interquartile range
    "std" - remove outliers by standard deviation
    "quantile" - remove outliers by quantile 0.01 and 0.99
    "skip" - do not remove outliers
    """
    def __init__(self, cols_to_transform: list[str], method: str = "iqr"):
        if cols_to_transform is None:
            raise ValueError("cols_to_transform parameter is should be filled")
        self.cols_to_transform = cols_to_transform
        self.method = method
        self.col_thresholds = None

    def fit(self, X, y=None):
        check_fill(X)
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
                raise ValueError(f"unknown method {self.method} for outlier remover")

            s = X[col][(X[col] >= left_bound) & (X[col] <= right_bound)]
            self.col_thresholds[col] = (s.min(), s.max())

        return self

    def transform(self, X):
        check_transform(X, fitted_item=self.col_thresholds, transformer_name="OutlierRemover")
        X_ = X.copy()

        for col in self.cols_to_transform:
            X_[col] = X_[col].clip(*self.col_thresholds[col])

        return X_


class WithAnotherColumnImputer(BaseEstimator, TransformerMixin):
    """
    impute missing values in one column with values from another column

    :param cols_to_impute: dictionary with column names as keys and column names to impute from as values
    """
    def __init__(self, cols_to_impute: dict[str, str] = None):
        if cols_to_impute is None:
            raise ValueError("cols_to_impute parameter is should be filled")
        self.cols_to_impute = cols_to_impute

    def fit(self, X, y=None):
        check_fill(X)
        self.cols_to_impute = {col: self.cols_to_impute[col] for col in self.cols_to_impute if col in X.columns}
        return self

    def transform(self, X):
        check_transform(X, fitted_item=self.cols_to_impute, transformer_name="WithAnotherColumnImputer")
        X_ = X.copy()

        for col in self.cols_to_impute:
            X_[col] = X_[col].fillna(X_[self.cols_to_impute[col]])

        return X_


class CatCaster(BaseEstimator, TransformerMixin):
    """
    cast columns to category type

    :param cols_to_cast: list of columns to cast to category type
    """
    def __init__(self, cols_to_cast: list[str]):
        self.cols_to_cast = cols_to_cast

    def fit(self, X, y=None):
        check_fill(X)
        self.cols_to_cast = [col for col in self.cols_to_cast if col in X.columns]
        return self

    def transform(self, X) -> pd.DataFrame:
        check_transform(X, is_check_fill=False)
        X_ = X.copy()
        X_[self.cols_to_cast] = X[self.cols_to_cast].astype("category")
        return X_


class FeaturesOrder(BaseEstimator, TransformerMixin):
    """
    order features in the same order as in the order_features list

    :param features_order: list of columns in the order you want them to be
    """
    def __init__(self, features_order: list[str]):
        self.features_order = features_order
        self.features_order_ = None

    def fit(self, X, y=None):
        check_fill(X)
        self.features_order_ = [col for col in self.features_order if col in X.columns]
        self.features_order_ += [col for col in X.columns if col not in self.features_order_]
        return self

    def transform(self, X):
        check_transform(X, fitted_item=self.features_order_, transformer_name="OrderFeatures")
        X_ = X[self.features_order_]
        return X_


class ScalerPicker(BaseEstimator, TransformerMixin):
    """
    scale columns with a scaler of your choice

    :param cols_to_scale: list of columns to scale
    :param scaler_type:
        "standard" - StandardScaler
        "minmax" - MinMaxScaler
    """
    def __init__(self, cols_to_scale: list[str], scaler_type: str = "standard"):
        self.cols_to_scale = cols_to_scale
        self.scaler_type = scaler_type
        self.scaler = None

    def _get_scaler_class(self):
        if self.scaler_type == "standard":
            return StandardScaler
        elif self.scaler_type == "minmax":
            return MinMaxScaler
        elif self.scaler_type == "robust":
            return RobustScaler
        elif self.scaler_type == "power":
            return PowerTransformer
        elif self.scaler_type == "skip":
            return None
        else:
            raise ValueError(f"unknown scaler type {self.scaler_type} should be standard or minmax")

    def fit(self, X, y=None):
        check_fill(X)
        self.cols_to_scale = [col for col in self.cols_to_scale if col in X.columns]
        scaler = self._get_scaler_class()
        if scaler:
            self.scaler = scaler().fit(X[self.cols_to_scale])
            self.scaler = scaler().set_output(transform="pandas").fit(X[self.cols_to_scale])
        else:
            self.scaler = "skip"
        return self

    def transform(self, X):
        check_transform(X, fitted_item=self.scaler, transformer_name="CustomScaler")
        X_ = X.copy()
        if self.scaler == "skip":
            return X_
        X_[self.cols_to_scale] = self.scaler.transform(X_[self.cols_to_scale])
        return X_


class SimpleImputerPicker(BaseEstimator, TransformerMixin):
    """
    impute missing values with a SimpleImputer

    :param strategy: strategy for SimpleImputer. Possible values: "constant", "mean", "median", "most_frequent", "max"
    :param cols_to_impute: dictionary with tuple column names as keys and fill values as values
        (only for strategy="constant")
    """
    def __init__(self, strategy: str = "constant", cols_to_impute: dict[tuple[str, ...], int] | list[str] = None):
        if cols_to_impute is not None and isinstance(cols_to_impute, dict):
            check_key_tuple_empty_intersection(cols_to_impute)
        self.cols_to_impute = cols_to_impute
        self.strategy = strategy
        self.imputer = None

    def fit(self, X, y=None):
        check_fill(X)
        if X.isnull().all().any():
            raise ValueError("there are columns with all missing values")

        if self.cols_to_impute is None:
            self.cols_to_impute = X.columns

        if self.strategy == "constant":
            self.imputer = {}
            for cols, fill_value in self.cols_to_impute.items():
                cols_to_impute = [col for col in cols if col in X.columns]
                if len(cols_to_impute) != 0:
                    self.imputer[tuple(cols_to_impute)] = (
                        SimpleImputer(strategy="constant", fill_value=fill_value, keep_empty_features=True)
                        .set_output(transform="pandas")
                        .fit(X[cols_to_impute])
                    )
        elif self.strategy in ("mean", "median", "most_frequent"):
            cols_to_impute = [col for col in self.cols_to_impute if col in X.columns]
            self.imputer = (
                SimpleImputer(strategy=self.strategy, keep_empty_features=True)
                .set_output(transform="pandas")
                .fit(X[cols_to_impute])
            )
        elif self.strategy == "max":
            cols_to_impute = [col for col in self.cols_to_impute if col in X.columns]
            self.imputer = {}

            for col in cols_to_impute:
                self.imputer[col] = (
                    SimpleImputer(strategy="constant", fill_value=X[col].max(), keep_empty_features=True)
                    .set_output(transform="pandas")
                    .fit(X[[col]])
                )
        else:
            raise ValueError(f"unknown strategy {self.strategy} should be constant, mean, median or most_frequent")

        return self

    def transform(self, X):
        check_transform(X, fitted_item=self.imputer, transformer_name="ConstantImputer")
        X_ = X.copy()

        if self.strategy == "constant":
            for cols, imputer in self.imputer.items():
                cols = list(cols)
                X_[cols] = imputer.transform(X_[cols])
        elif self.strategy == "max":
            for col, imputer in self.imputer.items():
                X_[col] = imputer.transform(X_[[col]])
        else:
            X_ = self.imputer.transform(X_)

        return X_
