import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from ivande_combiner.transformers import (
    CalendarExtractor,
    CatCaster,
    ColsOrder,
    NoInfoColsRemover,
    OutlierRemover,
    ScalerPicker,
    SimpleImputerPicker,
    WithAnotherColumnImputer,
)


class TestCalendarExtractor:
    @pytest.mark.parametrize(
        "input_data, expected_output, calendar_level",
        [
            (
                    {"date": pd.to_datetime(["2022-01-01", "2023-02-28"]), "value": [1, 2]},
                    {"value": [1, 2]},
                    0,
            ),
            (
                    {"date": pd.to_datetime(["2022-01-01", "2023-02-28"]), "value": [1, 2]},
                    {"value": [1, 2], "date_year": [2022, 2023], "date_month": [1, 2]},
                    2,
            ),
            (
                    {"date": pd.to_datetime(["2022-01-01 23:59:59", "2023-02-28 02:03:04"]), "value": [1, 2]},
                    {
                        "value": [1, 2],
                        "date_year": [2022, 2023],
                        "date_month": [1, 2],
                        "date_day": [1, 28],
                        "date_dayofweek": [5, 1],
                        "date_dayofyear": [1, 59],
                        "date_weekofyear": [52, 9],
                        "date_hour": [23, 2],
                    },
                    7,
            ),
            (
                    {
                        "date1": pd.to_datetime(["2022-01-01 23:59:59", "2023-02-28 02:03:04"]),
                        "date2": pd.to_datetime(["2022-01-01 23:59:59", "2023-03-28 02:03:04"]),
                        "value": [1, 2],
                    },
                    {
                        "value": [1, 2],
                        "date1_year": [2022, 2023],
                        "date1_month": [1, 2],
                        "date1_day": [1, 28],
                        "date1_dayofweek": [5, 1],
                        "date1_dayofyear": [1, 59],
                        "date1_weekofyear": [52, 9],
                        "date1_hour": [23, 2],
                        "date2_year": [2022, 2023],
                        "date2_month": [1, 3],
                        "date2_day": [1, 28],
                        "date2_dayofweek": [5, 1],
                        "date2_dayofyear": [1, 87],
                        "date2_weekofyear": [52, 13],
                        "date2_hour": [23, 2],
                    },
                    None,
            ),
        ],
        ids=[
            "calendar_level_0",
            "calendar_level_2",
            "calendar_level_7",
            "calendar_level_None_with_two_date_columns",
        ],
    )
    def test_calendar(self, input_data, expected_output, calendar_level):
        df = pd.DataFrame(input_data)
        expected = pd.DataFrame(expected_output)
        calendar_extractor = CalendarExtractor(calendar_level=calendar_level)
        calculated = calendar_extractor.fit_transform(df)
        pd.testing.assert_frame_equal(expected, calculated, check_dtype=False)


class TestNoInfoColsRemover:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.df = pd.DataFrame(
            {
                "col_1": [1, 2],
                "no_info_1": [1, 1],
                "no_info_2": [2, 2],
            }
        )

    def test_no_info_cols_removed(self):
        t = NoInfoColsRemover()
        expected = pd.DataFrame(
            {
                "col_1": [1, 2],
            }
        )
        calculated = t.fit_transform(self.df)
        pd.testing.assert_frame_equal(expected, calculated, check_dtype=False)

    def test_can_except_columns(self):
        t = NoInfoColsRemover(cols_to_except=["no_info_1"])
        expected = pd.DataFrame(
            {
                "col_1": [1, 2],
                "no_info_1": [1, 1],
            }
        )
        calculated = t.fit_transform(self.df)
        pd.testing.assert_frame_equal(expected, calculated, check_dtype=False)

    def test_raise_error_if_wrong_type_in_fit(self):
        with pytest.raises(ValueError) as excinfo:
            NoInfoColsRemover().fit("wrong_type")
        assert "X is not pandas DataFrame" in str(excinfo.value)

    def test_raise_error_if_not_fitted(self):
        with pytest.raises(NotFittedError) as excinfo:
            NoInfoColsRemover().transform(self.df)
        assert "NoInfoColsRemover transformer was not fitted" in str(excinfo.value)


class TestOutlierRemover:
    @pytest.mark.parametrize(
        "input_data, expected_output, method",
        [
            (
                {"col_1": [-51] + list(range(1, 100)) + [151]},
                {"col_1": [1] + list(range(1, 100)) + [99]},
                "iqr",
            ),
            (
                {"col_1": [-50] + list(range(1, 100)) + [150]},
                {"col_1": [-50] + list(range(1, 100)) + [150]},
                "iqr",
            ),
            (
                {"col_1": [-100] + list(range(-50, 51)) + [100]},
                {"col_1": [-50] + list(range(-50, 51)) + [50]},
                "std",
            ),
            (
                {"col_1": [-75] + list(range(-50, 51)) + [75]},
                {"col_1": [-75] + list(range(-50, 51)) + [75]},
                "std",
            ),
            (
                {"col_1": range(101)},
                {"col_1": [1] + list(range(1, 100)) + [99]},
                "quantile",
            ),
            (
                {"col_1": [-1000] + list(range(100)) + [1000]},
                {"col_1": [-1000] + list(range(100)) + [1000]},
                "skip",
            ),
        ],
        ids=[
            "iqr_test_has_effect",
            "iqr_test_no_effect",
            "std_test_has_effect",
            "std_test_no_effect",
            "quantile_test_always_has_effect",
            "skip_test_never_has_effect",
        ],
    )
    def test_method_param(self, input_data, expected_output, method):
        df = pd.DataFrame(input_data)
        expected = pd.DataFrame(expected_output)
        outlier_remover = OutlierRemover(method=method, cols_to_transform=["col_1"])
        calculated = outlier_remover.fit_transform(df)
        pd.testing.assert_frame_equal(expected, calculated, check_dtype=False)

    def test_can_catch_method_error(self):
        df = pd.DataFrame({"col_1": [1]})
        with pytest.raises(ValueError) as excinfo:
            OutlierRemover(method="wrong_method", cols_to_transform=["col_1"]).fit_transform(df)
        assert "unknown method wrong_method for outlier remover" in str(excinfo.value)


class TestWithAnotherColumnImputer:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.df = pd.DataFrame(
            {
                "col_1": [1, 2, 3],
                "col_2": [5, None, None],
            }
        )

    def test_impute_col_2_with_col_1(self):
        imputer = WithAnotherColumnImputer(cols_to_impute={"col_2": "col_1"})
        expected = pd.DataFrame(
            {
                "col_1": [1, 2, 3],
                "col_2": [5, 2, 3],
            }
        )
        calculated = imputer.fit_transform(self.df)
        pd.testing.assert_frame_equal(expected, calculated, check_dtype=False)


class TestCatCaster:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.df = pd.DataFrame(
            {
                "col_1": [1, 2, 3],
                "col_2": ["a", "b", "c"],
                "col_3": [4, 5, 6],
            }
        )

    def test_correct_outcome_column_type(self):
        t = CatCaster(cols_to_cast=["col_2", "col_1", "col_0"])
        calculated = t.fit_transform(self.df)
        assert all(calculated[col].dtype == "category" for col in ["col_1", "col_2"])
        assert calculated["col_3"].dtype == "int64"


class TestColsOrder:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.df = pd.DataFrame(
            {
                "col_3": [1, 2, 3],
                "col_1": ["a", "b", "c"],
                "col_0": [7, 8, 9],
                "col_2": [4, 5, 6],
            }
        )

    def test_correct_outcome_order(self):
        expected = ["col_1", "col_2", "col_3", "col_0"]
        t = ColsOrder(cols_order=expected[: -1])
        calculated = t.fit_transform(self.df)
        assert expected == list(calculated.columns)


class TestScalerPicker:
    s1 = np.linspace(1, 10, 10)
    s2 = np.linspace(10, 28, 10)

    @pytest.mark.parametrize(
        "input_data, expected_output, scaler_type",
        [
            (
                {"col_1": range(1, 12), "col_2": range(10, 21), "col_3": range(11)},
                {"col_1": np.linspace(0, 1, 11), "col_2": np.linspace(0, 1, 11), "col_3": range(11)},
                "minmax",
            ),
            (
                {"col_1": range(1, 11), "col_2": range(10, 30, 2), "col_3": range(10)},
                {
                    "col_1": (s1 - s1.mean()) / s1.std(ddof=0),
                    "col_2": (s1 - s1.mean()) / s1.std(ddof=0),
                    "col_3": range(10),
                },
                "standard",
            ),
            (
                {"col_1": [1, -2, 2], "col_2": [4, 1, -2], "col_3": range(3)},
                {"col_1": [0, -1.5, .5], "col_2": [1, 0, -1], "col_3": range(3)},
                "robust",
            ),
            (
                {"col_1": [1, 2, 3], "col_2": [-5, 0, 3], "col_3": range(3)},
                {"col_1": [-1.252189, 0.05687, 1.195319], "col_2": [-1.233597, .017901, 1.215696], "col_3": range(3)},
                "power",
            ),
            (
                {"col_1": [1, 2, 3], "col_2": [-5, 0, 3], "col_3": range(3)},
                {"col_1": [1, 2, 3], "col_2": [-5, 0, 3], "col_3": range(3)},
                "skip",
            ),
        ],
        ids=[
            "minmax",
            "standard",
            "robust",
            "power",
            "skip",
        ],
    )
    def test_scaler_type_param(self, input_data, expected_output, scaler_type):
        df = pd.DataFrame(input_data)
        expected = pd.DataFrame(expected_output)
        t = ScalerPicker(scaler_type=scaler_type, cols_to_scale=["col_1", "col_2"])
        calculated = t.fit_transform(df)
        pd.testing.assert_frame_equal(expected, calculated, check_dtype=False, rtol=.0001)

    def test_can_catch_method_error(self):
        df = pd.DataFrame({"col_1": [1]})
        with pytest.raises(ValueError) as excinfo:
            OutlierRemover(method="wrong_method", cols_to_transform=["col_1"]).fit_transform(df)
        assert "unknown method wrong_method for outlier remover" in str(excinfo.value)


class TestSimpleImputerPicker:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.df_nan = pd.DataFrame(
            {
                "col_1": [1, 2, 3],
                "col_2": [None, None, None],
                "col_3": [4, None, None],
            }
        )

    @pytest.mark.parametrize(
        "input_data, expected_output, strategy, cols_to_impute",
        [
            (
                {"col_1": [1, None, 2], "col_2": [None, 3, None], "col_3": [None, 7, None], "col_4": [4, 5, 6]},
                {"col_1": [1, 0, 2], "col_2": [0, 3, 0], "col_3": [1, 7, 1], "col_4": [4, 5, 6]},
                "constant",
                {("col_1", "col_2"): 0, ("col_3", ): 1, ("col_4", ): 2},
            ),
            (
                {"col_1": [1, 2, 3, None], "col_2": [None, 1, 2, 12]},
                {"col_1": [1, 2, 3, 2], "col_2": [5, 1, 2, 12]},
                "mean",
                ["col_1", "col_2", "col_3"],
            ),
            (
                {"col_1": [1, 2, 3, None], "col_2": [None, 1, 2, 12]},
                {"col_1": [1, 2, 3, 2], "col_2": [2, 1, 2, 12]},
                "median",
                ["col_1", "col_2", "col_3"],
            ),
            (
                {"col_1": [1, 2, 1, None], "col_2": [None, 12, 2, 12]},
                {"col_1": [1, 2, 1, 1], "col_2": [12, 12, 2, 12]},
                "most_frequent",
                ["col_1", "col_2", "col_3"],
            ),
            (
                {"col_1": [1, 3, 2, None], "col_2": [None, None, 1, 4]},
                {"col_1": [1, 3, 2, 3], "col_2": [4, 4, 1, 4]},
                "max",
                ["col_1", "col_2", "col_3"],
            ),
            (
                {"col_1": [1, 3, 2, None], "col_2": [None, None, 1, 4]},
                {"col_1": [1, 3, 2, 2], "col_2": [2.5, 2.5, 1, 4]},
                "mean",
                None,
            ),
        ],
        ids=[
            "constant_impute",
            "mean_impute",
            "median_impute",
            "most_frequent_impute",
            "max_impute",
            "cols_to_impute_is_none",
        ],
    )
    def test_correct_impute(self, input_data, expected_output, strategy, cols_to_impute):
        df = pd.DataFrame(input_data)
        expected = pd.DataFrame(expected_output)
        t = SimpleImputerPicker(strategy=strategy, cols_to_impute=cols_to_impute)
        calculated = t.fit_transform(df)
        pd.testing.assert_frame_equal(expected, calculated, check_dtype=False)

    def test_can_catch_wrong_strategy(self):
        with pytest.raises(ValueError) as excinfo:
            df = self.df_nan.copy()
            df.loc[0, "col_2"] = 1
            SimpleImputerPicker(strategy="wrong_strategy").fit_transform(df)
        assert "unknown strategy wrong_strategy should be constant, mean, median or most_frequent" in str(excinfo.value)

    def test_can_catch_column_intersection_in_cols_to_impute(self):
        with pytest.raises(ValueError) as excinfo:
            cols_to_impute = {("col_1", "col_1"): 0, ("col_3", ): 1, ("col_4", ): 2}
            SimpleImputerPicker(cols_to_impute=cols_to_impute).fit_transform(self.df_nan)
        assert "some keys have intersection between them" in str(excinfo.value)

    def test_can_catch_if_any_column_has_all_nans(self):
        with pytest.raises(ValueError) as excinfo:
            t = SimpleImputerPicker(strategy="mean")
            t.fit_transform(self.df_nan)
        assert "columns with all nans ['col_2']" in str(excinfo.value)
