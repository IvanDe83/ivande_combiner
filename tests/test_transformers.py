import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from ivande_combiner.transformers import (
    CalendarExtractor,
    CatCaster,
    NoInfoFeatureRemover,
    OrderFeatures,
    OutlierRemover,
    ScalerPicker,
    WithAnotherColumnImputer,
)


class TestCalendarExtractor:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.df = pd.DataFrame(
            {
                "date": ["2022-01-01", "2023-02-28"],
            }
        )

    def test_calendar_level_0(self):
        calendar_extractor = CalendarExtractor(date_col="date", calendar_level=0)
        calculated = calendar_extractor.fit_transform(self.df)
        assert calculated.empty

    def test_calendar_level_2(self):
        calendar_extractor = CalendarExtractor(date_col="date", calendar_level=2)
        expected = pd.DataFrame(
            {
                "year": [2022, 2023],
                "month": [1, 2],
            }
        )
        calculated = calendar_extractor.fit_transform(self.df)
        pd.testing.assert_frame_equal(expected, calculated, check_dtype=False)

    def test_calendar_level_5(self):
        calendar_extractor = CalendarExtractor(date_col="date", calendar_level=5)
        expected = pd.DataFrame(
            {
                "year": [2022, 2023],
                "month": [1, 2],
                "day": [1, 28],
                "dayofweek": [5, 1],
                "dayofyear": [1, 59],
            }
        )
        calculated = calendar_extractor.fit_transform(self.df)
        pd.testing.assert_frame_equal(expected, calculated, check_dtype=False)


class TestNoInfoFeatureRemover:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.df = pd.DataFrame(
            {
                "col_1": [1, 2],
                "no_info_1": [1, 1],
                "no_info_2": [2, 2],
            }
        )

    def test_no_info_feature_removed(self):
        no_info_feature_remover = NoInfoFeatureRemover()
        expected = pd.DataFrame(
            {
                "col_1": [1, 2],
            }
        )
        calculated = no_info_feature_remover.fit_transform(self.df)
        pd.testing.assert_frame_equal(expected, calculated, check_dtype=False)

    def test_can_except_columns(self):
        no_info_feature_remover = NoInfoFeatureRemover(cols_to_except=["no_info_1"])
        expected = pd.DataFrame(
            {
                "col_1": [1, 2],
                "no_info_1": [1, 1],
            }
        )
        calculated = no_info_feature_remover.fit_transform(self.df)
        pd.testing.assert_frame_equal(expected, calculated, check_dtype=False)

    def test_raise_error_if_wrong_type_in_fit(self):
        with pytest.raises(ValueError) as excinfo:
            NoInfoFeatureRemover().fit("wrong_type")
        assert "X is not pandas DataFrame" in str(excinfo.value)

    def test_raise_error_if_not_fitted(self):
        with pytest.raises(NotFittedError) as excinfo:
            NoInfoFeatureRemover().transform(self.df)
        assert "NoInfoFeatureRemover transformer was not fitted" in str(excinfo.value)


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


class TestOrderFeatures:
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
        t = OrderFeatures(order_features=expected[: -1])
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
