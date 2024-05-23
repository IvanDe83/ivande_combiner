import pandas as pd
import pytest

from ivande_combiner.transformers import CalendarExtractor, NoInfoFeatureRemover, OutlierRemover


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
