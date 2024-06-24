import numpy as np
import pandas as pd
import pytest

from ivande_combiner.feature_engineering import detect_anomaly, expand_df, generate_ratio


class TestFeatureEngineering:
    @pytest.mark.parametrize(
        "first_col, second_col, expected_output",
        [
            (
                [1, 4, 9, 16, 25],
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5],
            ),
            (
                [1, 4, 0, 16, 0],
                [1, 2, 3, 4, 5],
                [1, 2, 0, 4, 0],
            ),
            (
                [1, 4, 9, 16, 25],
                [1, 2, 0, 4, 0],
                [1, 2, 4, 4, 4],
            ),
            (
                [1, 4, 0, 0, 25],
                [1, 2, 0, 4, 0],
                [1, 2, 0, 0, 2],
            ),
            (
                [0, 0, 0, 0, 0],
                [1, 2, 0, 4, 0],
                [0, 0, 0, 0, 0],
            ),
            (
                [1, 4, 0, 0, 25],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ),
            (
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ),
        ],
        ids=[
            "all non zero",
            "first col has some zero",
            "second col has some zero",
            "both cols have zero at the same index",
            "first col has all zeros",
            "second col has all zeros",
            "both cols have all zeros",
        ],
    )
    def test_generate_ratio(self, first_col, second_col, expected_output):
        first_s = pd.Series(first_col)
        second_s = pd.Series(second_col)
        expected = pd.Series(expected_output, dtype=float)
        calculated = generate_ratio(first_s, second_s)
        assert calculated.equals(expected)

    def test_can_catch_different_length_columns_error(self):
        first_col = pd.Series([1, 2, 3, 4, 5])
        second_col = pd.Series([1, 2, 3, 4])
        with pytest.raises(ValueError) as excinfo:
            generate_ratio(first_col, second_col)
        assert "both columns must be of the same length" in str(excinfo.value)


class TestDetectAnomaly:
    @pytest.mark.parametrize(
        "input_data",
        [
            (
                {"col_1": [1, 2, 3], "col_2": [4, None, 6]}
            ),
            (
                {"col_1": [1, 2, 3], "col_2": [4, np.inf, 6]}
            ),
            (
                {"col_1": [1, 2, 3], "col_2": [4, 1e308, 6]}
            ),
        ],
        ids=[
            "detect_nan",
            "detect_inf",
            "detect_too_large",
        ],
    )
    def test_detect_anomaly(self, input_data):
        df = pd.DataFrame(input_data)
        with pytest.raises(ValueError) as excinfo:
            detect_anomaly(df)
        assert "anomalies detected" in str(excinfo.value)
        assert "col_2" in str(excinfo.value)

    def test_no_anomaly_error(self):
        df = pd.DataFrame({"col_1": [1, 2, 3], "col_2": [4, 5, 6]})
        detect_anomaly(df)


class TestExpandDf:
    @pytest.mark.parametrize(
        "input_data, expected_output, n",
        [
            (
                {"col_1": [1, 2, 3], "col_2": [4, 5, 6]},
                {"col_1": [], "col_2": []},
                0,
            ),
            (
                {"col_1": [1, 2, 3], "col_2": [4, 5, 6]},
                {"col_1": [1, 2, 3], "col_2": [4, 5, 6]},
                1,
            ),
            (
                {"col_1": [1, 2, 3], "col_2": [4, 5, 6]},
                {"col_1": [1, 1, 1, 2, 2, 2, 3, 3, 3], "col_2": [4, 4, 4, 5, 5, 5, 6, 6, 6]},
                3,
            ),
        ],
        ids=[
            "n=0",
            "n=1",
            "n=3",
        ],
    )
    def test_expand_df(self, input_data, expected_output, n):
        df = pd.DataFrame(input_data)
        expected = pd.DataFrame(expected_output)
        calculated = expand_df(df, n)

        pd.testing.assert_frame_equal(expected, calculated, check_dtype=False)
