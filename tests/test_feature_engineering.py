import pandas as pd
import pytest

from ivande_combiner.feature_engineering import generate_ratio


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
