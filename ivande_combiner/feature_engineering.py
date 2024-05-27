import numpy as np
import pandas as pd


def generate_ratio(first_col: pd.Series, second_col: pd.Series) -> pd.Series:
    """
    generate ratio of two columns. Replace inf with max value of non-inf values

    :param first_col: pandas series divisible column
    :param second_col: pandas series divisor column
    :return: pandas series quotient of two columns
    """
    if len(first_col) != len(second_col):
        raise ValueError("both columns must be of the same length")

    if first_col.eq(0).all() or second_col.eq(0).all():
        return pd.Series(0, index=first_col.index, dtype=float)

    zero_indices = (first_col == 0) & (second_col == 0)
    s = first_col / second_col
    s[zero_indices] = 0

    if np.isinf(s).any():
        max_value_not_inf = s[np.isfinite(s)].max()
        s.replace([np.inf, -np.inf], max_value_not_inf, inplace=True)

    return s
