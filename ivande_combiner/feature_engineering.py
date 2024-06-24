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


def detect_anomaly(df: pd.DataFrame | pd.Series):
    if isinstance(df, pd.Series):
        df = df.to_frame()

    threshold = 1e307
    anomaly_details = []

    # check for NaNs in each column
    nan_columns = df.columns[df.isna().any()].tolist()
    if nan_columns:
        anomaly_details.append(f"NaN values found in columns: {nan_columns}")

    # check for infinite values in each column
    inf_columns = df.columns[np.isinf(df).any()].tolist()
    if inf_columns:
        anomaly_details.append(f"infinite values found in columns: {inf_columns}")

    # check for values exceeding the threshold
    large_value_columns = df.columns[(df.abs() > threshold).any()].tolist()
    if large_value_columns:
        anomaly_details.append(f"values exceeding threshold found in columns: {large_value_columns}")

    # if any anomalies are found, raise a ValueError with the details
    if anomaly_details:
        error_message = "anomalies detected:\n" + "\n".join(anomaly_details)
        raise ValueError(error_message)


def expand_df(df: pd.DataFrame, n: int, copy: bool = False) -> pd.DataFrame:
    """
    Expands a pandas DataFrame by duplicating each row 'n' times.

    Parameters:
    df (pd.DataFrame): The DataFrame to be expanded.
    n (int): The number of times each row in the DataFrame should be duplicated.
    copy (bool): If True, returns a copy of the expanded DataFrame. Defaults to False.

    Returns:
    pd.DataFrame: The expanded DataFrame.

    Raises:
    ValueError: If 'n' is not a non-negative integer.
    TypeError: If 'df' is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The 'df' argument must be a pandas DataFrame.")

    if not isinstance(n, int) or n < 0:
        raise ValueError("The 'n' argument must be a non-negative integer.")

    if n == 0:
        return pd.DataFrame(columns=df.columns) if copy else df.iloc[0:0]

    if n == 1:
        return df.copy() if copy else df

    expand_idx = np.repeat(df.index, n)
    expanded_df = df.loc[expand_idx].reset_index(drop=True)

    return expanded_df.copy() if copy else expanded_df
