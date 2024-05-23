import pandas as pd
from sklearn.exceptions import NotFittedError


def check_fill(X: any) -> None:
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X is not pandas DataFrame")


def check_transform(X: any, fitted_item: any = None, transformer_name: str = "", is_check_fill: bool = True) -> None:
    check_fill(X)

    if is_check_fill and not fitted_item:
        raise NotFittedError(f"{transformer_name} transformer was not fitted")
