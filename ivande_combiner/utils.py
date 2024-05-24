import pandas as pd
from sklearn.exceptions import NotFittedError


def check_fill(X: any) -> None:
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X is not pandas DataFrame")


def check_transform(X: any, fitted_item: any = None, transformer_name: str = "", is_check_fill: bool = True) -> None:
    check_fill(X)

    if is_check_fill and not fitted_item:
        raise NotFittedError(f"{transformer_name} transformer was not fitted")


def check_key_tuple_empty_intersection(dict_data: dict[tuple, any]):
    all_elements = set()
    total_elements = 0

    for key in dict_data.keys():
        all_elements.update(key)
        total_elements += len(key)

    if len(all_elements) != total_elements:
        raise ValueError("some keys have intersection between them")
