from typing import Optional, Set

import pandas as pd
import numpy as np

import utils
import constants as const


def nan_to_str_fallback(value: Optional[str]) -> str:
    """
    Returns the input value if it string, otherwise 'unknown'
    Useful for columns that might contain NaN
    """
    return value if isinstance(value, str) else const.UNKNOWN_STR


def set_nans_to_median(df_original: pd.DataFrame) -> pd.DataFrame:
    """Sets all the nans in the df to the median of the column"""
    df = df_original.copy()
    for col in df_original.columns:
        nans = np.isnan(df_original[col])
        df.loc[nans, col] = df_original[col].median()
    return df


def category_to_set(category: str) -> Set[str]:
    """
    Converts a category string like "parf car, premium ad car, low mileage car"
    to a set of strings {'parf car', 'premium ad car', 'low mileage car'}
    """
    return set(category.split(', '))


def vehicle_type_to_cat_num(type_of_vehicle: str) -> int:
    """
    Groups the different types of vehicles into a smaller number
    of categories to handle sparsity issues by assigning a number
    to each meso-group of vehicles. After this has been
    run, the column should be made categorical
    """
    if not type_of_vehicle or not isinstance(type_of_vehicle, str):
        type_of_vehicle = 'others'

    for cat_num, cat in enumerate(const.VEHICLE_CATEGORIES, start=1):
        if type_of_vehicle in cat:
            return cat_num

    return 0


def clean_preliminary(df_original: pd.DataFrame, is_test: bool = False) \
        -> pd.DataFrame:
    """
    Runs the preliminary cleaning on the df before we start finding
    similarities.
    """
    df = df_original.copy()
    utils.drop_bad_cols(df)

    # We don't want our train results to be influenced by rows where a lot of
    # the data is missing. However, if it is the test data, we cannot simply
    # remove the rows that are annoying
    if not is_test:
        df = utils.remove_nan_rows(df)

    for str_col in const.STR_COLS:
        df[str_col] = df[str_col].apply(nan_to_str_fallback)

    df['fuel_type'] = df['fuel_type'].map(const.FUEL_TYPE_MAP)
    df['transmission'] = df['transmission'].map(const.TRANSMISSION_MAP)
    df['type_of_vehicle'] = df['type_of_vehicle'].apply(vehicle_type_to_cat_num)
    df['category'] = df['category'].apply(category_to_set)

    if not is_test:
        df['price'] = df['price'].apply(round)

    return df
