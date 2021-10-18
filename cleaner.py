from datetime import datetime
import json
import re
from typing import Optional, Set

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

import utils
import constants as const

SPLIT_RE = re.compile(r'; |,|\*|\s|&|\|')


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


def string_to_set(category: str) -> Set[str]:
    """
    Converts a category like "parf car, premium ad car, low mileage car"
    to a set of strings {'parf car', 'premium ad car', 'low mileage car'}
    """
    return set([cat.strip().lower() for cat in category.split(',')])


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


def handle_fuel_type_using_other_cols(df_original: pd.DataFrame) -> pd.DataFrame:
    """
    Tries to clean fuel type based on the description and features, falling
    back to 'petrol' for NaN values if nothing else can be found
    """
    def helper(row: pd.Series) -> str:
        if isinstance(row.fuel_type, str) and row.fuel_type:
            return row.fuel_type

        for val in [row.get('description'), row.get('features')]:
            if not isinstance(val, str):
                continue

            found_diesel = any('diesel' in token.strip().lower()
                               for token in SPLIT_RE.split(val))
            if found_diesel:
                return 'diesel'

        # If no fuel type is found from description or features, return petrol
        return 'petrol'

    df = df_original.copy()
    df.fuel_type = df.apply(helper, axis=1)
    return df


def handle_fuel_type_using_scraped_data(df_original: pd.DataFrame) -> pd.DataFrame:
    """
    Using extracted fueltype.csv values from WebScraping to fill na values in
    the dataset
    """
    if 'fuel_type' not in df_original:
        return df_original

    with open(const.FUEL_TYPE_PATH, 'r') as f:
        listing_id_to_fuel_type = json.load(f)['fuel_type']

    df = df_original.copy()
    fuel_type = df.listing_id.apply(lambda x: listing_id_to_fuel_type[str(x)])
    df.fuel_type = df.fuel_type.fillna(fuel_type)
    return df


def handle_date_fields(df_original: pd.DataFrame) -> pd.DataFrame:
    """
    * Agglomerating a singular 'registered_date' field with all values populated.
        - Removing 1 row with registered date in the future
    * Removing 'lifespan' since it has low data frequency
        - ~1500 rows with 'lifespan' - 'registered_date' as 7304
        - 22 rows with 'lifespan' - 'registered_date' other than 7304 but greater (and unique)
        - Alternative approach: Set other vehicles lifespan as 7304 which is the median and most frequent entry
    * Adding a new column for 'car_age'
    """
    cols = ['reg_date', 'lifespan', 'original_reg_date']
    if not all(col in df_original for col in cols):
        return df_original

    df = df_original.copy()
    df.lifespan = pd.to_datetime(df.lifespan)
    df.reg_date = pd.to_datetime(df.reg_date)
    df.original_reg_date = pd.to_datetime(df.original_reg_date)

    # Fixing NaNs across original_reg_date, reg_date by adding a new column
    df['registered_date'] = df.reg_date.fillna(df.original_reg_date)
    df = df.drop(columns=['reg_date', 'original_reg_date'])
    df = df.drop(df[df.registered_date > datetime.now()].index)

    df = df.drop(columns=['lifespan'])
    # Alternative
    # df.lifespan = df.lifespan.fillna(df.registered_date + pd.Timedelta(days=7304))

    # Remember to remove a row with manufactured as 2925 (bad value)
    df['car_age'] = datetime.now().year - df.manufactured
    df = df.drop(df[(df.car_age > const.MAX_CAR_AGE) | (df.car_age < 0)].index)

    return df


def handle_opc(df_original: pd.DataFrame) -> pd.DataFrame:
    """
    Replacing NaN values with 0 denoting "Non-OPC" vehicles.
    Replacing values with 1 denoting "OPC" vehicles
    """
    if 'opc_scheme' not in df_original:
        return df_original

    df = df_original.copy()
    df.opc_scheme = df.opc_scheme.fillna('0')
    df.loc[~df.opc_scheme.isin(['0']), 'opc_scheme'] = '1'
    df.opc_scheme = df.opc_scheme.astype('uint8')
    return df


def handle_make(df_original: pd.DataFrame) -> pd.DataFrame:
    """
    upon checking it's found that ALL ROWS HAVE model
    but not all rows have make available which can be extracted from Title
    """
    if 'title' not in df_original or 'make' not in df_original:
        return df_original

    df = df_original.copy()
    splitted_titles = df.title.apply(str.lower).str.split()
    df.make = df.make.fillna(splitted_titles.str[0])
    return df


def clean_preliminary(df_original: pd.DataFrame, is_test: bool = False,
                      clean_fuel_type_with_scraped: bool = False) \
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

    for df_func in [handle_date_fields, handle_make]: #handle_opc, - Removed because "category" gives us opc_car as a one-hot encoded column with same data
        df = df_func(df)

    intersect_cols = set(df.columns) & set(const.STR_COLS)
    for str_col in intersect_cols:
        df[str_col] = df[str_col].apply(nan_to_str_fallback)

    if 'transmission' in df:
        df.transmission = df.transmission.map(const.TRANSMISSION_MAP)

    if 'type_of_vehicle' in df:
        df.type_of_vehicle = df.type_of_vehicle.apply(vehicle_type_to_cat_num)

    if 'category' in df:
        df.category = df.category.apply(string_to_set)

    if 'fuel_type' in df:
        fuel_func = (handle_fuel_type_using_scraped_data
                     if clean_fuel_type_with_scraped
                     else handle_fuel_type_using_other_cols)
        df = fuel_func(df)

    if 'price' in df:
        df.price = df.price.apply(round)

    return df.reset_index()


def to_categorical_for_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts all relevant columns to categorical / one-hot encoded
    """
    # Fuel type
    fuel_type_dummies = pd.get_dummies(df.fuel_type)

    # Categories
    mlb = MultiLabelBinarizer()
    binary_cats = mlb.fit_transform(df.category)
    cols = [col.replace(' ', '_') for col in mlb.classes_]
    binary_cats_df = pd.DataFrame(binary_cats, columns=cols)
    binary_cats_df.drop(['electric_cars', 'hybrid_cars'], axis=1, inplace=True)
    binary_cats_df.rename(columns={'-': 'missing_category'}, inplace=True)

    return pd.concat([
        df,
        fuel_type_dummies,
        binary_cats_df
    ], axis=1)
