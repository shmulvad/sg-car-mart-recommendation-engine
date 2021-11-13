import json
import pickle
import re
from datetime import datetime
from typing import List, Optional, Set

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

import constants as const
import utils

SPLIT_RE = re.compile(r"; |,|\*|\s|&|\|")


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
    return set([cat.strip().lower() for cat in category.split(",")])


def vehicle_type_to_cat_num(type_of_vehicle: str) -> int:
    """
    Groups the different types of vehicles into a smaller number
    of categories to handle sparsity issues by assigning a number
    to each meso-group of vehicles. After this has been
    run, the column should be made categorical
    """
    if not type_of_vehicle or not isinstance(type_of_vehicle, str):
        type_of_vehicle = "others"

    for cat_num, cat in enumerate(const.VEHICLE_CATEGORIES, start=1):
        if type_of_vehicle in cat:
            return cat_num

    return 0


def handle_fuel_type_using_other_cols(df_original: pd.DataFrame) -> pd.DataFrame:
    """
    Tries to clean fuel type based on the description and features
    """
    def helper(row: pd.Series) -> str:
        if isinstance(row.fuel_type, str) and row.fuel_type:
            return row.fuel_type

        for val in [row.get("description"), row.get("features")]:
            if not isinstance(val, str):
                continue

            found_petrol = any(
                "petrol" in token.strip().lower() for token in SPLIT_RE.split(val)
            )

            if found_petrol:
                return "petrol"

            found_diesel = any(
                "diesel" in token.strip().lower() for token in SPLIT_RE.split(val)
            )
            if found_diesel:
                return "diesel"

        # If no fuel type is found from description or features, return nan
        return np.nan

    df = df_original.copy()
    df.fuel_type = df.apply(helper, axis=1)
    return df


def handle_fuel_type_using_scraped_data(df_original: pd.DataFrame) -> pd.DataFrame:
    """
    Using extracted fueltype.csv values from WebScraping to fill na values in
    the dataset
    """
    if "fuel_type" not in df_original:
        return df_original

    with open(const.FUEL_TYPE_PATH, "r") as f:
        listing_id_to_fuel_type = json.load(f)

    df = df_original.copy()
    fuel_type = df.listing_id.apply(lambda x: listing_id_to_fuel_type[str(x)])
    df.fuel_type = df.fuel_type.fillna(fuel_type)
    return df


def handle_fuel_type(df_original: pd.DataFrame) -> pd.DataFrame:
    """
    Tries to clean fuel type based on the description and features, falling
    back to scraped data if nothing can be found in the other columns
    """
    df = handle_fuel_type_using_other_cols(df_original)
    df = handle_fuel_type_using_scraped_data(df)
    return df


def handle_date_fields(
    df_original: pd.DataFrame, is_test: bool = False
) -> pd.DataFrame:
    """
    * Agglomerating a singular 'registered_date' field with all values populated.
        - Removing 1 row with registered date in the future
    * Removing 'lifespan' since it has low data frequency
        - ~1500 rows with 'lifespan' - 'registered_date' as 7304
        - 22 rows with 'lifespan' - 'registered_date' other than 7304 but greater (and unique)
        - Alternative approach: Set other vehicles lifespan as 7304 which is the median and most frequent entry
    * Adding a new column for 'car_age'
    """
    cols = ["reg_date", "lifespan", "original_reg_date"]
    if not all(col in df_original for col in cols):
        return df_original

    df = df_original.copy()
    df.lifespan = pd.to_datetime(df.lifespan)
    df.reg_date = pd.to_datetime(df.reg_date)
    df.original_reg_date = pd.to_datetime(df.original_reg_date)

    # Fixing NaNs across original_reg_date, reg_date by adding a new column
    df["registered_date"] = df.reg_date.fillna(df.original_reg_date)
    df = df.drop(columns=["reg_date", "original_reg_date"])
    if not is_test:
        df = df.drop(df[df.registered_date > datetime.now()].index)

    df = df.drop(columns=["lifespan"])
    # Alternative
    # df.lifespan = df.lifespan.fillna(df.registered_date + pd.Timedelta(days=7304))

    # Remember to remove a row with manufactured as 2925 (bad value)
    na_filler = pd.Series(pd.DatetimeIndex(df.registered_date).year)
    df.manufactured = df.manufactured.fillna(na_filler)
    df["car_age"] = datetime.now().year - df.manufactured
    if not is_test:
        df = df.drop(df[(df.car_age > const.MAX_CAR_AGE) | (df.car_age < 0)].index)

    return df


def handle_opc(df_original: pd.DataFrame) -> pd.DataFrame:
    """
    Replacing NaN values with 0 denoting "Non-OPC" vehicles.
    Replacing values with 1 denoting "OPC" vehicles
    """
    if "opc_scheme" not in df_original:
        return df_original

    df = df_original.copy()
    df.opc_scheme = df.opc_scheme.fillna("0")
    df.loc[~df.opc_scheme.isin(["0"]), "opc_scheme"] = "1"
    df.opc_scheme = df.opc_scheme.astype("uint8")
    return df


def handle_make(df_original: pd.DataFrame) -> pd.DataFrame:
    """
    upon checking it's found that ALL ROWS HAVE model
    but not all rows have make available which can be extracted from Title
    """
    if "title" not in df_original or "make" not in df_original:
        return df_original

    df = df_original.copy()
    splitted_titles = df.title.apply(str.lower).str.split(" |-")
    df.make = splitted_titles.str[0]

    a_file = open(const.MAKE_DICT_PATH, "rb")
    make_dict = pickle.load(a_file)
    a_file.close()

    df.make = df.make.map(make_dict)
    df.make = df.make.astype("category")

    return df


def handle_make_model(df_original: pd.DataFrame, replace_by_bins=False) -> pd.DataFrame:
    """
    Combines the make and model of the car, and assigns it an ordinal value
    between 0-1000, according to the average price for the (make, model)

    There is also an option to replace by bin value between 0-27 if we need
    lesser unique values
    """
    if "title" not in df_original or "make" not in df_original:
        return df_original

    df = df_original.copy()
    splitted_titles = df.title.apply(str.lower).str.split(" |-")
    df.make = splitted_titles.str[0]
    df['make_model'] = df.apply(lambda x: x['make']+' '+x['model'], axis=1)

    path = const.MAKE_MODEL_BIN_PATH if replace_by_bins else const.MAKE_MODEL_DICT_MEAN_PATH
    with open(path, "rb") as f:
        make_model_dict = pickle.load(f)

    df.make_model = df.make_model.map(make_model_dict)

    return df

def new_clean(df,is_test=False):
    utils.drop_bad_cols(df)
    df = handle_make_model(df,replace_by_bins=True)
    df = handle_date_fields(df,is_test)
    intersect_cols = set(df.columns) & set(const.STR_COLS)
    for str_col in intersect_cols:
        df[str_col] = df[str_col].apply(nan_to_str_fallback)

    if "transmission" in df:
        df.transmission = df.transmission.map(const.TRANSMISSION_MAP)

    if "fuel_type" in df:
        df = handle_fuel_type(df)

    df.drop(columns=const.NOMINAL_TO_REMOVE, inplace=True)
    df.drop(['arf', 'road_tax', 'dereg_value', 'depreciation'], axis=1, inplace=True)

    for col in ['type_of_vehicle', 'category', 'transmission', 'fuel_type', 'make_model']:
        df[col] = df[col].astype("category")

    df.mileage = df.mileage.apply(np.log)
    df.engine_cap = df.engine_cap.apply(np.square)
    return df


def clean_preliminary(df_original: pd.DataFrame,
                      is_test: bool = False) -> pd.DataFrame:
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

    df = handle_make_model(df)
    df = handle_date_fields(df, is_test)

    intersect_cols = set(df.columns) & set(const.STR_COLS)
    for str_col in intersect_cols:
        df[str_col] = df[str_col].apply(nan_to_str_fallback)

    if "transmission" in df:
        df.transmission = df.transmission.map(const.TRANSMISSION_MAP)

    if "type_of_vehicle" in df:
        df.type_of_vehicle = df.type_of_vehicle.apply(vehicle_type_to_cat_num)
        df.type_of_vehicle = df.type_of_vehicle.astype("category")

    if "category" in df:
        df.category = df.category.apply(string_to_set)

    if "fuel_type" in df:
        df = handle_fuel_type(df)

    if "price" in df:
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
    cols = [col.replace(" ", "_") for col in mlb.classes_]
    binary_cats_df = pd.DataFrame(binary_cats, columns=cols)
    binary_cats_df.drop(["electric_cars", "hybrid_cars"], axis=1, inplace=True)
    binary_cats_df.rename(columns={"-": "missing_category"}, inplace=True)

    df.drop(columns=["fuel_type", "category"], inplace=True)
    return pd.concat([df, fuel_type_dummies, binary_cats_df], axis=1)


def remove_nominal_cols(df: pd.DataFrame,
                        cols_to_keep: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Removes nominal columns that don't impart any useful information
    except the columns provided by the user
    """
    cols_to_remove = const.NOMINAL_TO_REMOVE
    if cols_to_keep:
        cols_to_remove = list(set(cols_to_remove) - set(cols_to_keep))

    return df.drop(columns=cols_to_remove)

def clean_sim_filled_data(df,is_test=False):
    df.drop(['Unnamed: 0'],axis=1,inplace = True)
    utils.drop_bad_cols(df)
    df = handle_make_model(df,replace_by_bins=True)
    df = handle_date_fields(df,is_test)

    cols_to_keep: list = []
    cols_to_remove = [col for col in const.NOMINAL_TO_REMOVE if col not in cols_to_keep]

    df.drop(columns=cols_to_remove,inplace=True)
    
    for cols in ['type_of_vehicle','category','transmission','fuel_type','make_model']:
        df[cols] = df[cols].astype("category")

    return df
