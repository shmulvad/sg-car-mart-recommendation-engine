#!/usr/bin/env python

import pandas as pd
import numpy as np
from datetime import datetime
import httpx
from bs4 import BeautifulSoup
import re
import time
import json


def handle_date_fields(dataF):
    df = dataF.copy()

    """
    * Agglomerating a singular 'registered_date' field with all values populated.
        - Removing 1 row with registered date in the future
    * Removing 'lifespan' since it has low data frequency
        - ~1500 rows with 'lifespan' - 'registered_date' as 7304
        - 22 rows with 'lifespan' - 'registered_date' other than 7304 but greater (and unique)
        - Alternative approach: Set other vehicles lifespan as 7304 which is the median and most frequent entry
    * Adding a new column for 'car_age'
    """

    df.lifespan = pd.to_datetime(df.lifespan)
    df.reg_date = pd.to_datetime(df.reg_date)
    df.original_reg_date = pd.to_datetime(df.original_reg_date)

    # Fixing NaNs across original_reg_date, reg_date by adding a new column
    df["registered_date"] = df.reg_date.fillna(df.original_reg_date)
    df = df.drop(columns=["reg_date", "original_reg_date"])
    df = df.drop(df[df.registered_date > datetime.now()].index)

    df = df.drop(columns=["lifespan"])
    # Alternative
    #     df.lifespan = df.lifespan.fillna(df.registered_date + pd.Timedelta(days=7304))

    # Remember to remove a row with manufactured as 2925 (bad value)
    df["car_age"] = datetime.now().year - df.manufactured
    df = df.drop(df[(df.car_age > 50) | (df.car_age < 0)].index)

    return df


def handle_opc(dataF):
    df = dataF.copy()

    """
    Replacing NaN values with 0 denoting "Non-OPC" vehicles.
    Replacing values with 1 denoting "OPC" vehicles
    """
    df.opc_scheme = df.opc_scheme.fillna("0")
    df.loc[~df.opc_scheme.isin(["0"]), "opc_scheme"] = "1"

    return df


def handle_make(dataF):
    df = dataF.copy()

    """
    Upon checking it's found that ALL ROWS HAVE model
    But not all rows have make available which can be extracted from Title
    """

    #     title = df[df.make.isna()].title
    #     revlen = [i[0] for i in sorted(title.str.split(" "), key = lambda x: len(x), reverse=True)]
    df.make = df.make.fillna((df.title.apply(str.lower).str.split(" ")).str[0])

    return df


def handle_fuel_type(dataF):
    df = dataF.copy()

    """
    Using extracted fueltype.csv values from WebScraping to fill na values in the dataset
    """

    with open("scraped_data/listing_id_TO_fuel_type.json", "r") as f:
        listing_id_to_fuel_type = json.load(f)["fuel_type"]

    fuel_type = df.listing_id.apply(lambda x: listing_id_to_fuel_type[str(x)])
    df.fuel_type = df.fuel_type.fillna(fuel_type)
    return df


df = pd.read_csv("data/train.csv")
df = handle_date_fields(df)
df = handle_opc(df)
df = handle_make(df)
df = handle_fuel_type(df)
