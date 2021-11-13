import re
from typing import Any, Callable, Dict

import httpx
import pandas as pd
from bs4 import BeautifulSoup

import constants as const

# Constants for the scraper
BASE_URL = 'https://www.sgcarmart.com'
USED_CARS_URL = f'{BASE_URL}/used_cars/info.php'
NEW_CARS_URL = f'{BASE_URL}/new_cars/newcars_specs.php'
MISSING = 'MISSING'

# Types
ListingId = int
CarCode, SubCode, FuelType = str, str, str
ScraperFunc = Callable[[Any], str]

# Global dictionaries
sub_code_dict: Dict[CarCode, SubCode] = {MISSING: MISSING}
fuel_type_dict: Dict[CarCode, FuelType] = {MISSING: MISSING}


def try_except_wrapper(scraper_func: ScraperFunc) -> ScraperFunc:
    """
    Run the given scraper function, but if it fails, return `MISSING`.
    """
    def wrapper(*args, **kwargs):
        try:
            return scraper_func(*args, **kwargs)
        except Exception as e:
            print(f'{scraper_func.__name__} failed with error: {e}')
            return MISSING
    return wrapper


@try_except_wrapper
def get_car_code_from_listing(listing_id: ListingId) -> CarCode:
    """Tries to scrape the car code based on the listing id"""
    r = httpx.get(f'{USED_CARS_URL}?ID={listing_id}')
    content = BeautifulSoup(r.content, 'html.parser')
    parent = content.find(class_='twoRow_info')
    link = str(parent.parent.find('a'))
    car_code = re.search(r"CarCode=(.+?)\"", link).group(1).replace("'", "")
    return car_code


@try_except_wrapper
def get_subcode_from_car_code(car_code: CarCode) -> SubCode:
    """Tries to scrape the subcode based on the car code"""
    global sub_code_dict
    if car_code in sub_code_dict:
        return sub_code_dict[car_code]

    r = httpx.get(f'{NEW_CARS_URL}?CarCode={car_code}')
    content = BeautifulSoup(r.content, 'html.parser')
    list_element = content.find(id="submodels_ul_link").find_all('a')
    sub_code_dict[car_code] = list_element
    return list_element


@try_except_wrapper
def get_fuel_type_only_on_car_code(car_code: CarCode) -> SubCode:
    """Tries to scrape the fuel type based on the car code"""
    global fuel_type_dict
    if car_code in fuel_type_dict:
        return fuel_type_dict[car_code]

    query = f'{NEW_CARS_URL}?CarCode={car_code}'
    r = httpx.get(query)
    content = BeautifulSoup(r.content, 'html.parser')
    fuel_type = content.find('td', text='Fuel type').nextSibling.nextSibling.text
    fuel_type_dict[car_code] = fuel_type
    return fuel_type


def get_all_car_codes_sub_car_codes(df_original: pd.DataFrame) -> pd.DataFrame:
    """
    Tries to scrape the car code, sub code and fuel type for all car listings
    in the given dataframe
    """
    df = df_original.copy()
    df['CarCode'] = df.listing_id.apply(get_car_code_from_listing)
    df['SubCode'] = df['CarCode'].apply(get_subcode_from_car_code)
    df['FuelTypeScraped'] = df['CarCode'].apply(get_fuel_type_only_on_car_code)
    return df


def main():
    """Entrypoint for the scraper"""
    df_train = pd.read_csv(const.TRAIN_PATH)
    df_test = pd.read_csv(const.TEST_PATH)
    df = pd.concat([df_train, df_test], axis=0)
    df_scraped = get_all_car_codes_sub_car_codes(df)
    df_scraped.to_csv(const.SCRAPED_PATH)


if __name__ == '__main__':
    main()
