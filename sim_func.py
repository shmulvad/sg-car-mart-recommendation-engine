from difflib import SequenceMatcher
from numbers import Number
from typing import Any, Set

import numpy as np
import pandas as pd

import utils


def equality(elm1: str, elm2: str, *args) -> float:
    """
    Returns 1.0 if two elements are exactly equal to each other, othwerwise 0.
    Suitable for i.e. strings that represent categorical values
    """
    return float(elm1 == elm2)


def get_sim_ratio(str1: str, str2: str, *args) -> float:
    """Returns the similarity ratio between two strings in range 0..1"""
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()


def sim_above_threshold(str1: str, str2: str, *args, threshold=0.95) -> float:
    """Returns 1.0 if two strings are very close to each other, otherwise 0"""
    return float(get_sim_ratio(str1, str2) >= threshold)


def numerical_difference(num1: Number, num2: Number,
                         df: pd.DataFrame, col: str, *args) -> float:
    """
    Computes the similarity between two number, and returns 1
    if there is no difference and 0 if the numbers are 'very'
    different
    """
    diff = num1 - num2
    diff_squared = diff * diff
    max_diff_squared = utils.get_max_squared_diff(df, col)
    diff_normalized = diff_squared / max_diff_squared
    sim = 1.0 - diff_normalized
    return np.clip(sim, 0.0, 1.0)


def jaccard_similarity(set1: Set[Any], set2: Set[Any], *args) -> float:
    """Returns the Jaccard similarity between two sets"""
    return len(set1.intersection(set2)) / len(set1.union(set2))


SIM_FUNCS = {
    'title': get_sim_ratio,
    'make': sim_above_threshold,
    'model': sim_above_threshold,
    'manufactured': numerical_difference,
    'type_of_vehicle': numerical_difference,
    'category': jaccard_similarity,
    'transmission': equality,
    'curb_weight': numerical_difference,
    'power': numerical_difference,
    'fuel_type': equality,
    'engine_cap': numerical_difference,
    'no_of_owners': numerical_difference,
    'depreciation': numerical_difference,
    'coe': numerical_difference,
    'road_tax': numerical_difference,
    'dereg_value': numerical_difference,
    'mileage': numerical_difference,
    'omv': numerical_difference,
    'arf': numerical_difference,
    'price': numerical_difference
}
