from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from tqdm import tqdm

import constants as const

cache = {}


def vprint(verbose: bool, *args, **kwargs) -> None:
    """Prints if verbose=True, otherwise does nothing"""
    if verbose:
        print(*args, **kwargs)


def isnan(value: Any) -> bool:
    """Returns True if value is NaN, otherwise False"""
    return value != value


def remove_nan_rows(df: pd.DataFrame, threshold=const.MAX_NUM_NANS) -> pd.DataFrame:
    """
    Removes all rows from dataset where the number of
    NaN columns is above threshold
    """
    number_nans = df.apply(lambda row: sum(map(isnan, row)), axis=1)
    return df[number_nans <= threshold]


def drop_bad_cols(df: pd.DataFrame) -> None:
    """Drops all COLS_TO_DROP from df inplace"""
    df.drop(const.COLS_TO_DROP, axis=1, inplace=True, errors='ignore')


def get_max_squared_diff(train_df: pd.DataFrame, col: str) -> float:
    """
    Returns the squared diff of the 5th quantile and 95th quantile
    """
    if col in cache:
        return cache[col]

    val1, val2 = train_df[col].quantile([0.05, 0.95])
    diff = val1 - val2
    diff_squared = diff * diff
    cache[col] = diff_squared
    return diff_squared


def get_median(train_df: pd.DataFrame, col: str) -> float:
    """Returns the median of the column in df"""
    key = f'{col}-median'
    if key in cache:
        return cache[key]

    val = train_df[col].median()
    cache[key] = val
    return val


def has_nan_in_critial_col(row: pd.Series) -> bool:
    """
    Checks whether a row has at least one critical
    column with a NaN
    """
    return any(isnan(row[col]) for col in const.CRITICAL_COLS if col in row)


def get_top_k_most_similar(sim_df, k=3000):
    """
    Gets the indices of the top k most similar rows based on a
    similarity df with the scores. If sim_df has shape (N, M)
    then the output dataframe will have shape (N, k) where k << M
    """
    indices = np.argsort(-sim_df, axis=1)
    top_k = indices.iloc[:, :k]
    return top_k


def create_weights_df(sim_df: pd.DataFrame, top_k: pd.DataFrame,
                      verbose=False) -> pd.DataFrame:
    """
    Creates a dataframe with similar shape as top_k where each
    row sums to 1 and the value in each entry denotes how much
    weights the corresponding index in top_k should be given
    """
    weights = []
    for i in tqdm(range(len(top_k)), disable=not verbose,
                  desc='Computing weights'):
        ws = np.array(sim_df.iloc[i, np.array(top_k.iloc[i])])
        ws /= np.sum(ws)
        weights.append(ws)
    return pd.DataFrame(weights, index=top_k.index)


def rmse(*args, **kwargs) -> float:
    """Returns the mean squared error"""
    return mse(*args, **kwargs, squared=False)


def preds_to_csv(preds: np.ndarray, out: Path = const.CSV_PREDS_OUT) -> None:
    """
    Takes a 1D numpy array of predictions and optionally an out path and writes
    them to a csv file. Assumes that the predictions are given in the same
    order as the test set.
    """
    df = pd.DataFrame(preds, columns=['Predicted'])
    df.to_csv(out, index_label='Id')


def get_make_model_dict(df_original: pd.DataFrame, replace_by_mean=False) -> dict:
    """
    Function used to generate dictionaries used to map (make,model) to nominal value

    Args:
        df_original (pd.DataFrame,replace_by_mean, optional): [description]. Defaults to False)

    Returns:
        Dictionary
    """
    df = df_original.copy()
    splitted_titles = df.title.apply(str.lower).str.split(" |-")
    df.make = splitted_titles.str[0]
    df['make_model'] = df.apply(lambda x: x['make'] + ' ' + x['model'], axis=1)
    test = df.groupby('make_model').mean().reset_index()
    test = test.sort_values('price')[['make_model', 'price']]

    if not replace_by_mean:
        new_make_model_dict = dict()
        prev = 0
        for i, untill in enumerate(const.MAKE_MODEL_BINS):
            subset = test[(test.price >= prev) & (test.price < untill)]
            prev = untill

            for key in subset.make_model.unique():
                new_make_model_dict[key] = i

        return new_make_model_dict

    test_min = const.MAKE_MODEL_PRICE_MIN
    test_max = const.MAKE_MODEL_PRICE_MAX
    test['price'] = test['price'].apply(lambda x: int(((x - test_min) / test_max)*1000))
    make_dict_mean_norm = pd.Series(test.price.values, index=test.make_model).to_dict()
    return make_dict_mean_norm
