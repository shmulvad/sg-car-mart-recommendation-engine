from typing import Any
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error as mse

from constants import MAX_NUM_NANS, COLS_TO_DROP, CRITICAL_COLS, \
                      CSV_PREDS_OUT

cache = {}


def vprint(verbose: bool, *args, **kwargs) -> None:
    """Prints if verbose=True, otherwise does nothing"""
    if verbose:
        print(*args, **kwargs)


def isnan(value: Any) -> bool:
    """Returns True if value is NaN, otherwise False"""
    return value != value


def remove_nan_rows(df: pd.DataFrame, threshold=MAX_NUM_NANS) -> pd.DataFrame:
    """
    Removes all rows from dataset where the number of
    NaN columns is above threshold
    """
    number_nans = df.apply(lambda row: sum(map(isnan, row)), axis=1)
    return df[number_nans <= threshold]  # .reset_index(drop=True)


def drop_bad_cols(df: pd.DataFrame) -> None:
    """Drops all COLS_TO_DROP from df inplace"""
    df.drop(COLS_TO_DROP, axis=1, inplace=True, errors='ignore')


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
    return any(isnan(row[col]) for col in CRITICAL_COLS)


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


def preds_to_csv(preds: np.ndarray, out: Path = CSV_PREDS_OUT) -> None:
    """
    Takes a 1D numpy array of predictions and optionally an out path and writes
    them to a csv file. Assumes that the predictions are given in the same
    order as the test set.
    """
    df = pd.DataFrame(preds, columns=['Predicted'])
    df.to_csv(out, index_label='Id')
