from typing import Optional

import pandas as pd
import numpy as np
from tqdm import tqdm

import utils
from constants import TO_SKIP, NAN_PENALTY
from sim_func import SIM_FUNCS, numerical_difference


def warm_cache(train_df: pd.DataFrame) -> None:
    """
    Warms up the cache by running the `numerical_difference` function
    for all numerical columns, thus computing the squared diff between
    5th and 95th quantiale that is used for comparison
    """
    for col, func in SIM_FUNCS.items():
        if func is numerical_difference:
            numerical_difference(0.1, 0.1, train_df, col)


def compute_row_similarity(row1: pd.Series, row2: pd.Series,
                           train_df: pd.DataFrame = None) -> float:
    """
    Computes the similarity between two rows and returns a float
    0..1 denoting how similar the rows are. The closer to 1.0, the more
    similar the rows are deemed
    """
    similarities_per_column = []
    num_nans = 0

    for col in row1.index:
        if col in TO_SKIP:
            continue

        value1, value2 = row1[col], row2[col]
        if utils.isnan(value1) or utils.isnan(value2):
            num_nans += 1
            continue

        sim_func = SIM_FUNCS.get(col, None)
        if sim_func is None:
            continue

        similarity = sim_func(value1, value2, train_df, col)
        similarities_per_column.append(similarity)

    if not similarities_per_column:
        return 0.0

    return np.mean(similarities_per_column) - num_nans * NAN_PENALTY


def compute_similarities(main_df: pd.DataFrame, train_df: pd.DataFrame,
                         verbose: bool = False) -> pd.DataFrame:
    """
    Computes the similarity between the rows in the dataframe that has missing
    values in critical columns compared to all other rows in DataFrame.
    """
    def compute_similarities_for_given_row(row: pd.Series) -> pd.Series:
        """
        Takes a single row as input and returns a series with the similarity
        between the row and all other rows in the dataframe
        """
        def compute_similarities_for_specific_row(row_to_compare: pd.Series) -> float:
            return compute_row_similarity(row, row_to_compare, train_df)

        return train_df.apply(compute_similarities_for_specific_row, axis=1)

    warm_cache(train_df)
    criticals = main_df[main_df.apply(utils.has_nan_in_critial_col, axis=1)]
    sim_df = criticals.parallel_apply(compute_similarities_for_given_row, axis=1)
    return sim_df


def replace_nan_with_most_similar(main_df: pd.DataFrame,
                                  train_df: Optional[pd.DataFrame] = None,
                                  sim_df: Optional[pd.DataFrame] = None,
                                  verbose: bool = False) -> pd.DataFrame:
    train_df = main_df.copy() if train_df is None else train_df
    if sim_df is None:
        utils.vprint(verbose, 'Computing similarities...')
        sim_df = compute_similarities(main_df, train_df, verbose)

    utils.vprint(verbose, 'Getting top k most similar...')
    top_k = utils.get_top_k_most_similar(sim_df)
    weights = utils.create_weights_df(sim_df, top_k, verbose=verbose)

    def replace_nan_with_most_similar_helper(row_original):
        row = row_original.copy()
        weights_i, top_k_i = weights.loc[row.name], top_k.loc[row.name]
        similar_row_indices = np.array(top_k_i)
        similar_rows = train_df.iloc[similar_row_indices]
        for col in row.index:
            is_numerical = SIM_FUNCS.get(col, None) is numerical_difference
            if not utils.isnan(row[col]) or not is_numerical:
                continue

            similar_values = np.array(similar_rows[col])
            is_valid = ~np.isnan(similar_values)
            is_valid_weight = is_valid * weights_i
            is_valid_weight /= np.sum(is_valid_weight)
            res = np.sum(is_valid_weight * similar_values)
            if utils.isnan(res):
                res = utils.get_median(train_df, col)

            row[col] = res

        return row

    df_out = main_df.copy()
    for i in tqdm(sim_df.index, disable=not verbose, desc='Replacing NaN rows'):
        df_out.loc[i] = replace_nan_with_most_similar_helper(df_out.loc[i])

    return df_out
