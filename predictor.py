
import pandas as pd
import pickle
import os
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, f1_score

import constants as const


def train_fill_ml_na(df_original: pd.DataFrame, target_col: str, use_price=False) -> float:
    """[summary]

    Args:
        df_original (pd.DataFrame): Dataframe to be processed
        target_col (str): Column to train model for
        use_price (bool): use price as a feature in the model. Defaults to False

    Returns:
        float: Mean absolute error, as a percentage of mean column value
    """
    regressor = (target_col != 'no_of_owners')
    df = df_original.copy()
    predict_df = df[df[target_col].isna()].drop([target_col], axis=1)
    drop_predict_cols = [col for col in predict_df.columns
                         if sum(predict_df[col].isna()) > predict_df.shape[0]*0.2]
    with open('scraped_data/cols_to_drop/'+target_col+'_cols.txt', "wb") as fp:
        pickle.dump(drop_predict_cols, fp)
    target = df[~df[target_col].isna()].drop(drop_predict_cols, axis=1).dropna()
    if use_price == False:
        for temp_df in [predict_df, target]:
            if 'price' in temp_df.columns:
                temp_df.drop(['price'], axis=1, inplace=True)

    X = target.drop([target_col], axis=1)
    Y = target[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.15, random_state=42)

    rf = RandomForestRegressor()
    if regressor == False:
        rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=const.RF_REG_RAND_GRID,
                                   n_iter=const.NUM_NA_TRAIN_ITER, cv=const.K_CROSS_FOLD_NA_TRAIN, verbose=10,
                                   random_state=42, n_jobs=-1)
    rf_random.fit(X_train, y_train)

    pred = rf_random.predict(X_test)
    error = None
    if regressor == False:
        error = f1_score(y_test, pred, average='weighted')
    else:
        error = (mean_absolute_error(y_test, pred)/df[target_col].mean())*100

    best_rf = rf_random.best_estimator_
    best_rf.fit(X, Y)
    filename = 'Models/Fill_na_models/'+target_col+'.sav'
    pickle.dump(best_rf, open(filename, 'wb'))

    return error


def fill_ml_na_col(df_original: pd.DataFrame, target_col: str, use_price=False) -> pd.DataFrame:
    """[summary]

    Args:
        df_original (pd.DataFrame): Dataframe to be processed
        target_col (str): column to fill values for
        use_price (bool): use price as a feature in the model. Defaults to False

    Returns:
        pd.DataFrame: Processed dataframe with
    """
    filename = 'Models/Fill_na_models/' + target_col + '.sav'
    if not os.path.exists(filename):
        print(target_col + " has not been trained")
        return df_original

    df = df_original.copy()
    predict_df = df[df[target_col].isna()].drop([target_col], axis=1)

    cols_to_drop_ml_infer = None
    with open('scraped_data/cols_to_drop/'+target_col+'_cols.txt', "rb") as fp:
        cols_to_drop_ml_infer = pickle.load(fp)

    predict_df.drop(cols_to_drop_ml_infer, axis=1, inplace=True)
    predict_df.dropna(inplace=True)
    if use_price == False and 'price' in predict_df.columns:
        predict_df.drop(['price'], axis=1, inplace=True)

    model = pickle.load(open(filename, 'rb'))
    if predict_df.shape[0] == 0:
        return df
    values = model.predict(predict_df)

    df.loc[predict_df.index, target_col] = values

    return df


def fill_ml_na(df_orginal: pd.DataFrame, training=False, use_price=False) -> pd.DataFrame:
    """[summary]

    Args:
        df_orginal (pd.DataFrame): Dataframe to fill values for
        training (bool, optional): Set True to Train Model. Defaults to False.
        use_price (bool): use price as a feature in the model. Defaults to False

    Returns:
        pd.DataFrame: Filled DataFrame
    """
    df = df_orginal.copy()
    to_do = {col: sum(df[col].isna())
             for col in df.columns if sum(df[col].isna()) > 0}
    to_do = tuple(
        sorted(to_do.items(), key=lambda item: item[1], reverse=True))

    for target_col, _ in to_do:
        if training:
            error = train_fill_ml_na(df, target_col, use_price)
            print("Error for training "+target_col+" is: ", error)

        df = fill_ml_na_col(df, target_col, use_price)

    return df
