import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split

import constants as const


def train_fill_ml_na(df_original: pd.DataFrame, num_iter, k_splits,
                    target_col: str, use_price=False) -> float:
    """The function is called to train the ML-model to predict the values for the 
    target column passed as an arguement

    Args:
        df_original (pd.DataFrame): Dataframe to be processed
        num_iter (int): Number of iterations for RandomGridSearchCV 
        k_splits (int): Number of splits for the k-fold cross validation
        target_col (str): Column to train model for
        use_price (bool, optional): use price as a feature in the model. Defaults to False

    Returns:
        float: Returns the test error for the model as a percentage of the mean value of the 
                target column for regression tasks and F1 score otehrwise. 

    """
    regressor = (target_col != 'no_of_owners')
    df = df_original.copy()
    predict_df = df[df[target_col].isna()].drop([target_col], axis=1)
    drop_predict_cols = [col for col in predict_df.columns
                         if sum(predict_df[col].isna()) > predict_df.shape[0]*0.2]
    
    with open('data/ml_cols_to_drop/'+target_col+'_cols.pkl', "wb") as fp:
        pickle.dump(drop_predict_cols, fp)

    target = df[~df[target_col].isna()].drop(drop_predict_cols, axis=1).dropna()
    
    if use_price == False:
        for temp_df in [predict_df, target]:
            if 'price' in temp_df.columns:
                temp_df.drop(['price'], axis=1, inplace=True)

    X = target.drop([target_col], axis=1)
    Y = target[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

    rf = RandomForestRegressor if regressor else RandomForestClassifier
    rf_random = RandomizedSearchCV(
        estimator=rf(),
        param_distributions=const.RF_REG_RAND_GRID,
        n_iter=num_iter,
        cv=k_splits,
        verbose=10,
        random_state=42,
        n_jobs=-1
    )
    rf_random.fit(X_train, y_train)

    pred = rf_random.predict(X_test)
    error = None
    if regressor == False:
        error = f1_score(y_test, pred, average='weighted')
    else:
        error = (mean_absolute_error(y_test, pred)/df[target_col].mean())*100

    best_rf = rf_random.best_estimator_
    best_rf.fit(X, Y)
    filename = 'data/Fill_na_models/'+target_col+'.sav'
    pickle.dump(best_rf, open(filename, 'wb'))

    return error


def fill_ml_na_col(df_original: pd.DataFrame, target_col: str, use_price=False) -> pd.DataFrame:
    """Once the ML models are trained, this function is used to infer the trained models and fill 
    the missing values

    Args:
        df_original (pd.DataFrame): Dataframe to be processed
        target_col (str): column to fill values for
        use_price (bool): use price as a feature in the model. Defaults to False

    Returns:
        pd.DataFrame: Processed dataframe with filled values
    """
    filename = 'data/Fill_na_models/' + target_col + '.sav'
    if not os.path.exists(filename):
        # print(target_col + " has not been trained")
        return df_original

    df = df_original.copy()
    predict_df = df[df[target_col].isna()].drop([target_col], axis=1)

    cols_to_drop_ml_infer = None
    with open('data/ml_cols_to_drop/'+target_col+'_cols.pkl', "rb") as fp:
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


def fill_ml_na(df_orginal: pd.DataFrame, training=False, use_price=False,
                num_iter = const.NUM_NA_TRAIN_ITER, k_splits = const.K_CROSS_FOLD_NA_TRAIN) -> pd.DataFrame:
    """This is the main function that is called to utilise the helper functions declared above. It 
    is used for both training and inference as per the given arguements.

    Args:
        df_orginal (pd.DataFrame): Dataframe to fill values for
        training (bool, optional): Set True to Train Model. Defaults to False.
        use_price (bool, optional): use price as a feature in the model. Defaults to False
        num_iter ([type], optional): Only needed when training, specifies number of iterations 
                                    for RandomGridSearchCV. Defaults to const.NUM_NA_TRAIN_ITER.
        k_splits ([type], optional):  Only needed when training, specifies number of splits 
                                    for the k-fold cross validation.Defaults to const.K_CROSS_FOLD_NA_TRAIN.

    Returns:
        pd.DataFrame: Dataframe with filled values
    """

    df = df_orginal.copy()
    to_do = {col: sum(df[col].isna())
             for col in df.columns if sum(df[col].isna()) > 0}
    to_do = tuple(
        sorted(to_do.items(), key=lambda item: item[1], reverse=True))

    for target_col, _ in to_do:
        if training:
            error = train_fill_ml_na(df, num_iter, k_splits, target_col, use_price)
            print("Error for training "+target_col+" is: ", error)

        df = fill_ml_na_col(df, target_col, use_price)

    return df
