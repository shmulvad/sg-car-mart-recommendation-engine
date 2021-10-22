
import pandas as pd
import numpy as np
import math
import pickle
import os
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,f1_score
from utils import rmse, vprint
import constants as const


def train_fill_ml_na(df_original:pd.DataFrame,target_col:str)-> float:
    """[summary]

    Args:
        df_original (pd.DataFrame): Dataframe to be processed
        target_col (str): Column to train model for

    Returns:
        float: Mean absolute error, as a percentage of mean column value
    """

    regressor = (target_col != 'no_of_owners')
    df = df_original.copy()
    predict_df = df[df[target_col].isna()].drop([target_col],axis=1)
    drop_predict_cols = [col for col in predict_df.columns if sum(predict_df[col].isna())>predict_df.shape[0]*0.2]
    target = df[~df[target_col].isna()].drop(drop_predict_cols,axis=1).dropna()

    X = target.drop([target_col],axis=1)
    Y = target[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

    rf = RandomForestRegressor()
    if regressor == False:
        rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = const.RF_REG_RAND_GRID,
                                    n_iter = const.NUM_NA_TRAIN_ITER, cv = 3, verbose=10, 
                                    random_state=42, n_jobs = -1)
    rf_random.fit(X_train, y_train)

    pred = rf_random.predict(X_test)
    error = None
    if regressor == False:
        error = f1_score(y_test,pred,average='weighted')
    else:  
        error = (mean_absolute_error(y_test,pred)/df[target_col].mean())*100

    best_rf = rf_random.best_estimator_
    best_rf.fit(X, Y)
    filename = 'Fill_na_models/'+target_col+'.sav'
    pickle.dump(best_rf, open(filename, 'wb'))

    return error


def fill_ml_na_col(df_original:pd.DataFrame,target_col:str)->pd.DataFrame:
    """[summary]

    Args:
        df_original (pd.DataFrame): Dataframe to be processed
        target_col (str): column to fill values for
        infer (bool, optional): Defaults to False, trains models in this setting

    Returns:
        pd.DataFrame: Processed dataframe with
    """
    filename = 'Fill_na_models/'+target_col+'.sav'
    if not os.path.exists(filename):
            print("Model has not been trained")
            return None

    df = df_original.copy()    
    predict_df = df[df[target_col].isna()].drop([target_col],axis=1)
    drop_predict_cols = [col for col in predict_df.columns if sum(predict_df[col].isna())>predict_df.shape[0]*0.2]
    predict_df.drop(drop_predict_cols,axis=1,inplace=True)
    predict_df.dropna(inplace=True)

    model = pickle.load(open(filename, 'rb'))
    values = model.predict(predict_df)

    df.loc[predict_df.index,target_col] = values

    return df

def fill_ml_na(df_orginal:pd.DataFrame,training = False)->pd.DataFrame:
    """[summary]

    Args:
        df_orginal (pd.DataFrame): Dataframe to fill values for
        training (bool, optional): Set True to Train Model. Defaults to False.

    Returns:
        pd.DataFrame: Filled DataFrame
    """

    df = df_orginal.copy()
    columns_todo = [col for col in df.columns if sum(df[col].isna())>0]
    to_do = {col:sum(df[col].isna()) for col in df.columns if sum(df[col].isna())>0}
    to_do = tuple(sorted(to_do.items(), key=lambda item: item[1],reverse=True))

    for target_col,num_na in to_do:
        if training:
            error = train_fill_ml_na(df,target_col)
            print("Error for training "+target_col+" is: ", error)
        df = fill_ml_na_col(df,target_col)

    return df


# def get_predictor(full_df, col, verbose=True):
#     df = full_df.copy()
#     retain_columns = [col for col in df.columns if sum(df[col].isna()) < 2784]
#     df = df[retain_columns]
#     df.dropna(inplace=True)

#     X = df[col]
#     Y = df.drop([col], axis=1)

#     X_train, X_test, y_train, y_test \
#         = train_test_split(X, Y, random_state=0, train_size=0.2)
#     reg = GradientBoostingRegressor(random_state=0)
#     reg.fit(X_train, y_train)

#     y_pred = reg.predict(X_test)
#     vprint(verbose, "The Mean squared error for the predictor is: ", rmse(y_test, y_pred))

#     # TODO: To be completed according to nature of cleaned data
#     return reg
