import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MultiLabelBinarizer,OneHotEncoder

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split



def clean_categories(main_df):    
    df = main_df.copy()
    categories =set()
    for i in range(0,len(df)):
        tokens = df['category'][i].split(',')

        for token in tokens:
            categories.add(token.strip().lower())
    
    def helper(category):
        tokens = category.split(',')
        categories = set([token.strip().lower() for token in tokens])
        return categories
    
    df['new_cats'] = df.category.apply(lambda x:helper(x))
    
    
    mlb = MultiLabelBinarizer()
    test = pd.DataFrame(mlb.fit_transform(df['new_cats']),columns=categories)
    test.drop(['electric cars','hybrid cars'],axis=1,inplace = True)
    test.rename(columns={"-": "Category_Missing_Value"},inplace=True)
    test.Category_Missing_Value = test.Category_Missing_Value.apply(lambda x: float('nan') if x == 1 else x)
    
    df = pd.concat([df, test], axis=1)
    df.drop(['category'],axis = 1,inplace = True)
    
    return df



def clean_fuel_type(full_df):
    df = full_df.copy()
    for i in range(0,len(df)):
        if not pd.isna(df['fuel_type'][i]):
            continue

        tokens_desc = None
        tokens_feat = None

        if type(df['description'][i]) == str:
            tokens_desc = re.split('; |,|\*|\s|&|\|',df['description'][i])
            tokens_desc = [token.strip().lower() for token in tokens_desc]

            if 'diesel' in tokens_desc:
                df.loc[i, 'fuel_type'] = 'diesel'
                continue

        if type(df['features'][i]) == str:
            tokens_feat = re.split('; |,|\*|\s|&|\|',df['features'][i])
            tokens_feat = [token.strip().lower() for token in tokens_feat]
            
            if 'diesel' in tokens_feat:
                df.loc[i, 'fuel_type'] = 'diesel'
                continue

    df['fuel_type'].fillna('petrol',inplace= True)
    ohe = OneHotEncoder(sparse=False)
    test = pd.DataFrame(ohe.fit_transform(df['fuel_type'].to_numpy().reshape(-1,1)),columns = ohe.categories_[0])
    
    df = pd.concat([df, test], axis=1)
    df.drop(['fuel_type'],axis = 1,inplace = True)
    
    return df

def get_predictor(full_df,col):
    df = full_df.copy()
    retain_columns = [col for col in df.columns if sum(df[col].isna())<2784]
    df = df[retain_columns]
    df.dropna(inplace = True)
    
    X = df[col]
    Y = df.drop([col],axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0,train_size=0.2)
    reg = GradientBoostingRegressor(random_state=0) 
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    print("The Mean squared error for the predictor is: ",mean_squared_error(y_test, y_pred, squared=False))
    
#     predict_df = df[df[col].isna()][retain_columns].drop(['make'],axis=1).dropna()
#     fill_values = reg.predict(predict_df)    
    
    "To be completed according to nature of cleaned data"
    
    return reg



