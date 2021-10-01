#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime


# In[2]:


#original_reg_date, reg_date, lifespan, fuel_type, opc_scheme


# In[3]:


def handle_date_fields(dataF):
    df = dataF.copy()
    
    '''
    * Agglomerating a singular 'registered_date' field with all values populated.
        - Removing 1 row with registered date in the future
    * Removing 'lifespan' since it has low data frequency
        - ~1500 rows with 'lifespan' - 'registered_date' as 7304
        - 22 rows with 'lifespan' other than 7304 but greater (and unique)
        - Alternative approach: Set other vehicles lifespan as 7304 which is the median and most frequent entry
    * Adding a new column for 'car_age'
    '''
    
    df.lifespan = pd.to_datetime(df.lifespan)
    df.reg_date = pd.to_datetime(df.reg_date)
    df.original_reg_date = pd.to_datetime(df.original_reg_date)

    # Fixing NaNs across original_reg_date, reg_date by adding a new column
    df["registered_date"] = df.reg_date.fillna(df.original_reg_date)
    df = df.drop(columns=['reg_date', 'original_reg_date'])
    df = df[~(df.registered_date > datetime.now())]
    
    df = df.drop(columns=['lifespan'])
    # Alternative 
#     df.lifespan = df.lifespan.fillna(df.registered_date + pd.Timedelta(days=7304))
    
    df["car_age"] = datetime.now().year - df.manufactured
    
    return df


# In[4]:


def handle_opc(dataF):
    df = dataF.copy()
    
    '''
    Replacing NaN values with 0 denoting "Non-OPC" vehicles.
    Replacing values with 1 denoting "OPC" vehicles
    '''
    df.opc_scheme = df.opc_scheme.fillna("0")
    df.loc[~df.opc_scheme.isin(["0"]), "opc_scheme"] = "1"
    
    return df


# In[5]:


df = pd.read_csv("train.csv")
df = handle_date_fields(df)
df = handle_opc(df)

