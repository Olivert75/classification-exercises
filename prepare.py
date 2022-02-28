import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from pydataset import data
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import warnings 
warnings.filterwarnings("ignore")

from acquire import get_titanic_data, get_iris_data, get_telco_data

#**********************************IRIS DATASET**********************************

def prep_iris(iris_df):
    #Drop the species_id and measurement_id columns
    new_iris_df = iris_df.drop(columns = ['species_id', 'measurement_id'])
    #Rename the species_name to species
    new_iris_df.rename(columns={'species_name':'species'},inplace=True)
    #Create a dummy data frame and then join dummy_df with new_iris_df
    dummy_df = pd.get_dummies(new_iris_df[['species']],dummy_na=False,drop_first=[True, True])
    new_iris_df = pd.concat([new_iris_df, dummy_df],axis=1)
    return new_iris_df

#**********************************TITANNIC DATASET**********************************

def prep_titanic(titanic_df):
    #Drop duplicates and unnecessary columns 
    new_titanic_df = titanic_df.drop(columns = ['passenger_id','embarked','pclass','deck','age'])
    #Fill empty value with southampton
    new_titanic_df['embark_town'] = new_titanic_df.embark_town.fillna(value='Southampton')
    #Create dummy dataframe and join it with new_titanic_df
    dummy_df = pd.get_dummies(new_titanic_df[['sex','embark_town']],dummy_na=False,drop_first=[True,True])
    new_titanic_df = pd.concat([new_titanic_df, dummy_df],axis=1)

#**********************************TELCO DATASET**********************************

def prep_telco(telco_df):
    new_telco_df = telco_df.drop(columns = ['customer_id','contract_type_id.1','payment_type_id.1','internet_service_type_id.1'])
    dummy_df = pd.get_dummies(new_telco_df[['churn','internet_service_type']],dummy_na=False,drop_first=[True,True])
    new_telco_df['total_charges'] = new_telco_df.total_charges.replace(' ', np.nan).astype(float)
    new_telco_df = pd.concat([new_telco_df, dummy_df],axis=1)
    return new_telco_df