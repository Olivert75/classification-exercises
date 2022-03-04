import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydataset import data

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import warnings 
warnings.filterwarnings("ignore")

from acquire import get_titanic_data, get_iris_data, get_telco_data

pd.set_option('display.max_columns',None) #pd.set_option('display.max_rows',None)

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

def iris_split_data(new_iris_df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames; stratify on species.
    return train, validate, test DataFrames.
    '''
    train_validate, test = train_test_split(new_iris_df, test_size=.2, random_state=123, stratify=new_iris_df.species)
    
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123, 
                                       stratify=train_validate.species)
    
    return train, validate, test

#**********************************TITANNIC DATASET**********************************

def prep_titanic(titanic_df):
    
    #Drop duplicates and unnecessary columns 
    new_titanic_df = titanic_df.drop(columns = ['passenger_id','embarked','pclass','deck','age'])
    
    #Fill empty value with southampton
    new_titanic_df['embark_town'] = new_titanic_df.embark_town.fillna(value='Southampton')
    
    #Create dummy dataframe and join it with new_titanic_df
    dummy_df = pd.get_dummies(new_titanic_df[['sex','embark_town']],dummy_na=False,drop_first=[True,True])
    new_titanic_df = pd.concat([new_titanic_df, dummy_df],axis=1)

    return new_titanic_df

def split_data(new_titanic_df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames; stratify on survived.
    return train, validate, test DataFrames.
    '''
    train_validate, test = train_test_split(new_titanic_df, test_size=.2, random_state=123, stratify=new_titanic_df.survived)
    
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123, 
                                       stratify=train_validate.survived)
    return train, validate, test

#**********************************TELCO DATASET**********************************

def prep_telco(telco_df):
    
    #Drop unnecassary columns
    new_telco_df = telco_df.drop(columns = ['customer_id','contract_type_id.1','payment_type_id.1','internet_service_type_id.1'])
    
    #Change the total_charges from object type to float type
    new_telco_df['total_charges'] = new_telco_df.total_charges.replace(' ', np.nan).astype(float)
    
    #Encode the catergoical columns
    obj_col = new_telco_df.select_dtypes("object").columns.to_list()
    
    for col in obj_col:
        dummy_df = pd.get_dummies(new_telco_df[[col]],dummy_na=False,drop_first=[True,True])
        new_telco_df = pd.concat([new_telco_df, dummy_df],axis=1)
    
    new_telco_df.drop(columns=obj_col,inplace =True)
    
    return new_telco_df

def split_data(new_telco_df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames; stratify on churned.
    return train, validate, test DataFrames.
    '''
    train_validate, test = train_test_split(new_telco_df, test_size=.2, random_state=123, stratify=new_telco_df.churn)
    
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123, 
                                       stratify=train_validate.churn)
    return train, validate, test