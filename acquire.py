import numpy as np
import pandas as pd
from pydataset import data
import os
from env import user_name, password, host

def get_connection(db, username=user_name, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user_name}:{password}@{host}/{db}'

##### Acquire Titanic Data #####

def new_titanic_data():
    '''
    This function reads in the titanic data from the Codeup db
    and returns a pandas DataFrame with all columns.
    '''
    sql_query = 'SELECT * FROM passengers'
    df =  pd.read_sql(sql_query, get_connection('titanic_db'))
    return df 


def get_titanic_data():
    if os.path.isfile('titanic_df.csv'):
        df = pd.read_csv('titanic_df.csv', index_col = 0)
    else:
        df = new_titanic_data()
        df.to_csv('titanic_df.csv')
    return df

##### Acquire Iris Data #####

def new_iris_data():
    sql_query = 'SELECT * FROM species JOIN measurements USING(species_id)'
    df = pd.read_sql(sql_query,get_connection('iris_db'))
    return df

def get_iris_data():
    if os.path.isfile('iris_df.csv'):
        df = pd.read_csv('iris_df.csv', index_col=0)
    else:
        df = new_iris_data()
        df.to_csv('iris_df.csv')
    return df



##### Acquire Telco Data #####

def new_telco_data():
    sql_query = '''
    Select * from customers
    join contract_types on contract_types.contract_type_id = customers.contract_type_id
    join payment_types on payment_types.payment_type_id = customers.payment_type_id
    join internet_service_types on internet_service_types.internet_service_type_id = customers.internet_service_type_id
    '''
    df = pd.read_sql(sql_query,get_connection('telco_churn'))
    return df

def get_telco_data():
    if os.path.isfile('telco_churn.csv'):
        df = pd.read_csv('telco_churn.csv', index_col=0)
    else:
        df = new_telco_data()
        df.to_csv('telco_churn.csv')
    return df
