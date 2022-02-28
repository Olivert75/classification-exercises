import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings 
warnings.filterwarnings("ignore")
import os
from pydataset import data
from scipy import stats
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

#**********************************TELCO DATASET**********************************