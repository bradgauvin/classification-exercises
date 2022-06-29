#!/usr/bin/env python
# coding: utf-8

# Make a new python module, `acquire.py` to hold the following data aquisition functions:
# 
# 7. Make a function named `get_titanic_data` that returns the titanic data from the codeup data science database as a pandas data frame. Obtain your data from the _Codeup Data Science Database_. 
# 
# 
# 8. Make a function named `get_iris_data` that returns the data from the `iris_db` on the codeup data science database as a pandas data frame. The returned data frame should include the actual name of the species in addition to the `species_id`s. Obtain your data from the _Codeup Data Science Database_. 
# 
# 9. Make a function named `get_telco_data` that returns the data from the `telco_churn` database in SQL. In your SQL, be sure to join all 4 tables together, so that the resulting dataframe contains all the contract, payment, and internet service options. Obtain your data from the _Codeup Data Science Database_. 
# 
# 10. Once you've got your `get_titanic_data`, `get_iris_data`, and `get_telco_data` functions written, now it's time to add caching to them. To do this, edit the beginning of the function to check for the local filename of `telco.csv`, `titanic.csv`, or `iris.csv`. If they exist, use the .csv file. If the file doesn't exist, then produce the SQL and pandas necessary to create a dataframe, then write the dataframe to a .csv file with the appropriate name. 
# 
# __Be sure to add env.py, titanic.csv, iris.csv, and telco.csv to your .gitignore file__

# In[7]:


import pandas as pd
import numpy as np
import os
from env import host, user, password


# In[8]:


def get_connection(db, user=user, host=host,password=password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


# In[11]:


# New Titanic Data Set
def new_titanic_data():
    sql_query = 'SELECT * FROM passengers'
    df = pd.read_sql(sql_query, get_connection('titanic_db'))
    return df


# In[13]:


# Acquire Titnaic Data 
def get_titanic_data():
    filename = "titanic.csv"
    
    # if file is available locally, read it
    if os.path.isfile(filename):
        print('Using cached csv')
        return pd.read_csv(filename, index_col=0)
    
    # if file not available locally, acquire data from SQL database
    # and write it as csv locally for future use
    else:
        # read the SQL query into a dataframe
        df = new_titanic_data()
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df  


# In[15]:


### Iris Data

def new_iris_data():
    sql_query = ''' 
                SELECT 
                    species_id,
                    species_name,
                    sepal_length,
                    sepal_width,
                    petal_length,
                    petal_width
                FROM measurements
                JOIN species USING(species_id)
              '''
    df = pd.read_sql(sql_query, get_connection('iris_db'))
    return df


# In[16]:


def get_iris_data():
    filename = "iris_df.csv"
    
    # if file is available locally, read it
    if os.path.isfile(filename):
        print('Using cached csv')
        return pd.read_csv(filename, index_col=0)
    
    # if file not available locally, acquire data from SQL database
    # and write it as csv locally for future use
    else:
        # read the SQL query into a dataframe
        df = new_iris_data()
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)
        return df 


# In[17]:


### Get Telco Data
def new_telco_data():
    sql_query = '''
            select * FROM customers
            join contract_types using (contract_type_id)
            join internet_service_types using (internet_service_id)
            join payment_types using (payment_type_id)
            '''
    df=pd.read_sql(sql_query, get_connection('telco_churn'))
    return df


# In[18]:


def get_telco_data():
    
    if os.path.isfile('telco.csv'):
        print('Using cached csv')
        df = pd.read_csv('telco.csv', index_col=0)
        
    else:
        df = new_telco_data()
        df.to_csv('telco.csv')
        
    return df
