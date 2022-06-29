import pandas as pd
import numpy as np
import os
from env import host, user, password


def get_connection(db, user=user, host=host,password=password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


# New Titanic Data Set
def new_titanic_data():
    sql_query = 'SELECT * FROM passengers'
    df = pd.read_sql(sql_query, get_connection('titanic_db'))
    return df

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

def get_telco_data():
    
    if os.path.isfile('telco.csv'):
        print('Using cached csv')
        df = pd.read_csv('telco.csv', index_col=0)
        
    else:
        df = new_telco_data()
        df.to_csv('telco.csv')
        
    return df
