#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Prep Iris data
def prep_iris(iris):
    iris = acquire.get_iris_data()
    iris2 =iris.drop(['species_id'], axis=1)
    iris3 = iris2.rename({'species_name':'species'}, axis=1)
    dummy_iris = pd.get_dummies(iris3[['species']], dummy_na=False, drop_first=[True])
    iris4 = pd.concat([iris3,dummy_iris], axis=1)
    return iris4

# Prep Titanic data
def prep_titanic(titanic):
    '''
    This function will clean the data...
    '''
    titanic = acquire.get_titanic_data()
    titanic = titanic.drop_duplicates()
    cols_to_drop = ['deck', 'embarked', 'class', 'age']
    titanic2 = titanic.drop(columns=cols_to_drop)
    titanic2['embark_town'] = titanic2.embark_town.fillna(value='Southampton')
    dummy_df = pd.get_dummies(titanic2[['sex', 'embark_town']], dummy_na=False, drop_first=[True, True])
    titanic3 = pd.concat([titanic2, dummy_df], axis=1)
    print('Data cleaned for duplicates, columns dropped [deck, embarked, class, age], filled na, and added numerical versions of sex and embark')
    return titanic3

# Prep Telco data
def prep_telco():
    telco=acquire.get_telco_data()
    # Drop Duplicates
    telco.drop(columns=['payment_type_id','internet_service_type_id','contract_type_id','customer_id'], inplace=True)
    # Get rid of whitespace
    telco.total_charges.replace(to_replace=[' ',''],value = np.nan,inplace=True)
    telco = telco[telco.total_charges != '']
    # Convert to float
    telco['total_charges'] = telco.total_charges.astype(float)
    
    # adding categorical variables to numeric
    telco['gender_binary'] = telco.gender.map({'Female':1, 'Male': 0})
    telco['partner_binary'] = telco.partner.map({'Yes':1,'No': 0})
    telco['dependents_binary'] = telco.dependents.map({'Yes':1, 'No':0})
    telco['phone_service_binary'] = telco.phone_service.map({'Yes':1, 'No':0})
    telco['paperless_billing_encoded'] = telco.paperless_billing.map({'Yes': 1, 'No': 0})
    telco['churn_encoded'] = telco.churn.map({'Yes': 1, 'No': 0})
    
    return telco

