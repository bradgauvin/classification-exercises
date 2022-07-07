#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from acquire import get_telco_data, get_titantic_data
from prepare import train_validate_test_split, prep_telco
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def decision_tree(df, d = 5, print_results = True):
    
    selected_features = input("Select Features: ")
    x_train = train[selected_features]
    y_train = train[['survived']]
    clf = DecisionTreeClassifier(max_depth=d, random_state=123)
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_train)
    if print_results:
        print("TRAINING RESULTS")
        print("----------------")
        print(f"Accuracy score on training set is: {clf.score(x_train, y_train):.2f}")
        print(classification_report(y_train, y_pred))

        tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()

        print(f"False positive rate: {fp/(fp+tn):.2%}")
        print(f"False negative rate: {fn/(fn+tp):.2%}")
        print(f"True positive rate: {tp/(tp+fn):.2%}")
        print(f"True negative rate: {tn/(fp+tn):.2%}")
        print("----------------")
    
    return clf

def validate_results(d):
    clf = decision_tree(df, d = d, print_results = False)
    print('')
    print(f'For decision tree of depth: {clf.max_depth}')
    print('VALIDATE RESULTS')
    print('Accuracy of Decision Tree classifier on validate set: {:.2f}'
         .format(clf.score(X_validate, y_validate)))

    # Produce y_predictions that come from the X_validate
    y_pred = clf.predict(x_validate)

    # Compare actual y values (from validate) to predicted y_values from the model run on X_validate
    print(classification_report(y_validate, y_pred))

