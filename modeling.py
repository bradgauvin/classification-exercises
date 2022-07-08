#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Data Handling
import pandas as pd
import numpy as np
# Acquire data sets
import acquire as acq
#Prepare data
import prepare as pp
# Modeling
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# For modeling, you need to set a target & features as in below (from titanic dataset):
# target = 'survived'
# features = ['pclass','sibsp','parch','fare','alone','sex_male','embark_town_Queenstown','embark_town_Southampton']']

## Splitter 
def train_validate_test_split(df, target, seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       stratify=train_validate[target])
    return train, validate, test

# train validate test splitting between target and features
def x_and_y(train, validate, test, features, target):
    x_train = train[features]
    y_train = train[target]
    
    x_validate = validate[features]
    y_validate = validate[target]
    
    x_test = test[features]
    y_test = test[target]
    
    return {'x_train': x_train, 'y_train': y_train, 'x_validate': x_validate, 'y_validate': y_validate,
            'x_test': x_test, 'y_test': y_test}
 
#------------------------------------------#    

# Decision Tree
def decision_tree(train, d, print_results, selected_features, target):
    
    X_train = split_data['x_train']
    y_train = split_data['y_train']
    clf = DecisionTreeClassifier(max_depth=d, random_state=123)
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_train)
    if print_results:
        print("TRAINING RESULTS")
        print("----------------")
        print(f"Max depth: {clf.max_depth}")
        print(f"Features: {selected_features}")
        print(f"Target: {target}")
        print(f"Accuracy score on training set is: {clf.score(split_data['x_train'],split_data['y_train']):.2%}")
        print(classification_report(y_train, y_pred))

        tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()

        print(f"False positive rate: {fp/(fp+tn):.2%}")
        print(f"False negative rate: {fn/(fn+tp):.2%}")
        print(f"True positive rate: {tp/(tp+fn):.2%}")
        print(f"True negative rate: {tn/(fp+tn):.2%}")
        print("----------------")
    
    return clf

    
#Validate results for decision tree
def dt_validate_results(d):
    clf = decision_tree(train, d = d, print_results = False)
    print('')
    print(f'For decision tree of depth: {clf.max_depth}')
    print('VALIDATE RESULTS')
    print('Accuracy of Decision Tree classifier on validate set: {:.2f}'
         .format(clf.score(x_validate, y_validate)))
    # And since accuracy isn't everything

    # Produce y_predictions that come from the X_validate
    y_pred = clf.predict(x_validate)

    # Compare actual y values (from validate) to predicted y_values from the model run on X_validate
    print(classification_report(y_validate, y_pred))
    
#------------------------------------------# 
# Random Forest
def random_forest(train, validate, features , target, min_samples_leaf, d, print_results = True):
    
    x_train = train[features]
    y_train = train[target]
    rf = RandomForestClassifier(max_depth=d, min_samples_leaf=min_samples_leaf, random_state=123)
    # Fit
    rf = rf.fit(X_train, y_train[target])
    
    # Predict
    y_pred = rf.predict(x_train)
    
    # Results
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
    if print_results:
        print("TRAINING RESULTS")
        print("----------------")
        # Feature importance
        print(f"Feature importance:\n{dict(zip(selected_features,rf.feature_importances_))}")
        print(f"Accuracy of random forest classifer on training set: {rf.score(x_train, y_train):.2%}")
        print(classification_report(y_train, y_pred))

        
        print("Confusion matrix: rows are truth, columns are pred")
        print("")
        print(confusion_matrix(y_train, y_pred))
        print("")
        print(f"False positive rate: {fp/(fp+tn):.2%}")
        print(f"False negative rate: {fn/(fn+tp):.2%}")
        print(f"True positive rate: {tp/(tp+fn):.2%}")
        print(f"True negative rate: {tn/(fp+tn):.2%}")
        print("----------------")
    train_report = classification_report(y_train, y_pred, output_dict=True)
    ### Predict for Validate 
    y_pred_val = rf.predict(validate[selected_features])
    ### Classification report
    validate_report = classification_report(validate[target],y_pred_val, output_dict=True)
    if print_results:
        print("VALIDATE RESULTS")
        print("----------------")
        print(classification_report(validate[target],y_pred_val))
    reports = {'train':train_report,'validate':validate_report}
    return reports


# Random forest ananlysis data points to be copy and pasted in notebook

# depths =[]
# min_samples = []
# accuracies = []
# precisions = []
# recalls = []
# v_accuracies = []
# v_precisions = []
# v_recalls = []

# for depth in range(10,1,-1):
#     for min_sample in range(1,10):
#         depths.append(depth)
#         min_samples.append(min_sample)
#         reports = random_forest(train, validate, features, min_sample, depth, False)
        
#         train_report = reports['train']
#         accuracies.append(train_report['accuracy'])
#         precisions.append(train_report['1']['precision'])
#         recalls.append(train_report['1']['recall'])
        
#         validate_report = reports['validate']
#         v_accuracies.append(validate_report['accuracy'])
#         v_precisions.append(validate_report['1']['precision'])
#         v_recalls.append(validate_report['1']['recall'])
        
# train_results_df= pd.DataFrame(data = {"max_depth":depths,"min_samples_leaf":min_samples,"accuracy":accuracies,
#                                        "precision":precisions,"recall":recalls})
# validate_results_df= pd.DataFrame(data = {"max_depth":depths,"min_samples_leaf":min_samples,
#                                           "accuracy":v_accuracies,"precision":v_precisions,"recall":v_recalls})

# _, ax = plt.subplots(1,3, figsize=(16,6))
# to_plot = ["accuracy","precision","recall"]
# for i, metric in enumerate(to_plot):
#     heatmap_df = train_results_df.pivot("max_depth","min_samples_leaf",metric)
#     sns.heatmap(heatmap_df, ax=ax[i])
#     plt.suptitle('Train performance')
#     ax[i].set_title(metric)
    
# _, ax = plt.subplots(1,3, figsize=(16,6))
# to_plot = ["accuracy","precision","recall"]
# for i, metric in enumerate(to_plot):
#     heatmap_df = validate_results_df.pivot("max_depth","min_samples_leaf",metric)
#     sns.heatmap(heatmap_df, ax=ax[i])
#     plt.suptitle('Validate performance')
#     ax[i].set_title(metric)
    
# combined_df = train_results_df.merge(validate_results_df,
#                                      on=['max_depth','min_samples_leaf'], suffixes=['_train','_validate'])

# combined_df["accuracy_diff"] = combined_df.accuracy_validate-combined_df.accuracy_train
# combined_df["precision_diff"] = combined_df.precision_validate-combined_df.precision_train
# combined_df["recall_diff"] = combined_df.recall_validate-combined_df.recall_train

# _, ax = plt.subplots(1,3, figsize=(16,6))
# to_plot = ["accuracy_diff","precision_diff","recall_diff"]
# for i, metric in enumerate(to_plot):
#     heatmap_df = combined_df.pivot("max_depth","min_samples_leaf",metric)
#     sns.heatmap(heatmap_df, ax=ax[i])
#     plt.suptitle('Difference between validate and train performance')
#     ax[i].set_title(metric)
#------------------------------------------# 
#K Nearest Neighbor

def k_nearest(train, features, target, k, validate = None, print_results = True):
    
    x_train = train[features]
    y_train = train[target]
    knn = KNeighborsClassifier(n_neighbors = k)
    # Fit
    knn = knn.fit(x_train, y_train)
    
    # Predict
    y_pred = knn.predict(x_train)
    
    # Results
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
    if print_results:
        print("----------------")
        print(f"USING k OF {knn.n_neighbors}")
        print("TRAINING RESULTS")
        print("----------------")
        # Feature importance
        # print(f"Feature importance:\n{dict(zip(selected_features,knn.feature_importances_))}")
        
        print(f"Accuracy of k-nearest neighbors classifer on training set: {knn.score(x_train, y_train):.2%}")
        print(classification_report(y_train, y_pred))

        
        print("Confusion matrix: rows are truth, columns are pred")
        print("")
        print(confusion_matrix(y_train, y_pred))
        print("")
        print(f"False positive rate: {fp/(fp+tn):.2%}")
        print(f"False negative rate: {fn/(fn+tp):.2%}")
        print(f"True positive rate: {tp/(tp+fn):.2%}")
        print(f"True negative rate: {tn/(fp+tn):.2%}")
        print("----------------")
    train_report = classification_report(y_train, y_pred, output_dict=True)
    if validate is not None:
        ### Predict for Validate 
        y_pred_val = knn.predict(validate[selected_features])
        ### Classification report
        validate_report = classification_report(validate[target],y_pred_val, output_dict=True)
        if print_results:
            print("----------------")
            print(f"USING k OF {knn.n_neighbors}")
            print("VALIDATE RESULTS")
            print("-------KNN---------")
            print(classification_report(validate[target],y_pred_val))
    else:
        validate_report = None
    reports = {'train':train_report,'validate':validate_report}
    return reports


#Performance data data points to be copy and pasted in notebook
# ks =[]
# accuracies = []
# precisions = []
# recalls = []
# v_accuracies = []
# v_precisions = []
# v_recalls = []
# class_to_analyze = '1'
# analysis_range = range(1,21,1)

# for k in analysis_range:
#     ks.append(k)
#     reports = k_nearest(train, features, target, k, validate,  False)

#     train_report = reports['train']
#     accuracies.append(train_report['accuracy'])
#     precisions.append(train_report[class_to_analyze]['precision'])
#     recalls.append(train_report[class_to_analyze]['recall'])

#     validate_report = reports['validate']
#     v_accuracies.append(validate_report['accuracy'])
#     v_precisions.append(validate_report[class_to_analyze]['precision'])
#     v_recalls.append(validate_report[class_to_analyze]['recall'])
        
# train_results_df= pd.DataFrame(data = {"k":ks,"accuracy":accuracies,"precision":precisions,"recall":recalls})
# validate_results_df= pd.DataFrame(data = {"k":ks,"accuracy":v_accuracies,"precision":v_precisions,
#                                           "recall":v_recalls})

# # Combine train and validate results to allow for plotting together
# combined_df = train_results_df.merge(validate_results_df,on=['k'], suffixes=['_train','_validate'])
# combined_df["accuracy_diff"] = combined_df.accuracy_validate-combined_df.accuracy_train
# combined_df["precision_diff"] = combined_df.precision_validate-combined_df.precision_train
# combined_df["recall_diff"] = combined_df.recall_validate-combined_df.recall_train

# # Melt metrics into same column to enable clean plotting with seaborn
# data = pd.melt(combined_df.drop(columns = ['accuracy_diff','precision_diff','recall_diff']),
#                id_vars =['k'], var_name='metric')

# _, ax = plt.subplots(1,3, figsize=(16,6))
# to_plot = ["accuracy","precision","recall"]
# for i, metric in enumerate(to_plot):
#     sns.lineplot(x = data[data.metric.str.contains(metric)].k, 
#                  y = data[data.metric.str.contains(metric)].value, 
#                  hue = data[data.metric.str.contains(metric)].metric,  
#                  ax=ax[i])
    
#     plt.suptitle(f'KNN Train and validate performance \nPrecision and recall based on class {class_to_analyze}')
#     plt.tight_layout()
#     #plt.xticks([analysis_range])
#     ax[i].legend(title = 'data set')
#     ax[i].set_title(metric)
#     if metric == "accuracy":
#         ax[i].set_title('Accuracy (Baseline accuracy shown in green)')
#         ax[i].axhline('baseline_accuracy', color = 'green')

#--------------------------------#
# Logistical Regression

def logistic_regression(train, features, target, c, validate = None, test = None, print_results = True):
    
    X_train = train[features]
    y_train = train[target]
    logit = LogisticRegression(C=c,random_state=123, max_iter=1000)
    # Fit
    logit = logit.fit(x_train, y_train)
    
    # Predict
    y_pred = logit.predict(x_train)
    
    # Results
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
    if print_results:
        print("----------------")
        print(f"USING c OF {logit.C}")
        print("TRAINING RESULTS")
        print("----------------")
        # Feature importance
        print(selected_features)
        print('Coefficient: \n', logit.coef_)
        print('Intercept: \n', logit.intercept_)
        
        
        print(f"Accuracy of logistic regression classifer on training set: S{logit.score(x_train, y_train):.2%}")
        print(classification_report(y_train, y_pred))

        
        print("Confusion matrix: rows are truth, columns are pred")
        print("")
        print(confusion_matrix(y_train, y_pred))
        print("")
        print(f"False positive rate: {fp/(fp+tn):.2%}")
        print(f"False negative rate: {fn/(fn+tp):.2%}")
        print(f"True positive rate: {tp/(tp+fn):.2%}")
        print(f"True negative rate: {tn/(fp+tn):.2%}")
        print("----------------")
    train_report = classification_report(y_train, y_pred, output_dict=True)
    if validate is not None:
        ### Predict for Validate 
        y_pred_val = logit.predict(validate[selected_features])
        ### Classification report
        validate_report = classification_report(validate[target],y_pred_val, output_dict=True)
        if print_results:
            print("----------------")
            print(f"USING c OF {logit.C}")
            print("VALIDATE RESULTS")
            print("-------Logistic Regression---------")
            print(classification_report(validate[target],y_pred_val))
    else:
        validate_report = None
        
    ## Test set performance
    if test is not None:
        ### Predict for Validate 
        y_pred_test = logit.predict(test[selected_features])
        ### Classification report
        test_report = classification_report(test[target],y_pred_test, output_dict=True)
        if print_results:
            print("----------------")
            print(f"USING c OF {logit.C}")
            print("TEST RESULTS")
            print("-------Logistic Regression---------")
            print(classification_report(test[target],y_pred_test))
    else:
        test_report = None
    reports = {'train':train_report,'validate':validate_report, 'test':test_report}
    return reports

# Performance Report 


def plot_train_validate_performance(train_results_df, validate_results_df):
    # Combine train and validate results to allow for plotting together
    combined_df = train_results_df.merge(validate_results_df,on=['c'], suffixes=['_train','_validate'])
    combined_df["accuracy_diff"] = combined_df.accuracy_validate-combined_df.accuracy_train
    combined_df["precision_diff"] = combined_df.precision_validate-combined_df.precision_train
    combined_df["recall_diff"] = combined_df.recall_validate-combined_df.recall_train
    
    # Melt metrics into same column to enable clean plotting with seaborn
    data = pd.melt(combined_df.drop(columns = ['accuracy_diff','precision_diff','recall_diff']),
                   id_vars =['c'], var_name='metric')
    
    # Plot train and validate performance
    _, ax = plt.subplots(1,3, figsize=(16,6))
    to_plot = ["accuracy","precision","recall"]
    for i, metric in enumerate(to_plot):
        sns.lineplot(x = data[data.metric.str.contains(metric)].c, 
                     y = data[data.metric.str.contains(metric)].value, 
                     hue = data[data.metric.str.contains(metric)].metric,  
                     ax=ax[i])

        plt.suptitle('Train and validate performance')
        plt.tight_layout()

        ax[i].legend(title = 'data set')
        ax[i].set_title(metric)
        # if metric == "accuracy":
        #     ax[i].set_title('Accuracy (Baseline accuracy shown in green)')
        #     ax[i].axhline(baseline_accuracy, color = 'green')
    
    # Plot performance difference between train and validate
    _, ax = plt.subplots(1,3, figsize=(16,6))
    to_plot = ["accuracy_diff","precision_diff","recall_diff"]
    for i, metric in enumerate(to_plot):
        sns.lineplot(x = combined_df.c, y = combined_df[metric], ax=ax[i])
        plt.suptitle('Difference between validate and train performance')
        ax[i].set_title(metric)
        plt.tight_layout()
    
    return combined_df

# Logistical regression data points to be copy and pasted in notebook
# cs =[]
# accuracies = []
# precisions = []
# recalls = []
# v_accuracies = []
# v_precisions = []
# v_recalls = []
# c_values = [0.01, 0.1, 1, 10, 100, 1000]

# for c in np.arange(0.01, 2.00, 0.1):
#     cs.append(c)
#     reports = logistic_regression(train, features, target, c, validate, None, False)

#     train_report = reports['train']
#     accuracies.append(train_report['accuracy'])
#     precisions.append(train_report['1']['precision'])
#     recalls.append(train_report['1']['recall'])

#     validate_report = reports['validate']
#     v_accuracies.append(validate_report['accuracy'])
#     v_precisions.append(validate_report['1']['precision'])
#     v_recalls.append(validate_report['1']['recall'])
        
# train_results_df= pd.DataFrame(data = {"c":cs,"accuracy":accuracies,"precision":precisions,"recall":recalls})
# validate_results_df= pd.DataFrame(data = {"c":cs,"accuracy":v_accuracies,"precision":v_precisions,
#                                           "recall":v_recalls})

