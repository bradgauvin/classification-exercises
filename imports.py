#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from env import get_db_url
import os

# python data science library's
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


# visualizations
from pydataset import data
import matplotlib.pyplot as plt
import seaborn as sns


# state properties
np.random.seed(123)
pd.set_option("display.max_columns", None)

# warnings ignore
import warnings
warnings.filterwarnings("ignore")

