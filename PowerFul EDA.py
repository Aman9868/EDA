#!/usr/bin/env python
# coding: utf-8

# In[4]:


#LIBRARIES
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold,GridSearchCV,RandomizedSearchCV,cross_val_score
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,AdaBoostClassifier,GradientBoostingClassifier,RandomForestRegressor,BaggingRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression,LogisticRegression,Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
import sklearn.metrics as metrics
from sklearn.metrics import r2_score,roc_auc_score,classification_report,mean_squared_error,accuracy_score,confusion_matrix,precision_score,recall_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMRegressor
import sweetviz
import warnings
warnings.filterwarnings('ignore')


# In[5]:


# READING
train = pd.read_csv("D:/R/House Prices kaggle/train.csv")
test = pd.read_csv("D:/R/House Prices kaggle/test.csv")


# In[6]:


# BINDING
master=pd.concat([train,test],ignore_index=True)
print(train.shape,test.shape,master.shape)
master.head()


# # COMPARISON WITH SALEPRICE#

# In[9]:


myrep=sweetviz.analyze([train,"Train"],target_feat='SalePrice')


# In[12]:


myrep.show_html('Vis.html')


# # TRAIN AND TEST COMPARISON WITH SALEPRICE#

# In[15]:


feature=sweetviz.FeatureConfig(skip="Id")#Remove Id Variable
myrep1=sweetviz.compare([train,"Train"],[test,"Test"],'SalePrice',feature)


# In[16]:


myrep1.show_html('Vist.html')


# # COMPARISON SUB CATEGORIES IN COLUMNS#

# In[27]:




