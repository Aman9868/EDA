#!/usr/bin/env python
# coding: utf-8

# In[377]:


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
import warnings
warnings.filterwarnings('ignore')


# In[378]:


#READING
train=pd.read_csv('D:/R/hr analys/train.csv')
test=pd.read_csv('D:/R/hr analys/test.csv')


# In[379]:


# BINDING
master=pd.concat([train,test],ignore_index=True)
print(train.shape,test.shape,master.shape)
master.head()


# In[380]:


### Chck Dtypes
master.info()


# In[358]:


# Check column names
print(master.columns)


# In[381]:


# check na
master.isnull().sum()/len(master)*100


# In[382]:


# CHECK UNIQUE VALUES

master.apply(lambda x : len(x.unique()))


# In[383]:


##SEPERATION##
cat=['education','gender','recruitment_channel','department','previous_year_rating','KPIs_met80','awards_won','no_of_trainings']
num=['age','avg_training_score','length_of_service']
final=master[cat+num]


# In[384]:


#CAT VARIABLE
fig, ax=plt.subplots(3,2,figsize=(20,20))
for variable,subplot in zip(cat,ax.flatten()):
    sns.countplot(final[variable],ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)


# In[385]:


#NUM VARIABLE

fig, ax=plt.subplots(3,figsize=(20,20))
for variable,subplot in zip(num,ax.flatten()):
    sns.distplot(final[variable],ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)


# In[386]:


# VISUALISING BOXPLOT
fig, ax=plt.subplots(3,figsize=(10,15))
for variable,subplot in zip(num,ax.flatten()):
    sns.boxplot(final[variable], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)


# In[387]:



# Category vs Target
fig,axes = plt.subplots(4,2,figsize=(20,20))
for idx,cat_col in enumerate(cat):
    row,col = idx//2,idx%2
    sns.countplot(x=cat_col,data=master,hue='is_promoted',ax=axes[row,col])


plt.subplots_adjust(hspace=1)


# In[388]:


#CATEGORY VS NUMERIC vs target
g=sns.FacetGrid(master,hue="is_promoted",col='gender',size=7)
g.map(sns.distplot,"age").add_legend()
plt.show()

g=sns.FacetGrid(master,hue="is_promoted",col='gender',size=7)
g.map(sns.distplot,"avg_training_score").add_legend()
plt.show()

g=sns.FacetGrid(master,hue="is_promoted",col='gender',size=7)
g.map(sns.distplot,"length_of_service").add_legend()
plt.show()


# In[389]:


# MISSING IMPUTATIONS##

#1.EDUCATION
mis=master[master.education.isna()]
mis
master.education=master.education.fillna('Primary')
master.education.value_counts()


# In[390]:


#2.PREVIOUS YEAR RATING
misd=master[master.previous_year_rating.isna()]
misd
master.previous_year_rating=master.previous_year_rating.fillna(0)
master.previous_year_rating.value_counts()


# In[391]:


###########OUTLIER TREATMENTS#############

#AGE
sorted(master.age)
quantile1,quantile3=np.percentile(master.age,[25,75])

#IQR
iqr=quantile3-quantile1
print(iqr)
#UPPER AND LOWER BOUND
lb=quantile1 -(1.5 * iqr)
up=quantile3 +(1.5 * iqr)
print(lb,up)

# TREATMENT
master.age.loc[master.age > up]=up
sns.boxplot(master['age'])


# In[392]:


# length of esrvice

sorted(master['length_of_service'])
quantile1,quantile3=np.percentile(master.length_of_service,[25,75])

#IQR
iqr=quantile3-quantile1
print(iqr)
#UPPER AND LOWER BOUND
lb=quantile1 -(1.5 * iqr)
up=quantile3 +(1.5 * iqr)
print(lb,up)

# TREATMENT
master.length_of_service.loc[master.length_of_service > up]=up
sns.boxplot(master['length_of_service'])


# In[393]:


#### FEATURE ENGINEERING#########
master['tot_traning']=master.no_of_trainings*master.avg_training_score
master['performance']=master.awards_won + master.KPIs_met80
master['starts_at']=master.age-master.length_of_service
master['work_frac']=master.length_of_service / master.age


# In[394]:


###MAPPING#####3
#1.Gender
g={'f':0,'m':1}
master.gender=master.gender.map(g)
master.gender=master.gender.astype(int)
master.gender.value_counts()


# In[395]:


e={'Sales & Marketing':0,'Operations':1,'Procurement':2,'Technology':3,'Analytics':4,'Finance':5,'HR':6,'Legal':7,'R&D':8}
master.department=master.department.map(e)
master.department=master.department.astype(int)
master.department.value_counts()


# In[396]:


# recruitment channel
d={'other':0,'sourcing':1,'referred':2}
master.recruitment_channel=master.recruitment_channel.map(d)
master.recruitment_channel=master.recruitment_channel.astype(int)
master.recruitment_channel.value_counts()


# In[397]:


master.info()

contvars=master[['work_frac','age','starts_at','avg_training_score','tot_traning','length_of_service','performance','awards_won','previous_year_rating']]
#####CORRELATION MATRIX######
#correlation matrix
corrmat = contvars.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8,annot = True,square=True);


# In[398]:


#REGION
master.region=master.region.str.extract('(\d+)')
master.region=master.region.astype(int)
master.head()


# In[ ]:




