#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from sklearn.metrics import r2_score,roc_auc_score,classification_report,mean_squared_error,accuracy_score,confusion_matrix
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
import warnings
warnings.filterwarnings('ignore')


# In[2]:


train=pd.read_csv('D:/R/Hr analytics new/train.csv')
test=pd.read_csv('D:/R/Hr analytics new/test.csv')


# In[3]:


#Combine test and train into one file
master = pd.concat([train, test],ignore_index=True)
print(train.shape, test.shape, master.shape)
master.head()


# In[4]:


master.info()


# In[5]:


#CHECK NA
master.isnull().sum()/len(master)*100


# In[6]:


# CHECK UNIQUE VALUES

master.apply(lambda x : len(x.unique()))


# In[7]:


#2.GENDER

sns.catplot(x="gender", kind="count", data=master);
master['gender']=master['gender'].fillna('Male')
gf={'Other':0,'Female':1,'Male':2}
master.gender=master.gender.map(gf)
master.gender=master.gender.astype(int)
master['gender'].value_counts()


# In[8]:


#3.ENROLLED UNIVERSITY


sns.catplot(x="enrolled_university", kind="count", data=master);
master['enrolled_university']=master['enrolled_university'].fillna('no_enrollment')
gs={'no_enrollment':0,'Full time course':1,'Part time course':2}
master.enrolled_university=master.enrolled_university.map(gs)
master.enrolled_university=master.enrolled_university.astype(int)
master['enrolled_university'].value_counts()


# In[9]:


#4.EDUCATION LEVEL


sns.catplot(x="education_level", kind="count", data=master);
master['education_level']=master['education_level'].fillna('Graduate')
ge={'Primary School':0,'Phd':1,'High School':2,'Masters':3,'Graduate':4}
master.education_level=master.education_level.map(ge)
master.education_level=master.education_level.astype(int)
master['education_level'].value_counts()


# In[10]:


#5.Major Discipline

sns.countplot(x='major_discipline',data=master,hue="target")
sns.catplot(x="major_discipline", kind="count", data=master);
master['major_discipline']=master['major_discipline'].fillna('STEM')
gt={'No Major':0,'Arts':1,'Business Degree':2,'Other':3,'Humanities':4,'STEM':5}
master.major_discipline=master.major_discipline.map(gt)
master.major_discipline=master.major_discipline.astype(int)
master['major_discipline'].value_counts()


# In[11]:


#6.EXPERIENCE
master['experience'].value_counts()
sns.countplot(x='experience',data=master,hue="target")
master['experience'].replace({'<1':'1','>20':'20'},inplace=True)
master['experience']=master['experience'].fillna('20')
master['experience']=master['experience'].astype('int')


# In[12]:


#7.COMPANY SIZE
master['company_size'].value_counts()
master['company_size'].replace({'<10':'10','10/49':'30','50-99':'75','100-500':'300','500-999':'750',
                            '1000-4999':'3000','5000-9999':'7500','10000+':'10000'},inplace=True)
sns.catplot(x="company_size", kind="count", data=master);
master['company_size']=master['company_size'].fillna('75')
sns.countplot(x='company_size',data=master,hue="target")
master['company_size']=master['company_size'].astype('int')


# In[13]:


#8.LAST_NEWJOB
master['last_new_job'].value_counts()
sns.catplot(x="last_new_job", kind="count", data=master);
sns.countplot(x='last_new_job',data=master,hue="target")
master['last_new_job'].replace({'>4':4,'never':0},inplace=True)
master['last_new_job']=master['last_new_job'].fillna(1)
master['last_new_job']=master['last_new_job'].astype('int')


# In[14]:


#9.COMPANY_TYPE

sns.catplot(x="company_type",kind="count",data=master)
sns.countplot(x="company_type",data=master,hue="target")
master['company_type']=master['company_type'].fillna('Pvt Ltd')
gy={'Other':0,'NGO':1,'Early Stage Startup':2,'Public Sector':3,'Funded Startup':4,'Pvt Ltd':5}
master.company_type=master.company_type.map(gy)
master.company_type=master.company_type.astype(int)
master['company_type'].value_counts()


# In[15]:


#TARGET VARIABLE #IMBALANCE DATASET
master['target'].value_counts(normalize=True)*100


# In[16]:


#REGION
master.city=master.city.str.extract('(\d+)')
master.city=master.city.astype(int)
master.head()


# In[17]:


#RELEVTNT

sns.countplot(x="relevent_experience",data=master,hue="target")
gs={'Has relevent experience':1,'No relevent experience':0}
master.relevent_experience=master.relevent_experience.map(gs)
master.relevent_experience=master.relevent_experience.astype(int)
master.relevent_experience.value_counts()


# In[21]:


# CITY INDEX # Negative Skewness
sns.distplot(master.city_development_index)
master.city_development_index.skew()


# In[27]:


#BOXPLOT WITH DISTPLOT
f,(ax_box,ax_hist)=plt.subplots(2,sharex=True,gridspec_kw={'height_ratios':(.15,.85)})
sns.distplot(master.training_hours,ax=ax_hist)
sns.boxplot(master.training_hours,ax=ax_box)
ax_box.set(yticks=[])
sns.despine(ax=ax_hist)
sns.despine(ax=ax_box,left=True)


# In[18]:


##FEATURE ENGINERING##
master['Texp_Till']=master.experience + master.last_new_job
master.head()


# In[19]:



contvars=master[['city_development_index','experience','training_hours','Texp_Till','last_new_job','city']]
#####CORRELATION MATRIX######
#correlation matrix
corrmat = contvars.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8,annot = True,square=True);


# In[ ]:





# In[ ]:





# In[ ]:




