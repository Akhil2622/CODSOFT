#!/usr/bin/env python
# coding: utf-8

# # Task 1: Titanic Survival Prediction

# In[1]:


import numpy as pd 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df= pd.read_csv("tested.csv")


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


df['Survived'].value_counts()


# In[8]:


sns.countplot(x=df['Survived'], hue=df['Pclass'])


# In[9]:


df['Sex']


# In[10]:


sns.countplot(x=df['Sex'], hue=df['Survived'])


# In[11]:


df.groupby('Sex')[['Survived']].mean()


# In[12]:


df['Sex'].unique()


# In[13]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df['Sex']=labelencoder.fit_transform(df['Sex'])
df.head(100)


# In[14]:


df['Sex'], df['Survived']


# In[15]:


sns.countplot(x=df['Sex'], hue=df['Survived'])


# In[16]:


df.isna().sum()


# In[17]:


df_final =df
df_final.head(10)


# In[18]:


X=df[['Pclass','Sex']]
Y=df['Survived']


# In[19]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size =0.2, random_state=0)


# In[20]:


from sklearn.linear_model import LogisticRegression
log=LogisticRegression(random_state=0)
log.fit(X_train, Y_train)


# # Model Prediction
# 

# In[21]:


pred =print(log.predict(X_test))


# In[23]:


pred=print((X_test))


# In[24]:


pred=print((Y_test))


# In[29]:


import warnings
warnings.filterwarnings("ignore")

res= log.predict([[2,1]])

if(res==0):
    print("Sorry! Not Survived")
else:
        print("Survived")


# In[ ]:




