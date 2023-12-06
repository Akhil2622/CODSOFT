#!/usr/bin/env python
# coding: utf-8

# # Sales Prediction 
# 

# In[11]:


import warnings as w
import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
w.filterwarnings('ignore')


# In[12]:


df = pd.read_csv("advertising.csv")


# In[13]:


df.head()


# In[14]:


df.describe()


# In[15]:


df.info()


# In[16]:


df['TV']=df['TV'].astype(int)
df['Radio']=df['Radio'].astype(int)
df['Newspaper']=df['Newspaper'].astype(int)
df['Sales']=df['Sales'].astype(int)


# In[17]:


df.isna().sum()


# In[26]:


sns.pairplot(df)


# In[27]:


sns.heatmap(df.corr(), annot=True);


# In[28]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,mean_absolute_error,mean_squared_error,r2_score)


# In[29]:


#splitting the data into x and y

X = df.iloc[:, 0:-1]
Y = df.iloc[:,-1]

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


# In[30]:


model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)


# In[31]:


print("Mean Absolute Error:", mean_absolute_error(Y_test, Y_pred))
print("Mean Squared Error:", mean_squared_error(Y_test, Y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(Y_test, Y_pred)))
print("R2 Score:", r2_score(Y_test, Y_pred))


# In[32]:


sns.regplot(x=Y_test, y=Y_pred)


# # Model predicting 

# In[33]:


model.predict([[230,44,39]])

