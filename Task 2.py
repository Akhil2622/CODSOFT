#!/usr/bin/env python
# coding: utf-8

# # IRIS FLOWER CLASSIFICATION

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[19]:


from sklearn.datasets import load_iris
iris = load_iris()
data = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])


# In[4]:


df= pd.read_csv("IRIS.csv")


# In[5]:


df.head()


# In[38]:


df.info()


# In[39]:


df.describe()


# In[20]:


X = data.drop('target', axis=1)
y = data['target']


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[24]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[25]:


knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, y_train)


# In[26]:


y_pred = knn_classifier.predict(X_test)


# In[29]:


plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# In[30]:


sns.pairplot(data, hue='target', markers=['o', 's', 'D'])
plt.show()


# In[27]:


accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)


# In[28]:


print(f"Accuracy: {accuracy}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)


# In[37]:


from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))
accuracy is 1.0
