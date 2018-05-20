
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:


ad_data=pd.read_csv('advertising.csv')


# In[3]:


ad_data.head()


# In[4]:


ad_data.describe()


# In[5]:


ad_data.info()


# ## Exploratory Data Analysis
# 
# 

# In[8]:


sns.set_style('whitegrid')
ad_data['Age'].hist(bins=30)
plt.xlabel('Age')


# In[13]:


sns.jointplot(data=ad_data,x='Age',y='Area Income')


# In[15]:


sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,color='red',kind='kde');


# In[16]:


sns.jointplot(data=ad_data,x='Daily Time Spent on Site',y='Daily Internet Usage')


# ** Finally, create a pairplot with the hue defined by the 'Clicked on Ad' column feature.**

# In[23]:


sns.pairplot(ad_data,hue='Clicked on Ad',palette='bwr')


# # Logistic Regression
# 
# 

# ** Split the data into training set and testing set using train_test_split**

# In[24]:


from sklearn.model_selection import train_test_split


# In[30]:


X=ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y=ad_data[['Clicked on Ad']]


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# ** Train and fit a logistic regression model on the training set.**

# In[41]:


from sklearn.linear_model import LogisticRegression


# In[43]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# ## Predictions and Evaluations
# 

# In[44]:


predictn=logmodel.predict(X_test)


# In[47]:


from sklearn.metrics import classification_report


# In[49]:


print(classification_report(y_test,predictn))

