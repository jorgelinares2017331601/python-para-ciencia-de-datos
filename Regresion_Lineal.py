#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


# In[5]:


iris=sb.load_dataset("iris")
iris.head()


# In[6]:


iris.describe()


# In[7]:


sb.pairplot(iris)


# In[8]:


from sklearn.model_selection import train_test_split
x=iris[["petal_length"]]
y=iris[["petal_width"]]
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.30,random_state=42)


# In[9]:


from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(x_train,y_train)


# In[10]:


print(lm.intercept_)


# In[11]:


lm.coef_


# In[12]:


predicciones=lm.predict(x_test)


# In[13]:


predicciones


# In[16]:


plt.scatter(y_test,predicciones)
plt.plot([0,2.5],[0,2.5])
plt.show()


# In[18]:


plt.scatter(x_test,predicciones)


# mertricas

# In[21]:


from sklearn import metrics
print("MAE",metrics.mean_absolute_error(y_test,predicciones))
print("MSE",metrics.mean_squared_error(y_test,predicciones))
print("RMSE",np.sqrt(metrics.mean_squared_error(y_test,predicciones)))


# In[ ]:




