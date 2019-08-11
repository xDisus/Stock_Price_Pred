#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import datetime as dt
import pandas_datareader.data as web

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import os
import quandl


# In[28]:


start = dt.datetime(2018,1,1)



stockList = ['^BVSP','PETR4.SA']

df = pd.DataFrame()
df_aux = pd.DataFrame()
for i in stockList:
    df_aux = web.DataReader(i, 'yahoo', start)
    nome_coluna = []
    for x in df_aux.columns:
        nome_coluna.append(str(i)+'_'+str(x))
    df_aux.columns=nome_coluna
    df = pd.concat([df, df_aux], axis=1)


# In[29]:


df.head()


# In[30]:


num_days = 1
df['prediction'] = df['PETR4.SA_Adj Close'].shift(-num_days)
df.dropna(inplace=True)


# In[31]:


X = df.drop(['prediction'], 1)
#Y = np.array(df['prediction'])
#X_proc = preprocessing.scale(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, df['prediction'], test_size=0.8)


# In[32]:


df['prediction'].head()


# In[34]:


#Performing the Regression on the training data
clf = RandomForestRegressor()
clf.fit(X_train, Y_train)
prediction = (clf.predict(X))

print('Score train {}'.format(clf.score(X_train,Y_train)))
print('Score test {}'.format(clf.score(X_test,Y_test)))
print('Score full {}'.format(clf.score(X,df['prediction'])))


# In[35]:


df['pred'] = prediction
df['prediction'] = df['prediction'].shift(+num_days)
df['pred'] = df['pred'].shift(+num_days)
df['Data'] = df.index
df.tail()


# In[36]:


last_x_days = 30
plt.figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(df['Data'].iloc[-last_x_days*2:-last_x_days+2], df['PETR4.SA_Adj Close'].iloc[-last_x_days*2:-last_x_days+2], color = 'black')
plt.plot(df['Data'].iloc[-last_x_days:], df['prediction'].iloc[-last_x_days:], color = 'red')
plt.plot(df['Data'].iloc[-last_x_days:], df['pred'].iloc[-last_x_days:])
plt.show()


# In[ ]:




