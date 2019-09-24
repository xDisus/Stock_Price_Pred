#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from tqdm import tqdm


# In[62]:


class stockPred:
    def __init__(self,starting_date):
        self.df = pd.DataFrame()
        self.start = dt.datetime.strptime(starting_date, "%d/%m/%y")

    
    def stockAdd(self,stock):
        self.df_aux = pd.DataFrame()
        self.df_aux = web.DataReader(stock, 'yahoo', self.start)
        nome_coluna = []
        for x in self.df_aux.columns:
            nome_coluna.append(str(stock)+'_'+str(x))
        self.df_aux.columns=nome_coluna
        self.df = pd.concat([self.df, self.df_aux], axis=1)
        return self.df
    
    
    def predDays(self,stock,days):
        self.days = days
        self.stock = stock
        self.df[str(stock)+'_shift'+str(days)] = self.df[str(stock)+'_Adj Close'].shift(-days)
        self.df_dropna = self.df.dropna()
        return self.df, self.df_dropna
    
    def trainTest(self):
        self.X_dropna = self.df_dropna.drop([str(self.stock)+'_shift'+str(self.days)], 1)
        self.X = self.df.drop([str(self.stock)+'_shift'+str(self.days)], 1)
        self.Y_dropna = self.df_dropna[str(self.stock)+'_shift'+str(self.days)]
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X_dropna, self.Y_dropna, test_size=0.8)
        return self.X_dropna, self.X
    
    def date_by_adding_business_days(self):
        business_days_to_add = self.days
        current_date = self.base_date
        while business_days_to_add > 0:
            current_date += dt.timedelta(days=1)
            weekday = current_date.weekday()
            if weekday >= 5: # sunday = 6
                continue
            business_days_to_add -= 1
        return current_date

        
    def trainModel(self,base_date,verbose = 0):
        self.base_date = dt.datetime.strptime(base_date, "%d/%m/%y")
        self.clf = RandomForestRegressor(max_depth=100, random_state=0,n_estimators=300, n_jobs = -1)
        self.clf.fit(self.X_train, self.Y_train)
        self.prediction = (self.clf.predict(self.X.loc[self.base_date:self.base_date]))
        self.pred_day = self.date_by_adding_business_days()
        if verbose == 1:
            print('Score train {}'.format(self.clf.score(self.X_train,self.Y_train)))
            print('Score test {}'.format(self.clf.score(self.X_test,self.Y_test)))
            print('Score full {}'.format(self.clf.score(self.X_dropna,self.Y_dropna)))
        return self.clf, self.prediction, self.pred_day
        
        


# In[41]:


#Stating the class, setting starting date, building db

x = stockPred('01/01/19')
y = x.stockAdd('JBSS3.SA')
#y = x.stockAdd('^BVSP')


# In[64]:


#predicting stock pricing for the next N days

x = stockPred('01/01/15')
y = x.stockAdd('JBSS3.SA')

pred_list = []
day_list = []
for i in tqdm(range(1,3)):
    x.predDays('JBSS3.SA',i)
    x.trainTest()
    a,pred,dia = x.trainModel('20/09/19',verbose =0)
    pred_list.append(pred)
    day_list.append(dia)
    
for a,b in zip(day_list,pred_list):
    print('Data:{} - Valor {}'.format(a,b))


# In[54]:


#y = x.stockAdd('JBSS3.SA')


# In[13]:


w = y['JBSS3.SA_Adj Close'].loc['2019-09-16':'2019-09-23']


# In[14]:


#w['Pred'] = pred_list
w.head(10)


# In[ ]:




