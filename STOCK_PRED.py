import pandas as pd
import datetime as dt
import pandas_datareader.data as web

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor



start = dt.datetime(2018,1,1)



stockList = ['PETR4.SA']

df = pd.DataFrame()
for i in stockList:
    df = web.DataReader(i, 'yahoo', start)


num_days = 5
df['prediction'] = df['Adj Close'].shift(-num_days)
df.dropna(inplace=True)



X = np.array(df.drop(['prediction'], 1))
Y = np.array(df['prediction'])
X_proc = preprocessing.scale(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8)




#Performing the Regression on the training data
clf = RandomForestRegressor()
clf.fit(X_train, Y_train)
prediction = (clf.predict(X))

print('Score train {}'.format(clf.score(X_train,Y_train)))
print('Score test {}'.format(clf.score(X_test,Y_test)))
print('Score full {}'.format(clf.score(X,Y)))


dt_pred = pd.DataFrame()

dt_pred['X'] = X[:,0]
dt_pred['Y_atual'] = Y
dt_pred['Y_pred'] = prediction
dt_aux = pd.DataFrame()
dt_aux['Date'] = df.index
dt_pred['Date'] = dt_aux.iloc[-dt_pred.shape[0]:]
dt_pred.dropna(inplace = True)


last_x_days = 90
plt.figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(dt_pred['Date'].iloc[-last_x_days:], dt_pred['Y_atual'].iloc[-last_x_days:], color = 'red')
plt.plot(dt_pred['Date'].iloc[-last_x_days:], dt_pred['Y_pred'].iloc[-last_x_days:])
plt.show()




